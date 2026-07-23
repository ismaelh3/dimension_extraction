"""
Object-agnostic glass + studio render engine (Blender + Cycles).

This is the GENERAL half of the old render_glass.py: a physically-honest ray-
traced studio that works for ANY textured/assembled .glb, plus the glass
material and light-path setup that make transparent/reflective products read
correctly. It has NO object-specific logic — snowglobe scene dressing (snow,
water, penguins) lives in scenes/snowglobe.py, and the driver is render_asset.py.

Like render_glass.py before it, this runs INSIDE Blender's bundled Python
(`blender --background --python render_asset.py`), NOT importable from the venv.

What it provides:
  * make_glass()      — a fresh Principled BSDF (transmission 1.0, IOR, low
                        roughness, dispersion) on a region; discards any baked
                        diffuse so the surface is actually see-through. Optional
                        thin-shell solidify + Laplacian relax for hollow domes.
  * build_studio()    — flat neutral world + key/fill/rim softboxes (their
                        rectangles are the glassy highlights) + a soft ground
                        plane + a 3/4 camera, all sized to the subject bbox.
  * configure_render()— Cycles with deep transmission bounces (shallow defaults
                        terminate refraction rays BLACK = murky plastic),
                        caustics, and Metal-GPU autodetect.
  * fit_sphere / replace_with_sphere — recover an ideal glass ball from a bumpy
                        carved cap (used by container/snowglobe presets).
  * add_interior()    — seat an extracted/generated interior .glb inside a
                        fitted shell, preserving its real colour.
"""

import math
import sys

import bpy
from mathutils import Vector


def set_input(node, names, value):
    """Set the first matching Principled input by name — the socket names
    drifted across Blender versions (e.g. 'Transmission' -> 'Transmission
    Weight' in 4.0), so we try a few."""
    for n in names:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return True
    return False


# ---------------------------------------------------------------------------
# scene setup
# ---------------------------------------------------------------------------
def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_asset(path):
    bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def world_bbox(objs):
    mn = Vector((math.inf,) * 3)
    mx = Vector((-math.inf,) * 3)
    for o in objs:
        for corner in o.bound_box:
            w = o.matrix_world @ Vector(corner)
            mn = Vector(map(min, mn, w))
            mx = Vector(map(max, mx, w))
    return mn, mx


def fit_sphere(objs):
    """Least-squares sphere (centre, radius) through the world-space vertices
    of `objs`. A spherical dome is recovered from its bumpy carved cap.
    Algebraic fit: x²+y²+z² = 2c·p + d."""
    import numpy as np
    pts = []
    for o in objs:
        m = o.matrix_world
        pts.extend([m @ v.co for v in o.data.vertices])
    p = np.array([[v.x, v.y, v.z] for v in pts])
    A = np.hstack([2 * p, np.ones((len(p), 1))])
    b = (p ** 2).sum(axis=1)
    cx, cy, cz, d = np.linalg.lstsq(A, b, rcond=None)[0]
    r = math.sqrt(max(d + cx * cx + cy * cy + cz * cz, 0.0))
    return Vector((cx, cy, cz)), r


def replace_with_sphere(glass_objs, name="glass_dome"):
    """Swap a ringy carved dome for an ideal icosphere fitted to it — a regular
    mesh that solidifies into a clean thin shell. Returns (sphere, centre,
    radius); removes the originals."""
    center, radius = fit_sphere(glass_objs)
    for o in glass_objs:
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5, radius=radius,
                                          location=center)
    sph = bpy.context.active_object
    sph.name = name
    print(f"[render_studio] fitted glass sphere: r={radius*100:.2f} cm  "
          f"centre=({center.x*100:.1f},{center.y*100:.1f},{center.z*100:.1f}) cm")
    return sph, center, radius


def make_glass(obj, tint, roughness, ior, thickness, smooth_iters,
               dispersion=0.0):
    mat = bpy.data.materials.new("glass_pbr")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    set_input(bsdf, ["Base Color"], (*tint, 1.0))
    set_input(bsdf, ["Metallic"], 0.0)
    set_input(bsdf, ["Roughness"], roughness)
    set_input(bsdf, ["IOR"], ior)
    set_input(bsdf, ["Transmission Weight", "Transmission"], 1.0)
    # Chromatic dispersion (Abbe): splits the refracted rays by wavelength so
    # the glass gets faint rainbow fringing at its edges — the tell that reads
    # as real cut crystal rather than a uniform-IOR CG shell. Socket exists in
    # Blender 4.x+/5.1; guarded so it no-ops on builds without it.
    if dispersion > 0:
        set_input(bsdf, ["Dispersion"], dispersion)
    mat.use_screen_refraction = True  # helps EEVEE; harmless in Cycles
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for poly in obj.data.polygons:      # smooth shading = clean refraction
        poly.use_smooth = True

    # Relax a faceted/ringy hull surface into a smooth dome. A thin glass shell
    # refracts through both walls, so facet noise the solid block used to
    # average out turns into foil-like chaos — smoothing first is what makes the
    # hollow shell read as glass. (Laplacian; a sphere is stable under it.)
    if smooth_iters > 0:
        sm = obj.modifiers.new("relax", "SMOOTH")
        sm.iterations = smooth_iters
        sm.factor = 0.5  # >0.5 is unstable (Laplacian overshoots into spikes)

    # A visual hull is SOLID, so glass on it reads as a crystal paperweight
    # (light refracts once through the whole block). A real hollow product is a
    # thin shell: light refracts at the front wall, crosses air, and refracts
    # again at the back wall. Solidify turns the outer surface into a wall of
    # `thickness`, offset -1 so the original surface stays the OUTER face.
    if thickness > 0:
        mod = obj.modifiers.new("shell", "SOLIDIFY")
        mod.thickness = thickness
        mod.offset = -1.0  # keep the original surface as the OUTER face
        mod.use_even_offset = True
        mod.use_rim = True


def add_interior(path, center, radius, fill):
    """Import an extracted/generated interior .glb and seat it inside a fitted
    shell: scale so its height is `fill`*diameter, centre on the axis, rest in
    the lower hemisphere. Preserves the interior's REAL colour — a UV image
    texture or per-vertex COLOR_0 — falling back to matte off-white only when it
    has neither."""
    from mathutils import Matrix
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    objs = [o for o in bpy.context.scene.objects
            if o not in before and o.type == "MESH"]
    if not objs:
        print(f"[render_studio] interior '{path}' imported no mesh — skipping")
        return []
    mn, mx = world_bbox(objs)
    height = mx.z - mn.z
    if height <= 0:
        return objs
    s = fill * 2 * radius / height
    pivot = Vector(((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, mn.z))  # bottom-centre
    floor = Vector((center.x, center.y, center.z - 0.45 * radius))
    M = Matrix.Translation(floor) @ Matrix.Scale(s, 4) @ Matrix.Translation(-pivot)

    def _has_image_tex(o):
        return any(slot and slot.use_nodes and any(
            n.type == "TEX_IMAGE" for n in slot.node_tree.nodes)
            for slot in o.data.materials)

    def _has_vertex_color(o):
        return len(getattr(o.data, "color_attributes", [])) > 0

    for o in objs:
        o.matrix_world = M @ o.matrix_world
        if _has_image_tex(o):
            kind = "kept baked texture"
        elif _has_vertex_color(o):
            # Build a Principled material driven by the mesh colour attribute so
            # per-vertex colours actually render (Blender assigns a bare default
            # material to COLOR_0 glbs, which ignores the colours).
            vmat = bpy.data.materials.new("interior_vcol")
            vmat.use_nodes = True
            nt = vmat.node_tree
            fb = nt.nodes.get("Principled BSDF")
            ca = nt.nodes.new("ShaderNodeVertexColor")
            layer = o.data.color_attributes[0].name
            ca.layer_name = layer
            nt.links.new(ca.outputs["Color"], fb.inputs["Base Color"])
            set_input(fb, ["Roughness"], 0.55)
            o.data.materials.clear()
            o.data.materials.append(vmat)
            kind = f"vertex colours ('{layer}')"
        else:
            fig = bpy.data.materials.new("figurine")
            fig.use_nodes = True
            fb = fig.node_tree.nodes.get("Principled BSDF")
            set_input(fb, ["Base Color"], (0.88, 0.89, 0.92, 1.0))
            set_input(fb, ["Roughness"], 0.7)
            o.data.materials.clear()
            o.data.materials.append(fig)
            kind = "matte off-white"
    print(f"[render_studio] interior material: {kind}")
    print(f"[render_studio] interior seated: scale {s:.2f}, "
          f"height {fill*2*radius*100:.1f} cm inside r={radius*100:.1f} cm sphere")
    return objs


def aim(obj, target):
    """Point an object's local -Z at target (lights and cameras both look
    down -Z)."""
    direction = (target - Vector(obj.location))
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_area(name, loc, target, size, power):
    light = bpy.data.lights.new(name, "AREA")
    light.shape = "RECTANGLE"
    light.size = size
    light.size_y = size * 0.6
    light.energy = power
    obj = bpy.data.objects.new(name, light)
    bpy.context.collection.objects.link(obj)
    obj.location = loc
    aim(obj, target)
    return obj


def build_studio(objs, az_deg=38.0, el_deg=16.0):
    """Neutral world + key/fill/rim softboxes + ground plane + 3/4 camera, all
    sized to the subject bbox. az/el place the camera (also used by presets that
    want the interior to face the lens)."""
    mn, mx = world_bbox(objs)
    center = (mn + mx) / 2
    size = max(mx - mn)

    # flat neutral world — even reflection tone + Fresnel fill
    world = bpy.data.worlds.new("studio")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.55, 0.56, 0.58, 1.0)
    bg.inputs["Strength"].default_value = 0.35
    bpy.context.scene.world = world

    # three softboxes; their rectangles are the glassy highlights. Powers scale
    # with object size (area-light irradiance falls off with distance², and the
    # studio sits a few object-widths away).
    p = size * size * 9000  # ~120 W for a 13 cm object
    add_area("key",  center + Vector((-1.3, -1.6, 2.1)) * size, center,
             size * 2.2, p)
    add_area("fill", center + Vector(( 1.7, -1.1, 0.9)) * size, center,
             size * 2.4, p * 0.35)
    add_area("rim",  center + Vector(( 0.4,  1.9, 1.4)) * size, center,
             size * 1.6, p * 0.75)

    # ground plane: soft gloss for a floor reflection + contact shadow
    bpy.ops.mesh.primitive_plane_add(size=size * 24,
                                     location=(center.x, center.y, mn.z))
    plane = bpy.context.active_object
    pmat = bpy.data.materials.new("floor")
    pmat.use_nodes = True
    pb = pmat.node_tree.nodes.get("Principled BSDF")
    set_input(pb, ["Base Color"], (0.20, 0.21, 0.23, 1.0))
    set_input(pb, ["Roughness"], 0.35)
    plane.data.materials.append(pmat)

    # camera: 3/4 view, slightly above, aimed at the object centre
    az, el, dist = math.radians(az_deg), math.radians(el_deg), size * 3.0
    cam_loc = center + Vector((math.sin(az) * math.cos(el),
                               -math.cos(az) * math.cos(el),
                               math.sin(el))) * dist
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 85
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = cam_loc
    aim(cam, center)
    bpy.context.scene.camera = cam


def configure_render(args):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True

    # Light-path depth. Cycles defaults (transmission 8, volume 0) are far too
    # shallow for a SOLID glass ball filled with water: a camera ray refracts
    # outer-wall -> inner-wall -> medium -> interior -> ... and any ray that
    # exhausts its transmission budget terminates BLACK. That dead-black core is
    # exactly what makes crystal read as murky plastic, so give it headroom
    # (max must stay above the transmission count).
    scene.cycles.transmission_bounces = args.transmission_bounces
    scene.cycles.max_bounces = max(args.max_bounces,
                                   args.transmission_bounces + 4)
    scene.cycles.volume_bounces = 2  # let a liquid medium scatter, not just clip
    scene.cycles.glossy_bounces = 8  # reflections-within-reflections on the dome

    # Caustics: the focused light a glass ball throws beneath it. Off by default
    # in Cycles (noisy), but on a static hero render the extra samples are worth
    # it — a strong "this is real glass" cue.
    scene.cycles.caustics_reflective = bool(args.caustics)
    scene.cycles.caustics_refractive = bool(args.caustics)
    if args.caustics:
        scene.cycles.blur_glossy = 0.5  # let bright caustic spikes through

    scene.render.resolution_x = args.res
    scene.render.resolution_y = args.res
    scene.render.film_transparent = False
    scene.view_settings.exposure = args.exposure
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = args.out

    # GPU (Metal on this Mac) if available, else CPU
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        scene.cycles.device = "GPU"
        print(f"[render_studio] Cycles device: GPU/METAL "
              f"({len(prefs.devices)} devices)")
    except Exception as e:  # noqa: BLE001
        scene.cycles.device = "CPU"
        print(f"[render_studio] Cycles device: CPU ({e})")
