"""
Snowglobe render preset — the object-specific scene dressing that used to live
inline in render_glass.py. Fits the carved dome to an ideal glass ball, fills it
with water, seats the interior (an extracted/generated family, or procedural
penguins if penguin.py is present), and scatters drifting snow.

This is the ONLY snowglobe-aware render code. Everything it calls for glass,
lighting, camera, and interior seating comes from render_studio.py, so nothing
here leaks into the general path.

Entry point: build(studio, glass, other, args) -> list[bpy object] to frame.
"""

import math
import os
import sys

from mathutils import Vector

# penguin.py (procedural figurines) is optional and snowglobe-specific; the
# deliverable path uses the EXTRACTED interior (--interior) instead.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import penguin  # noqa: E402
except ImportError:
    penguin = None


def add_snow_floor(studio, center, radius, floor_z):
    """A shallow, softly-drifted white mound the interior stands on. A squashed
    icosphere (lower half hidden in the base) with noise displacement for
    drifts; matte snow with a little subsurface softness."""
    import bpy
    # 0.82*r keeps the mound clear of the glass at the rim (the globe's radius at
    # floor height is ~0.91*r); a thin lens so it reads as a shallow snow bed.
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5, radius=0.82 * radius,
                                           location=(center.x, center.y, floor_z))
    mound = bpy.context.active_object
    mound.name = "snow_floor"
    mound.scale = (1.0, 1.0, 0.11)
    for p in mound.data.polygons:
        p.use_smooth = True

    tex = bpy.data.textures.new("snow_drift", "CLOUDS")
    tex.noise_scale = 0.35
    disp = mound.modifiers.new("drift", "DISPLACE")
    disp.texture = tex
    disp.strength = 0.04 * radius
    disp.mid_level = 0.4

    mat = bpy.data.materials.new("snow")
    mat.use_nodes = True
    b = mat.node_tree.nodes.get("Principled BSDF")
    studio.set_input(b, ["Base Color"], (0.96, 0.97, 1.0, 1.0))
    studio.set_input(b, ["Roughness"], 0.85)
    studio.set_input(b, ["Subsurface Weight", "Subsurface"], 0.15)
    studio.set_input(b, ["Subsurface Radius"], (0.01, 0.01, 0.015))
    mound.data.materials.append(mat)
    return [mound]


def add_water(studio, center, radius, strength):
    """Fill the globe with a water medium (IOR 1.33) + faint cool volume
    absorption. Sits just inside the inner glass wall (small air gap avoids
    coincident faces / nested-dielectric artifacts)."""
    import bpy
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5, radius=radius,
                                           location=center)
    w = bpy.context.active_object
    w.name = "water"
    for p in w.data.polygons:
        p.use_smooth = True
    mat = bpy.data.materials.new("water")
    mat.use_nodes = True
    nt = mat.node_tree
    b = nt.nodes.get("Principled BSDF")
    studio.set_input(b, ["Base Color"], (1, 1, 1, 1))
    studio.set_input(b, ["Roughness"], 0.0)
    studio.set_input(b, ["IOR"], 1.33)
    studio.set_input(b, ["Transmission Weight", "Transmission"], 1.0)
    out = nt.nodes.get("Material Output")
    absorb = nt.nodes.new("ShaderNodeVolumeAbsorption")
    absorb.inputs["Color"].default_value = (0.80, 0.92, 0.96, 1.0)
    absorb.inputs["Density"].default_value = 1.6 * strength
    nt.links.new(absorb.outputs["Volume"], out.inputs["Volume"])
    w.data.materials.append(mat)
    print(f"[snowglobe] water fill r={radius*100:.1f} cm  IOR 1.33")
    return w


def add_snow_particles(studio, center, radius, n):
    """Scatter `n` tiny white flakes inside the globe, sharing one mesh
    datablock so it stays cheap. Bright matte white so they read through the
    glass + water; biased a touch downward like settling snow."""
    import bpy
    import random as _r
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1,
                                           radius=0.011 * radius, location=center)
    proto = bpy.context.active_object
    proto.name = "snow_flake"
    for p in proto.data.polygons:
        p.use_smooth = True
    mat = bpy.data.materials.new("snowflake")
    mat.use_nodes = True
    b = mat.node_tree.nodes.get("Principled BSDF")
    studio.set_input(b, ["Base Color"], (1.0, 1.0, 1.0, 1.0))
    studio.set_input(b, ["Roughness"], 0.6)
    studio.set_input(b, ["Emission Color"], (1.0, 1.0, 1.0, 1.0))
    studio.set_input(b, ["Emission Strength"], 0.25)
    proto.data.materials.append(mat)

    mesh = proto.data
    rng = _r.Random(7)
    made = [proto]
    placed = 0
    while placed < n:
        x, y, z = (rng.random() * 2 - 1 for _ in range(3))
        if x * x + y * y + z * z > 0.80 ** 2:
            continue
        loc = Vector((center.x + x * radius,
                      center.y + y * radius,
                      center.z + z * radius * 0.9 - 0.05 * radius))
        if placed == 0:
            proto.location = loc
        else:
            o = bpy.data.objects.new(f"snow_{placed}", mesh)
            bpy.context.collection.objects.link(o)
            o.location = loc
            made.append(o)
        placed += 1
    print(f"[snowglobe] {placed} snow particles")
    return made


def place_penguins(center, radius, floor_z, count, az_deg=38.0):
    """Seat procedural penguins on the snow mound, turned to face the camera
    direction. Requires the optional penguin.py."""
    if penguin is None:
        print("[snowglobe] penguin.py not present — skipping procedural penguins "
              "(use --penguins 0 with --interior for the extracted family)")
        return []
    cam_dir = (math.sin(math.radians(az_deg)), -math.cos(math.radians(az_deg)))
    seat_z = floor_z + 0.10 * radius          # stand on the snow bed
    # (dx, dy as fractions of radius, height fraction, yaw jitter deg, seed)
    layout = [
        (0.17, -0.05, 0.58, -14, 1),
        (-0.20, 0.14, 0.46, 22, 2),
        (0.02, 0.26, 0.38, 4, 3),
    ]
    objs = []
    for i in range(min(count, len(layout))):
        dx, dy, hf, jit, seed = layout[i]
        loc = Vector((center.x + dx * radius, center.y + dy * radius, seat_z))
        a = math.radians(jit)
        fx = cam_dir[0] * math.cos(a) - cam_dir[1] * math.sin(a)
        fy = cam_dir[0] * math.sin(a) + cam_dir[1] * math.cos(a)
        objs += penguin.build_penguin(loc, hf * radius, (fx, fy), seed)
    if count > len(layout):
        print(f"[snowglobe] penguin layout capped at {len(layout)}")
    print(f"[snowglobe] placed {min(count, len(layout))} penguins")
    return objs


def build(studio, glass, other, args):
    """Construct the snowglobe scene. Fits the glass region to an ideal sphere,
    materials it as clear crystal, then adds snow floor + water + interior +
    drifting snow. Returns the full mesh list for build_studio()."""
    if not glass:
        print("[snowglobe] no glass region matched — rendering base only")
        return other
    sphere, c, r = studio.replace_with_sphere(glass)
    studio.make_glass(sphere, args.tint, args.roughness, args.ior,
                      args.thickness, smooth_iters=0,  # an icosphere is smooth
                      dispersion=args.dispersion)

    extras = []
    inner_r = r - max(args.thickness, 0.0)
    floor_z = c.z - 0.42 * r
    extras += add_snow_floor(studio, c, r, floor_z)
    if args.water > 0:
        extras.append(add_water(studio, c, inner_r * 0.99, args.water))
    if args.penguins > 0:
        extras += place_penguins(c, r, floor_z, args.penguins)
    elif args.interior:
        extras += studio.add_interior(args.interior, c, r, args.interior_fill)
    if args.snow > 0:
        extras += add_snow_particles(studio, c, inner_r, args.snow)
    print(f"[snowglobe] glass wall: "
          f"{'solid block' if args.thickness <= 0 else f'{args.thickness*1000:.1f} mm shell'}")
    return other + [sphere] + extras
