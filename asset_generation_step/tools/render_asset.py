"""
Stage 3 — honest ray-traced product render (Blender + Cycles), object-agnostic.

preview_render.py is a lambert/opaque software rasterizer: it CANNOT show
transparency, reflections, or Fresnel, so a glass product renders there as matte
plastic. This renders the same asset in a real ray tracer, so glass looks like
glass. It is the GENERAL driver: the reusable engine is render_studio.py, and
object-specific scene dressing lives in scenes/<name>.py.

Runs INSIDE Blender's bundled Python — drive it through the launcher
(tools/render_preview.py / `make render-asset`), which finds the Blender binary:

    blender --background --python render_asset.py -- --in a.glb --out a.png [--scene <name>]

Behaviour:
  * every object whose name contains --glass-substr (default "glass") gets a
    fresh transmissive Principled BSDF (see render_studio.make_glass); others
    keep their imported textured material.
  * with --scene <name>, scenes/<name>.build() constructs an object-specific
    scene (e.g. snowglobe: sphere-fit dome + water + interior + snow). With
    --scene none (default), the general path materials the glass region in place
    and optionally seats --interior inside it. Works for ANY subject.

General knobs:  --in --out --samples --res --roughness --ior --thickness
    --smooth-iters --exposure --tint R,G,B --glass-substr --dispersion
    --transmission-bounces --max-bounces --caustics --interior --interior-fill
Scene-specific knobs (consumed only by the named preset): --penguins --water
    --snow --dome-shape
"""

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_studio as studio  # noqa: E402


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    # general
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="out", required=True)
    p.add_argument("--scene", default="none",
                   help="scene preset in tools/scenes/ (e.g. 'snowglobe'); "
                        "'none' = general glass-in-place render")
    p.add_argument("--samples", type=int, default=160)
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--roughness", type=float, default=0.02)
    p.add_argument("--ior", type=float, default=1.5)
    p.add_argument("--thickness", type=float, default=0.003,
                   help="glass wall thickness in metres; 0 = solid block")
    p.add_argument("--smooth-iters", type=int, default=30,
                   help="Laplacian smoothing passes on a bumpy carved glass "
                        "surface (general path)")
    p.add_argument("--interior", default="",
                   help="optional .glb seated inside the glass region")
    p.add_argument("--interior-fill", type=float, default=0.62,
                   help="interior height as a fraction of the shell diameter")
    p.add_argument("--exposure", type=float, default=-1.1)
    p.add_argument("--tint", default="1.0,1.0,1.0")
    p.add_argument("--glass-substr", default="glass")
    p.add_argument("--dispersion", type=float, default=0.04,
                   help="Principled dispersion on the glass (crystal edge tell)")
    p.add_argument("--transmission-bounces", type=int, default=24,
                   help="Cycles transmission bounces; the default 8 terminates "
                        "deep rays BLACK (murky-plastic look)")
    p.add_argument("--max-bounces", type=int, default=32)
    p.add_argument("--caustics", type=int, default=1,
                   help="1 = reflective+refractive caustics; 0 = off")
    # scene-preset passthrough (ignored by the general path)
    p.add_argument("--dome-shape", choices=["sphere", "hull"], default="sphere")
    p.add_argument("--penguins", type=int, default=2)
    p.add_argument("--water", type=float, default=1.0)
    p.add_argument("--snow", type=int, default=320)
    return p.parse_args(argv)


def load_scene(name):
    """Import scenes/<name>.py, or return None for 'none'/unknown."""
    if not name or name == "none":
        return None
    try:
        return importlib.import_module(f"scenes.{name}")
    except ImportError as e:
        print(f"[render_asset] scene '{name}' not found ({e}) — general path")
        return None


def main():
    args = parse_args()
    args.tint = tuple(float(x) for x in args.tint.split(","))

    studio.clean_scene()
    meshes = studio.import_asset(args.inp)
    if not meshes:
        print("[render_asset] no meshes imported — aborting")
        sys.exit(1)

    glass = [o for o in meshes if args.glass_substr.lower() in o.name.lower()]
    other = [o for o in meshes if o not in glass]
    print(f"[render_asset] {len(meshes)} meshes  "
          f"glass={[o.name for o in glass]}  base={[o.name for o in other]}")
    if not glass:
        print("[render_asset] WARNING: no object matched "
              f"'{args.glass_substr}' — nothing will be transparent")

    scene = load_scene(args.scene)
    if scene is not None:
        meshes = scene.build(studio, glass, other, args)
    else:
        # general path: material the glass region in place, seat interior if any
        for o in glass:
            studio.make_glass(o, args.tint, args.roughness, args.ior,
                              args.thickness, args.smooth_iters,
                              dispersion=args.dispersion)
        if args.interior and glass:
            c, r = studio.fit_sphere(glass)
            meshes = other + glass + studio.add_interior(
                args.interior, c, r, args.interior_fill)

    studio.build_studio(meshes)
    studio.configure_render(args)
    print(f"[render_asset] rendering {args.samples} spp @ {args.res}px "
          f"(scene={args.scene}) ...")
    import bpy
    bpy.ops.render.render(write_still=True)
    print(f"[render_asset] wrote {args.out}")


if __name__ == "__main__":
    main()
