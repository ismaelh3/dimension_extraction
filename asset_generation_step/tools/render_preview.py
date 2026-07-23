"""
Launcher for the Blender/Cycles product render (tools/render_asset.py).

render_asset.py runs inside Blender's own Python, and Blender is not on PATH
here (it's the .app). This venv-side wrapper finds the Blender binary and shells
out headlessly, so renders run like any other step:

    SUBJECT=widget make render-asset
    SUBJECT=snowglobe SCENE=snowglobe make render-asset
    SUBJECT=widget venv/bin/python asset_generation_step/tools/render_preview.py

Resolution order for the binary:  $BLENDER env  >  `blender` on PATH  >
the standard macOS app location.

Input : work/<SUBJECT>_final.glb  (falls back to _textured.glb, then _assembled.glb)
Output: work/<SUBJECT>_render.png  (override with OUTPUT=/path.png)

Env knobs:
  SCENE        scene preset (e.g. 'snowglobe'); default 'none' = general render
  INTERIOR     .glb to seat inside the glass region (else auto: <SUBJECT>_interior_textured
               / _interior_hull if present); 'none' to force-disable
  general      SAMPLES RES ROUGHNESS IOR EXPOSURE TINT THICKNESS SMOOTH_ITERS
               GLASS_SUBSTR DISPERSION TRANSMISSION_BOUNCES MAX_BOUNCES CAUSTICS
               INTERIOR_FILL
  snowglobe    DOME_SHAPE PENGUINS WATER SNOW  (consumed only by --scene snowglobe)
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(HERE)                       # asset_generation_step/
RENDER_SCRIPT = os.path.join(HERE, "render_asset.py")

MAC_APP = "/Applications/Blender.app/Contents/MacOS/Blender"


def find_blender():
    cand = os.environ.get("BLENDER") or shutil.which("blender") or MAC_APP
    if not os.path.exists(cand):
        sys.exit(f"[render_preview] Blender not found (tried {cand}). "
                 f"Set BLENDER=/path/to/blender.")
    return cand


def resolve_input(work, subject):
    for suffix in ("_final.glb", "_textured.glb", "_assembled.glb"):
        p = os.path.join(work, f"{subject}{suffix}")
        if os.path.exists(p):
            if suffix != "_final.glb":
                print(f"[render_preview] no _final.glb; using "
                      f"{os.path.basename(p)}")
            return p
    sys.exit(f"[render_preview] no asset for '{subject}' in {work} "
             f"(need _final / _textured / _assembled .glb)")


def resolve_interior(work, subject, scene):
    """Which interior .glb to seat, or '' for none. Explicit INTERIOR wins.
    For the snowglobe preset, the interior is only used when PENGUINS=0 (else
    procedural penguins fill it); every other scene auto-attaches if present."""
    interior = os.environ.get("INTERIOR")
    if interior == "none":
        return ""
    if scene == "snowglobe" and os.environ.get("PENGUINS") != "0" \
            and interior is None:
        return ""  # procedural penguins are the default snowglobe interior
    if interior:
        return interior
    for suffix in ("_interior_textured.glb", "_interior_hull.glb"):
        cand = os.path.join(work, f"{subject}{suffix}")
        if os.path.exists(cand):
            return cand
    return ""


def main():
    subject = os.environ.get("SUBJECT", "product_000")
    scene = os.environ.get("SCENE", "none")
    work = os.path.join(BASE_DIR, "work")
    inp = resolve_input(work, subject)
    out = os.environ.get("OUTPUT") or os.path.join(work, f"{subject}_render.png")

    blender = find_blender()
    cmd = [blender, "--background", "--factory-startup",
           "--python", RENDER_SCRIPT, "--",
           "--in", inp, "--out", out, "--scene", scene]

    interior = resolve_interior(work, subject, scene)
    if interior:
        cmd += ["--interior", interior]
        print(f"[render_preview] interior: {interior}")

    passthrough = {
        "SAMPLES": "--samples", "RES": "--res", "ROUGHNESS": "--roughness",
        "IOR": "--ior", "EXPOSURE": "--exposure", "TINT": "--tint",
        "THICKNESS": "--thickness", "SMOOTH_ITERS": "--smooth-iters",
        "GLASS_SUBSTR": "--glass-substr", "DISPERSION": "--dispersion",
        "TRANSMISSION_BOUNCES": "--transmission-bounces",
        "MAX_BOUNCES": "--max-bounces", "CAUSTICS": "--caustics",
        "INTERIOR_FILL": "--interior-fill",
        # snowglobe-preset knobs (ignored by the general path)
        "DOME_SHAPE": "--dome-shape", "PENGUINS": "--penguins",
        "WATER": "--water", "SNOW": "--snow",
    }
    for env, flag in passthrough.items():
        if os.environ.get(env):
            cmd += [flag, os.environ[env]]

    print(f"[render_preview] {blender}\n[render_preview] in : {inp}\n"
          f"[render_preview] out: {out}   scene={scene}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"[render_preview] Blender exited {r.returncode}")
    print(f"[render_preview] done -> {out}")


if __name__ == "__main__":
    main()
