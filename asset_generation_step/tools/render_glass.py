"""
Backward-compat shim: render_glass.py is now the `snowglobe` preset of the
general render_asset.py. Kept so existing `make render-glass` calls and any
direct `blender --python render_glass.py -- ...` invocations keep working with
the snow/water/penguin dressing.

The engine moved to render_studio.py; the snowglobe scene to scenes/snowglobe.py;
the general driver is render_asset.py. New work should call render_asset.py with
`--scene snowglobe` (or another preset / `--scene none`).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# default the scene to snowglobe unless the caller already chose one
if "--scene" not in sys.argv:
    sys.argv += ["--scene", "snowglobe"]

import render_asset  # noqa: E402

if __name__ == "__main__":
    render_asset.main()
