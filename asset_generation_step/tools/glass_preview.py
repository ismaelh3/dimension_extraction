"""
Backward-compat shim: glass_preview.py is the snowglobe preset of the general
render_preview.py launcher. Kept so `SUBJECT=snowglobe make render-glass` and
existing invocations keep working, writing to work/<SUBJECT>_glass.png.

New work should call render_preview.py with SCENE=<preset> (or SCENE=none).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("SCENE", "snowglobe")
os.environ.setdefault(
    "OUTPUT",
    os.path.join(os.path.dirname(HERE), "work",
                 f"{os.environ.get('SUBJECT', 'product_000')}_glass.png"))

import render_preview  # noqa: E402

if __name__ == "__main__":
    render_preview.main()
