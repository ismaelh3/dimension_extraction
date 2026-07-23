"""
Vendor + patch TripoSR for local single-image->3D reconstruction.

TripoSR (github.com/VAST-AI-Research/TripoSR) is the generative fallback for the
one thing extraction can't recover: the interior of a sealed refractive object
(seen only through the glass, so shape-from-silhouette merges it to a blob). It
is cloned into a gitignored third_party/ and patched to run on this machine's
stack (transformers 5.x, no torchmcubes/moderngl, MPS). All patches are
idempotent — re-running is safe and re-applies nothing already present.

Usage:  venv/bin/python asset_generation_step/tools/setup_triposr.py
        (generate_interior.py calls ensure() automatically)

Returns / prints the TripoSR checkout path.
"""

import os
import subprocess
import sys

REPO = "https://github.com/VAST-AI-Research/TripoSR.git"
# repo root = three levels up from this file (asset_generation_step/tools/..)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRIPOSR_DIR = os.environ.get("TRIPOSR_DIR",
                             os.path.join(ROOT, "third_party", "TripoSR"))


def _patch(path, old, new, marker, label):
    """Idempotent single-shot patch. Skips if `marker` (a substring unique to
    the patched form) is already present; else replaces `old`->`new`. Prints a
    clear warning if `old` isn't found (upstream drifted)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if marker in text:
        print(f"  · {label}: already applied")
        return
    if old not in text:
        print(f"  ! {label}: anchor not found — TripoSR may have changed; "
              f"inspect {os.path.relpath(path, TRIPOSR_DIR)}")
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new, 1))
    print(f"  ✓ {label}")


def apply_patches():
    d = TRIPOSR_DIR

    # (1) isosurface: skimage marching_cubes instead of the torchmcubes C ext,
    #     and DROP the torchmcubes [2,1,0] axis swap (skimage returns x,y,z).
    _patch(
        os.path.join(d, "tsr/models/isosurface.py"),
        "from torchmcubes import marching_cubes",
        "from skimage.measure import marching_cubes as _skimage_marching_cubes\n"
        "import numpy as _np\n"
        "import torch as _torch\n\n\n"
        "def marching_cubes(volume, level):\n"
        "    \"\"\"torchmcubes-compatible wrapper over skimage (no CUDA ext).\"\"\"\n"
        "    vol = volume.detach().cpu().numpy()\n"
        "    verts, faces, _n, _v = _skimage_marching_cubes(vol, level)\n"
        "    v = _torch.from_numpy(_np.ascontiguousarray(verts)).float()\n"
        "    fcs = _torch.from_numpy(_np.ascontiguousarray(faces)).long()\n"
        "    return v, fcs",
        marker="_skimage_marching_cubes",
        label="isosurface: skimage marching_cubes")
    _patch(
        os.path.join(d, "tsr/models/isosurface.py"),
        "        v_pos = v_pos[..., [2, 1, 0]]\n        v_pos = v_pos / (self.resolution - 1.0)",
        "        # skimage already returns x,y,z order — no torchmcubes swap\n"
        "        v_pos = v_pos / (self.resolution - 1.0)",
        marker="skimage already returns",
        label="isosurface: drop [2,1,0] swap")

    # (2)+(3) lazy rembg / bake_texture in run.py (onnxruntime + moderngl absent)
    _patch(
        os.path.join(d, "run.py"),
        "import numpy as np\nimport rembg\nimport torch\nimport xatlas\n"
        "from PIL import Image\n\nfrom tsr.system import TSR\n"
        "from tsr.utils import remove_background, resize_foreground, save_video\n"
        "from tsr.bake_texture import bake_texture",
        "import numpy as np\nimport torch\nimport xatlas\n"
        "from PIL import Image\n\nfrom tsr.system import TSR\n"
        "from tsr.utils import remove_background, resize_foreground, save_video\n"
        "# rembg (onnxruntime) and bake_texture (moderngl GL) are lazy-imported",
        marker="lazy-imported",
        label="run.py: drop eager rembg/bake_texture imports")
    _patch(
        os.path.join(d, "run.py"),
        "else:\n    rembg_session = rembg.new_session()",
        "else:\n    import rembg  # lazy\n    rembg_session = rembg.new_session()",
        marker="import rembg  # lazy",
        label="run.py: lazy rembg session")
    _patch(
        os.path.join(d, "run.py"),
        '        timer.start("Baking texture")\n'
        "        bake_output = bake_texture(",
        '        timer.start("Baking texture")\n'
        "        from tsr.bake_texture import bake_texture  # lazy: moderngl GL\n"
        "        bake_output = bake_texture(",
        marker="from tsr.bake_texture import bake_texture  # lazy",
        label="run.py: lazy bake_texture")
    # (5) pre-create the per-image output dir (skipped in the --no-remove-bg path)
    _patch(
        os.path.join(d, "run.py"),
        '    out_mesh_path = os.path.join(output_dir, str(i), '
        'f"mesh.{args.model_save_format}")\n    if args.bake_texture:',
        "    os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)\n"
        '    out_mesh_path = os.path.join(output_dir, str(i), '
        'f"mesh.{args.model_save_format}")\n    if args.bake_texture:',
        marker="os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)",
        label="run.py: ensure output dir")

    # (3b) lazy rembg in utils.py
    _patch(
        os.path.join(d, "tsr/utils.py"),
        "import PIL.Image\nimport rembg\nimport torch",
        "import PIL.Image\nimport torch",
        marker="import PIL.Image\nimport torch",
        label="utils.py: drop eager rembg")
    _patch(
        os.path.join(d, "tsr/utils.py"),
        "    if do_remove:\n        image = rembg.remove(",
        "    if do_remove:\n        import rembg  # lazy\n        image = rembg.remove(",
        marker="import rembg  # lazy",
        label="utils.py: lazy rembg.remove")

    # (4) ViT checkpoint key remap (transformers 4.35 -> 5.x) + strict=False
    _patch(
        os.path.join(d, "tsr/system.py"),
        "        ckpt = torch.load(weight_path, map_location=\"cpu\")\n"
        "        model.load_state_dict(ckpt)\n"
        "        return model",
        "        ckpt = torch.load(weight_path, map_location=\"cpu\")\n"
        "        ckpt = cls._remap_vit_keys(ckpt)\n"
        "        model.load_state_dict(ckpt, strict=False)\n"
        "        return model\n\n"
        "    @staticmethod\n"
        "    def _remap_vit_keys(ckpt):\n"
        "        \"\"\"DINO ViT tokenizer ckpt was saved with transformers 4.35;\n"
        "        transformers 5.x renamed the ViT encoder submodules. Verified\n"
        "        1:1 (192<->192). Order matters: rewrite the specific\n"
        "        attention.output.dense before the generic output.dense.\"\"\"\n"
        "        out = {}\n"
        "        for k, v in ckpt.items():\n"
        "            nk = k\n"
        "            if \".encoder.layer.\" in nk:\n"
        "                nk = nk.replace(\".encoder.layer.\", \".layers.\")\n"
        "                nk = nk.replace(\".attention.attention.query\", \".attention.q_proj\")\n"
        "                nk = nk.replace(\".attention.attention.key\", \".attention.k_proj\")\n"
        "                nk = nk.replace(\".attention.attention.value\", \".attention.v_proj\")\n"
        "                nk = nk.replace(\".attention.output.dense\", \".attention.o_proj\")\n"
        "                nk = nk.replace(\".intermediate.dense\", \".mlp.fc1\")\n"
        "                nk = nk.replace(\".output.dense\", \".mlp.fc2\")\n"
        "            out[nk] = v\n"
        "        return out",
        marker="_remap_vit_keys",
        label="system.py: ViT key remap + strict=False")


def ensure():
    """Clone TripoSR if missing, apply patches, return the checkout dir."""
    if not os.path.isdir(os.path.join(TRIPOSR_DIR, "tsr")):
        os.makedirs(os.path.dirname(TRIPOSR_DIR), exist_ok=True)
        print(f"[setup_triposr] cloning TripoSR -> {TRIPOSR_DIR}")
        r = subprocess.run(["git", "clone", "--depth", "1", REPO, TRIPOSR_DIR])
        if r.returncode != 0:
            sys.exit("[setup_triposr] git clone failed")
    else:
        print(f"[setup_triposr] TripoSR present at {TRIPOSR_DIR}")
    print("[setup_triposr] applying patches (idempotent):")
    apply_patches()
    return TRIPOSR_DIR


if __name__ == "__main__":
    print(ensure())
