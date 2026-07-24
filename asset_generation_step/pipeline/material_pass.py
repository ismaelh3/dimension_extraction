"""
Stage 3, M2 v3 — per-region PBR materials on the textured asset.

The texture bake paints everything with one material, so glass renders as
matte paint. This pass splits the mesh into two material regions and
re-exports:

    dome (glass) — low roughness, so viewers light it with sharp specular
                   highlights; optional alpha / transmission knobs below
    base (rock)  — high roughness matte

Region separation is geometric, no new photos needed: a snowglobe's radius-
vs-height profile pinches at the NECK between globe and base; every face
whose centroid sits above the pinch is glass. Override with NECK_Y=<0..1>
(fraction of object height) if the auto-detect picks the wrong minimum for
some other product shape.

Knobs:  GLASS_ROUGHNESS (default 0.08)   BASE_ROUGHNESS (default 0.9)
        GLASS_ALPHA (default 1.0 = opaque; <1 uses alphaMode BLEND — the
            painted interior fades with it, so keep it subtle, e.g. 0.9)
        GLASS_TRANSMISSION (default 0 = off; >0 adds the
            KHR_materials_transmission extension — real refraction in
            viewers that support it, but it makes the baked interior
            scene fade toward a tint; try 0.2-0.4)

Usage:  SUBJECT=snowglobe venv/bin/python asset_generation_step/pipeline/material_pass.py

Input:  work/<SUBJECT>_textured.glb   (from texture_hull.py)
Output: work/<SUBJECT>_final.glb      (+ _split_debug.png showing the split)
"""

import os
import sys

import numpy as np
import trimesh
from pygltflib import GLTF2

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # asset_generation_step/
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
sys.path.insert(0, os.path.join(BASE_DIR, 'pipeline'))
from preview_render import raster_preview  # noqa: E402
import glassiness  # noqa: E402

SUBJECT_ID = os.environ.get('SUBJECT', 'product_000')
IN_GLB     = os.path.join(BASE_DIR, 'work', f'{SUBJECT_ID}_textured.glb')
OUT_GLB    = os.path.join(BASE_DIR, 'work', f'{SUBJECT_ID}_final.glb')
DEBUG_PNG  = os.path.join(BASE_DIR, 'work', f'{SUBJECT_ID}_split_debug.png')
MASKS_ROOT = os.path.join(BASE_DIR, 'masks')

GLASS_ROUGHNESS    = float(os.environ.get('GLASS_ROUGHNESS', '0.08'))
BASE_ROUGHNESS     = float(os.environ.get('BASE_ROUGHNESS', '0.9'))
GLASS_ALPHA        = float(os.environ.get('GLASS_ALPHA', '1.0'))
GLASS_TRANSMISSION = float(os.environ.get('GLASS_TRANSMISSION', '0'))
NECK_Y             = os.environ.get('NECK_Y')            # manual neck override
GLASS_THRESHOLD    = float(os.environ.get('GLASS_THRESHOLD', '0.35'))  # face glassiness cutoff


def find_neck(mesh, bins=160):
    """Height (absolute y) of the pinch in the radius profile: the local
    minimum of per-height max radius, searched in the middle of the object
    so the tapering top/bottom caps can't win."""
    v = mesh.vertices
    y0, y1 = v[:, 1].min(), v[:, 1].max()
    cx, cz = mesh.bounds.mean(axis=0)[[0, 2]]
    r = np.hypot(v[:, 0] - cx, v[:, 2] - cz)
    idx = np.clip(((v[:, 1] - y0) / (y1 - y0) * (bins - 1)).astype(int),
                  0, bins - 1)
    prof = np.full(bins, np.nan)
    for b in range(bins):
        sel = idx == b
        if sel.any():
            prof[b] = r[sel].max()
    lo_b, hi_b = int(bins * 0.20), int(bins * 0.80)
    window = prof[lo_b:hi_b]
    b_min = lo_b + int(np.nanargmin(window))
    return y0 + (b_min + 0.5) / bins * (y1 - y0)


def submesh_with_uv(mesh, uv, face_mask, material):
    faces = mesh.faces[face_mask]
    used = np.unique(faces)
    remap = np.full(len(mesh.vertices), -1, np.int64)
    remap[used] = np.arange(len(used))
    return trimesh.Trimesh(
        vertices=mesh.vertices[used], faces=remap[faces], process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv[used], material=material))


def main():
    print("=" * 60)
    print("STAGE 3 — MATERIAL PASS (M2 v3)")
    print("=" * 60)
    if not os.path.exists(IN_GLB):
        print(f"[!] No textured asset at {IN_GLB} — run texture_hull.py first.")
        sys.exit(1)
    src_extras = GLTF2().load(IN_GLB).scenes[0].extras or {}
    mesh = trimesh.load(IN_GLB, force='mesh')
    uv = np.asarray(mesh.visual.uv)
    texture = mesh.visual.material.baseColorTexture
    print(f"Subject: {SUBJECT_ID}   {len(mesh.faces):,} faces\n")

    # WHICH faces are glass. Prefer IMAGE-DERIVED glassiness (depth see-through +
    # specular, projected onto the mesh) when glass maps exist — general for any
    # shape. Fall back to the geometric NECK pinch (a dome-on-pedestal
    # assumption) only when there are no glass maps (e.g. the snowglobe path or
    # NECK_Y is set explicitly).
    face_glass = None
    if NECK_Y is None:
        face_glass = glassiness.bake_face_glassiness(mesh, SUBJECT_ID, MASKS_ROOT)

    if face_glass is not None:
        glass_mask = face_glass > GLASS_THRESHOLD
        print(f"[*] glass from image glassiness (> {GLASS_THRESHOLD}): "
              f"{glass_mask.sum():,} glass / {(~glass_mask).sum():,} opaque faces")
    else:
        y0, y1 = mesh.bounds[:, 1]
        if NECK_Y is not None:
            neck = y0 + float(NECK_Y) * (y1 - y0)
            print(f"[*] Neck height overridden: {float(NECK_Y):.2f} of height")
        else:
            neck = find_neck(mesh)
            print(f"[*] No glass maps — neck fallback at "
                  f"{(neck - y0) / (y1 - y0):.2f} of height ({neck * 100:.2f} cm)")
        centroid_y = mesh.vertices[mesh.faces][:, :, 1].mean(axis=1)
        glass_mask = centroid_y > neck
        print(f"    glass: {glass_mask.sum():,} faces   "
              f"base: {(~glass_mask).sum():,} faces")

    glass_mat = trimesh.visual.material.PBRMaterial(
        name='glass_dome', baseColorTexture=texture,
        metallicFactor=0.0, roughnessFactor=GLASS_ROUGHNESS)
    if GLASS_ALPHA < 1.0:
        glass_mat.baseColorFactor = [1.0, 1.0, 1.0, GLASS_ALPHA]
        glass_mat.alphaMode = 'BLEND'
    base_mat = trimesh.visual.material.PBRMaterial(
        name='rock_base', baseColorTexture=texture,
        metallicFactor=0.0, roughnessFactor=BASE_ROUGHNESS)

    scene = trimesh.Scene()
    # keep 'glass' in the name so render_asset / assemble_container detect it;
    # skip a region if it's empty (a fully-glass or fully-opaque object).
    if glass_mask.any():
        scene.add_geometry(submesh_with_uv(mesh, uv, glass_mask, glass_mat),
                           geom_name='glass_dome')
    if (~glass_mask).any():
        scene.add_geometry(submesh_with_uv(mesh, uv, ~glass_mask, base_mat),
                           geom_name='rock_base')
    scene.export(OUT_GLB)

    g = GLTF2().load(OUT_GLB)
    if GLASS_TRANSMISSION > 0:
        for m in g.materials:
            if m.name and 'glass' in m.name:
                m.extensions = m.extensions or {}
                m.extensions['KHR_materials_transmission'] = {
                    'transmissionFactor': GLASS_TRANSMISSION}
        if 'KHR_materials_transmission' not in (g.extensionsUsed or []):
            g.extensionsUsed = (g.extensionsUsed or []) + \
                ['KHR_materials_transmission']
        print(f"    KHR_materials_transmission = {GLASS_TRANSMISSION:g}")
    meta = {
        'method': 'image_glassiness' if face_glass is not None else 'neck',
        'glass_roughness': GLASS_ROUGHNESS, 'base_roughness': BASE_ROUGHNESS,
        'glass_alpha': GLASS_ALPHA, 'glass_transmission': GLASS_TRANSMISSION,
        'glass_faces': int(glass_mask.sum()),
    }
    if face_glass is not None:
        meta['glass_threshold'] = GLASS_THRESHOLD
    else:
        meta['neck_fraction_of_height'] = round((neck - y0) / (y1 - y0), 4)
    src_extras['material_pass'] = meta
    g.scenes[g.scene or 0].extras = src_extras
    g.save(OUT_GLB)

    # debug render: region split tinted (blue = glass, warm = opaque) so a bad
    # split is obvious at a glance. Colour vertices by glass-face membership.
    dbg = np.zeros((len(mesh.vertices), 4), np.uint8)
    dbg[:] = [210, 150, 90, 255]
    vglass = np.zeros(len(mesh.vertices), bool)
    if glass_mask.any():
        vglass[mesh.faces[glass_mask].reshape(-1)] = True
    dbg[vglass] = [110, 160, 235, 255]
    raster_preview(mesh, vertex_colors=dbg).save(DEBUG_PNG)

    size_mb = os.path.getsize(OUT_GLB) / 1024 / 1024
    print(f"\n[*] Wrote {OUT_GLB}  ({size_mb:.1f} MB)")
    print(f"    split debug: {DEBUG_PNG}")
    print("\nDone. Inspect at https://gltf-viewer.donmccurdy.com — the")
    print("specular/transparency look depends on the viewer's lighting.")


if __name__ == '__main__':
    main()
