"""
Stage 3 — assemble a transparent-container product into ONE deliverable GLB.

General form of the old assemble_snowglobe.py: any subject whose extracted asset
is a transparent shell + opaque body with a separately-produced interior (carved
or generated). Combines them into one <SUBJECT>_assembled.glb:
  * base part(s) — every non-glass region of work/<SUBJECT>_final.glb, kept with
                   their baked textures.
  * glass shell  — the region(s) whose geometry/material name contains
                   GLASS_SUBSTR ('glass'), re-materialled to CLEAR glass
                   (KHR_materials_transmission + KHR_materials_ior, texture
                   stripped so it is actually see-through).
  * interior     — INTERIOR_GLB (default work/<SUBJECT>_interior_textured.glb),
                   scaled to a fraction of the shell and seated inside so it
                   reads through the glass.

A solid transmissive shell with the interior embedded is faithful to a
paperweight-style object (the interior refracts through it).

Usage:  SUBJECT=snowglobe venv/bin/python asset_generation_step/pipeline/assemble_container.py
Knobs:  INTERIOR_FILL (0.60 of shell diameter)  SEAT (0.42 below centre, ×radius)
        SHELL_SHAPE (sphere|bbox)  GLASS_SUBSTR (glass)  GLASS_ROUGHNESS (0.05)
        GLASS_IOR (1.5)  INTERIOR_GLB (…/<SUBJECT>_interior_textured.glb)
Output: work/<SUBJECT>_assembled.glb
"""

import math
import os
import sys

import numpy as np
import trimesh
from pygltflib import GLTF2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBJECT  = os.environ.get('SUBJECT', 'product_000')
DOME_SRC = os.path.join(BASE_DIR, 'work', f'{SUBJECT}_final.glb')
INTERIOR = os.environ.get(
    'INTERIOR_GLB',
    os.path.join(BASE_DIR, 'work', f'{SUBJECT}_interior_textured.glb'))
OUT_GLB  = os.path.join(BASE_DIR, 'work', f'{SUBJECT}_assembled.glb')

INTERIOR_FILL   = float(os.environ.get('INTERIOR_FILL', '0.60'))
GLASS_ROUGHNESS = float(os.environ.get('GLASS_ROUGHNESS', '0.05'))
GLASS_IOR       = float(os.environ.get('GLASS_IOR', '1.5'))
SEAT            = float(os.environ.get('SEAT', '0.42'))   # floor below centre, ×radius
SHELL_SHAPE     = os.environ.get('SHELL_SHAPE', 'sphere')  # sphere|bbox
GLASS_SUBSTR    = os.environ.get('GLASS_SUBSTR', 'glass').lower()


def world_mesh(scene, name):
    """Return geometry `name` from a scene, baked into world space."""
    g = scene.geometry[name].copy()
    T = scene.graph.get(name)[0]
    g.apply_transform(T)
    return g


def fit_sphere(points):
    """Least-squares sphere through vertices: x^2+y^2+z^2 = 2c.p + d."""
    p = np.asarray(points)
    A = np.hstack([2 * p, np.ones((len(p), 1))])
    b = (p ** 2).sum(axis=1)
    cx, cy, cz, d = np.linalg.lstsq(A, b, rcond=None)[0]
    r = math.sqrt(max(d + cx*cx + cy*cy + cz*cz, 0.0))
    return np.array([cx, cy, cz]), r


def is_glass(scene, name):
    """A region is the glass shell if its geometry name or material name
    contains GLASS_SUBSTR (material_pass.py names it 'glass_dome')."""
    if GLASS_SUBSTR in name.lower():
        return True
    mat = getattr(getattr(scene.geometry[name], 'visual', None), 'material', None)
    return bool(mat and GLASS_SUBSTR in (getattr(mat, 'name', '') or '').lower())


def load_interior():
    peng = trimesh.load(INTERIOR)
    if hasattr(peng, 'to_geometry'):
        return peng.to_geometry()
    return peng if isinstance(peng, trimesh.Trimesh) else \
        list(peng.geometry.values())[0]


def main():
    if not os.path.exists(DOME_SRC):
        sys.exit(f"[assemble] need {DOME_SRC} (run material_pass first)")
    if not os.path.exists(INTERIOR):
        sys.exit(f"[assemble] need {INTERIOR} (texture the interior first, or "
                 f"set INTERIOR_GLB)")

    scene_src = trimesh.load(DOME_SRC)
    names = list(scene_src.geometry.keys())
    glass_names = [n for n in names if is_glass(scene_src, n)]
    base_names = [n for n in names if n not in glass_names]
    if not glass_names:
        sys.exit(f"[assemble] no glass region (name/material containing "
                 f"'{GLASS_SUBSTR}') in {os.path.basename(DOME_SRC)} — did "
                 f"material_pass.py run? regions: {names}")
    print(f"[assemble] glass={glass_names}  base={base_names}")

    dome = trimesh.util.concatenate([world_mesh(scene_src, n)
                                     for n in glass_names])
    bases = [(n, world_mesh(scene_src, n)) for n in base_names]

    if SHELL_SHAPE == 'bbox':
        mn, mx = dome.bounds
        center = (mn + mx) / 2
        radius = float((mx - mn).max() / 2)
    else:
        center, radius = fit_sphere(dome.vertices)
    print(f"[assemble] shell ({SHELL_SHAPE}): r={radius*100:.2f} cm  "
          f"centre=({center[0]*100:.1f},{center[1]*100:.1f},{center[2]*100:.1f}) cm")

    # interior — scale to a fraction of the shell, seat in the lower hemisphere.
    # glTF is Y-up, so Y is the vertical (height) axis.
    peng = load_interior()
    mn, mx = peng.bounds
    p_h = mx[1] - mn[1]
    s = INTERIOR_FILL * 2 * radius / p_h
    pivot = np.array([(mn[0]+mx[0])/2, mn[1], (mn[2]+mx[2])/2])  # bottom-centre
    seat = np.array([center[0], center[1] - SEAT * radius, center[2]])
    peng.apply_translation(-pivot)
    peng.apply_scale(s)
    peng.apply_translation(seat)
    print(f"[assemble] interior scaled x{s:.2f} -> height "
          f"{INTERIOR_FILL*2*radius*100:.1f} cm, seated inside")

    # clear-glass material on the shell (texture stripped)
    dome.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name='glass_shell',
            baseColorFactor=[255, 255, 255, 255],
            metallicFactor=0.0,
            roughnessFactor=GLASS_ROUGHNESS,
        ))

    scene = trimesh.Scene()
    for n, g in bases:
        scene.add_geometry(g, geom_name=n)
    scene.add_geometry(peng, geom_name='interior')
    scene.add_geometry(dome, geom_name='glass_shell')
    scene.export(OUT_GLB)

    # post-edit: real glass = transmission + ior on the shell material
    g = GLTF2().load(OUT_GLB)
    for m in g.materials:
        if GLASS_SUBSTR in (m.name or '').lower():
            m.extensions = m.extensions or {}
            m.extensions['KHR_materials_transmission'] = {'transmissionFactor': 1.0}
            m.extensions['KHR_materials_ior'] = {'ior': GLASS_IOR}
    for ext in ('KHR_materials_transmission', 'KHR_materials_ior'):
        if ext not in (g.extensionsUsed or []):
            g.extensionsUsed = (g.extensionsUsed or []) + [ext]
    g.save(OUT_GLB)

    size_mb = os.path.getsize(OUT_GLB) / 1e6
    print(f"[assemble] wrote {OUT_GLB}  ({size_mb:.1f} MB)")
    print(f"[assemble] shell=clear glass (transmission 1.0, IOR {GLASS_IOR}), "
          f"base+interior textured. Inspect at gltf-viewer.donmccurdy.com")


if __name__ == '__main__':
    main()
