"""
Stage 3 — turntable visual hull (many-view shape-from-silhouette).

build_silhouette_mesh.py carves from a few NAMED orthographic views
(front/side/back/top). For a dense 360-degree turntable that is far too coarse:
the intersection of 3 silhouettes of a clustered subject (the penguin family)
collapses to a blob. This carves from ALL the turntable azimuths at once, which
tightens the hull enough to bring out real shape (the adult's head/beak, the
separate chicks).

Assumptions (v1, orthographic turntable):
  * The frames are one continuous rotation about the vertical axis, captured in
    order, at ~even angular spacing -> azimuth_i = i * 360/N.
  * Each group mask is normalised to a common scale by OBJECT HEIGHT (the one
    dimension that is constant across azimuths), which also cancels the per-frame
    refraction magnification, and centred horizontally on the rotation axis
    (mask bbox centre).
  * Orthographic projection (object small in frame). Refraction still bends the
    silhouettes, so this is a soft hull — but far better than 3 views.

Masks: masks/<SUBJECT>/turntable/*.png  (0/255), sorted = rotation order.
Usage:  SUBJECT=snowglobe_interior venv/bin/python .../turntable_carve.py
        knobs: HEIGHT_VOX (256), MARGIN (0.08), SMOOTH (1.2 gaussian vox),
               HEIGHT_CM (7.0), FRONT_INDEX (0 = which mask faces -Z / 'front')
Output: work/<SUBJECT>_hull.glb
"""

import glob
import math
import os
import sys

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from skimage import measure

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBJECT  = os.environ.get('SUBJECT', 'snowglobe_interior')
MASK_DIR = os.path.join(BASE_DIR, 'masks', SUBJECT, 'turntable')
OUT_GLB  = os.path.join(BASE_DIR, 'work', f'{SUBJECT}_hull.glb')

HEIGHT_VOX  = int(os.environ.get('HEIGHT_VOX', '256'))   # voxels along height
MARGIN      = float(os.environ.get('MARGIN', '0.08'))     # horizontal padding (height units)
SMOOTH      = float(os.environ.get('SMOOTH', '1.2'))      # gaussian sigma (voxels)
HEIGHT_CM   = float(os.environ.get('HEIGHT_CM', '7.0'))
FRONT_INDEX = int(os.environ.get('FRONT_INDEX', '0'))


def load_normalised_silhouettes():
    """Return list of (azimuth_rad, sil bool[Hn,Wn]) — each mask rescaled so its
    object height = Hn, centred horizontally on its bbox centre, feet at row 0."""
    paths = sorted(glob.glob(os.path.join(MASK_DIR, '*.png')))
    if not paths:
        sys.exit(f"[turntable] no masks in {MASK_DIR}")
    import cv2
    Hn = HEIGHT_VOX
    Wn = int(round(Hn * (1 + 2 * (0.6 + MARGIN))))   # wide enough for any azimuth
    n = len(paths)
    sils, widths = [], []
    for i, p in enumerate(paths):
        m = cv2.imread(p, 0) > 127
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
        oh = r1 - r0 + 1
        scale = Hn / oh
        crop = m[r0:r1 + 1, c0:c1 + 1].astype(np.uint8) * 255
        new_w = max(1, int(round(crop.shape[1] * scale)))
        crop = cv2.resize(crop, (new_w, Hn), interpolation=cv2.INTER_NEAREST) > 127
        canvas = np.zeros((Hn, Wn), bool)
        x_off = Wn // 2 - new_w // 2
        x0 = max(0, x_off); xc0 = max(0, -x_off)
        w = min(new_w - xc0, Wn - x0)
        if w > 0:
            canvas[:, x0:x0 + w] = crop[:, xc0:xc0 + w]
        canvas = canvas[::-1]           # flip so row 0 = feet (bottom)
        az = 2 * math.pi * ((i - FRONT_INDEX) % n) / n
        sils.append((az, canvas))
        widths.append(new_w / Hn)
    print(f"[turntable] {len(sils)} views, azimuth step {360/n:.1f} deg, "
          f"width/height range {min(widths):.2f}-{max(widths):.2f}")
    return sils, Hn, Wn


def carve(sils, Hn, Wn):
    R = 0.6 + MARGIN                              # half-extent in height units
    nxz = int(round(2 * R * Hn))
    # grid coords (height units): Y up [0,1], X/Z in [-R, R]
    ax = (np.arange(nxz) + 0.5) / nxz * 2 * R - R
    ay = (np.arange(Hn) + 0.5) / Hn               # [0,1]
    X = ax[:, None]                               # (nx,1) over x
    Z = ax[None, :]                               # (1,nz) over z
    occ = np.ones((Hn, nxz, nxz), bool)
    vrow = np.clip((ay * (Hn - 1)).round().astype(int), 0, Hn - 1)   # (Hn,)
    for az, sil in sils:
        # horizontal image coord for each (x,z): u = x cos - z sin  (height units)
        u = X * math.cos(az) - Z * math.sin(az)                     # (nx,nz)
        ucol = np.clip((Wn / 2 + u * Hn).round().astype(int), 0, Wn - 1)
        # inside[y,x,z] = sil[vrow[y], ucol[x,z]]
        inside = sil[vrow[:, None, None], ucol[None, :, :]]
        occ &= inside
    print(f"[turntable] grid {Hn}x{nxz}x{nxz}, {occ.mean()*100:.1f}% occupied")
    return occ


def main():
    sils, Hn, Wn = load_normalised_silhouettes()
    occ = carve(sils, Hn, Wn)
    vol = occ.astype(np.float32)
    if SMOOTH > 0:
        vol = gaussian_filter(vol, SMOOTH)
    vol = np.pad(vol, 2)                          # closed border for marching cubes
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5)
    verts -= 2
    # axes: vol indexed [y, x, z] -> map to (x, y, z); scale height units -> cm
    vox_cm = HEIGHT_CM / Hn
    V = np.empty_like(verts)
    V[:, 0] = verts[:, 1] * vox_cm                # x
    V[:, 1] = verts[:, 0] * vox_cm                # y (up)
    V[:, 2] = verts[:, 2] * vox_cm                # z
    V *= 0.01                                     # cm -> metres (glTF)
    mesh = trimesh.Trimesh(vertices=V, faces=faces, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    comps = mesh.split(only_watertight=False)
    if len(comps) > 1:
        mesh = max(comps, key=lambda m: len(m.faces))   # keep largest body
    trimesh.smoothing.filter_taubin(mesh, iterations=8)
    b = mesh.bounds
    mesh.export(OUT_GLB)
    print(f"[turntable] wrote {OUT_GLB}")
    print(f"[turntable] {len(mesh.faces)} faces  watertight {mesh.is_watertight}  "
          f"bbox cm W{(b[1][0]-b[0][0])*100:.1f} H{(b[1][1]-b[0][1])*100:.1f} "
          f"D{(b[1][2]-b[0][2])*100:.1f}")


if __name__ == '__main__':
    main()
