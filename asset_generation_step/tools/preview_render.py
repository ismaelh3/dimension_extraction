"""
Honest z-buffered software renderer for .glb previews.

matplotlib's Poly3DCollection (used by the sweep scripts' quick renders) has
no depth buffer — it depth-sorts whole triangles, and on dense near-flat
regions that z-fighting draws contour-band artifacts that look exactly like
geometry ridges. This module rasterizes with a real per-pixel z-buffer and
barycentric interpolation, so what you see is what the mesh actually is.
It is also the preview path for textured meshes (matplotlib can't sample a
texture per pixel at all).

Orthographic camera, +Y-up world (glTF convention), lambert + headlight.

Usage (module):
    from preview_render import raster_preview
    img = raster_preview(mesh)                       # PIL.Image, flat gray
    img = raster_preview(mesh, vertex_colors=rgba)   # painted vertices
    img = raster_preview(mesh, uv=uv, texture=pil)   # textured

Usage (CLI):
    venv/bin/python asset_generation_step/tools/preview_render.py in.glb out.png
"""

import numpy as np
from PIL import Image

BG = 255  # white background


def _view_matrix(azim_deg, elev_deg):
    """Rotation taking world coords to camera coords (camera looks down -Z),
    matching fidelity_sweep.render's matplotlib view convention."""
    az, el = np.radians(azim_deg), np.radians(elev_deg)
    # yaw about world Y, then pitch about camera X
    yaw = np.array([[np.cos(az), 0, np.sin(az)],
                    [0, 1, 0],
                    [-np.sin(az), 0, np.cos(az)]])
    pitch = np.array([[1, 0, 0],
                      [0, np.cos(el), np.sin(el)],
                      [0, -np.sin(el), np.cos(el)]])
    return pitch @ yaw


def raster_preview(mesh, size=760, azim=25, elev=-14,
                   vertex_colors=None, uv=None, texture=None,
                   shade_mix=0.45):
    """Render mesh to a PIL RGB image. Color source precedence:
    uv+texture > vertex_colors > flat gray. shade_mix controls how much
    lambert shading modulates the color (0 = unlit, 1 = fully lit)."""
    R = _view_matrix(azim, elev)
    verts = mesh.vertices @ R.T
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    center, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.08
    scale = (size - 1) / (2 * half)
    # pixel coords: x right, y down; z increases toward the viewer
    px = (verts[:, 0] - center[0] + half) * scale
    py = (center[1] + half - verts[:, 1]) * scale
    pz = verts[:, 2]

    normals = mesh.face_normals @ R.T
    light = np.array([0.35, 0.45, 0.82])
    light /= np.linalg.norm(light)
    lam = np.clip(normals @ light, 0, 1)
    shade = (1 - shade_mix) + shade_mix * lam

    if texture is not None:
        tex = np.asarray(texture.convert('RGB'), dtype=np.float32)
        th, tw = tex.shape[:2]
    vc = None if vertex_colors is None else \
        np.asarray(vertex_colors)[:, :3].astype(np.float32)

    img = np.full((size, size, 3), float(BG), np.float32)
    zbuf = np.full((size, size), -np.inf, np.float32)

    tri_px = np.stack([px[mesh.faces], py[mesh.faces]], axis=2)  # (F, 3, 2)
    # back-to-front isn't needed with a z-buffer; skip back faces for speed
    front = normals[:, 2] > 0
    order = np.flatnonzero(front)

    for f in order:
        (x0, y0), (x1, y1), (x2, y2) = tri_px[f]
        xmin, xmax = int(min(x0, x1, x2)), int(np.ceil(max(x0, x1, x2)))
        ymin, ymax = int(min(y0, y1, y2)), int(np.ceil(max(y0, y1, y2)))
        if xmax < 0 or ymax < 0 or xmin >= size or ymin >= size:
            continue
        xmin, ymin = max(xmin, 0), max(ymin, 0)
        xmax, ymax = min(xmax, size - 1), min(ymax, size - 1)
        gx, gy = np.meshgrid(np.arange(xmin, xmax + 1),
                             np.arange(ymin, ymax + 1))
        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(det) < 1e-12:
            continue
        w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / det
        w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / det
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        i0, i1, i2 = mesh.faces[f]
        z = w0 * pz[i0] + w1 * pz[i1] + w2 * pz[i2]
        ys, xs = gy[inside], gx[inside]
        zi = z[inside]
        closer = zi > zbuf[ys, xs]
        if not closer.any():
            continue
        ys, xs, zi = ys[closer], xs[closer], zi[closer]
        b0, b1, b2 = w0[inside][closer], w1[inside][closer], w2[inside][closer]
        if texture is not None and uv is not None:
            u = b0 * uv[i0, 0] + b1 * uv[i1, 0] + b2 * uv[i2, 0]
            v = b0 * uv[i0, 1] + b1 * uv[i1, 1] + b2 * uv[i2, 1]
            tx = np.clip((u * (tw - 1)).astype(np.int32), 0, tw - 1)
            ty = np.clip(((1 - v) * (th - 1)).astype(np.int32), 0, th - 1)
            col = tex[ty, tx]
        elif vc is not None:
            col = (b0[:, None] * vc[i0] + b1[:, None] * vc[i1]
                   + b2[:, None] * vc[i2])
        else:
            col = np.full((len(ys), 3), 205.0, np.float32)
        zbuf[ys, xs] = zi
        img[ys, xs] = col * shade[f]

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


if __name__ == '__main__':
    import sys
    import trimesh
    mesh = trimesh.load(sys.argv[1], force='mesh')
    vc = None
    if hasattr(mesh.visual, 'vertex_colors'):
        vc = np.asarray(mesh.visual.vertex_colors)
        if len(np.unique(vc.reshape(-1, vc.shape[-1]), axis=0)) <= 1:
            vc = None  # flat single color: render as plain gray
    raster_preview(mesh, vertex_colors=vc).save(sys.argv[2])
    print(f"wrote {sys.argv[2]}")
