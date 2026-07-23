"""Render scene presets for render_asset.py. Each preset module exposes a
`build(studio, glass, other, args)` that constructs its object-specific scene
(dome fit, interior seating, environmental dressing) using the object-agnostic
helpers in tools/render_studio.py, and returns the full list of meshes to frame.
A subject with no preset uses render_asset's general path instead."""
