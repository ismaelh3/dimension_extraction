#!/usr/bin/env python
"""
build.py — ONE-COMMAND asset build, driven by a per-subject manifest.

    SUBJECT=x make build

  * If subjects/<x>.yaml exists  -> run the whole pipeline from it.
  * If it doesn't               -> a short WIZARD asks a few class/geometry
    questions, WRITES subjects/<x>.yaml, then builds from it.

This collapses the old ~8-command, ~15-env-knob flow (segment -> measure ->
carve/generate -> texture -> material -> assemble -> render) into a single
command. The manifest is the one source of truth: reproducible, diffable,
editable, and CI-friendly.

Design:
  * object CLASS (opaque_solid | transparent_hollow | transparent_container)
    selects a PROFILE = which stages run + fixed knobs (glass? assemble?).
    This is where "is it reflective? is there an object inside?" land.
  * GEOMETRY (carve | generative) chooses silhouette carving vs TripoSR (for
    clusters / thin / hard-to-carve shapes where a visual hull fails).
  * AUTO-DETECTED, never asked: face/texture budgets from the measured size;
    DEPTH_CARVE from the cavity probe (carve geometry only).
  * `overrides:` in the manifest holds only the genuinely empirical values
    (SIDE_FROM, interior ORIENT, INTERIOR_FILL...) — passed to steps verbatim.

Non-interactive wizard (automation/tests): set ANS_* env vars (see run_wizard).
Force a full rebuild ignoring existing outputs with FORCE=1.
"""

import glob
import json
import os
import shutil
import subprocess
import sys

import yaml

BASE = os.path.dirname(os.path.abspath(__file__))          # asset_generation_step/
REPO = os.path.dirname(BASE)
PY = sys.executable
sys.path.insert(0, os.path.join(BASE, "pipeline"))
import glassiness  # noqa: E402
SUBJECTS_DIR = os.path.join(BASE, "subjects")
WORK = os.path.join(BASE, "work")
MASKS = os.path.join(BASE, "masks")
DEFAULT_FRAMES = os.path.join(REPO, "instance_segmentation_step", "frames")

SEG      = os.path.join(REPO, "instance_segmentation_step", "segmentation.py")
SEG_OUT  = os.path.join(REPO, "instance_segmentation_step", "output")
DEPTH    = os.path.join(REPO, "depth_estimation_step", "depth_estimation.py")
DEPTH_JSON = os.path.join(REPO, "depth_estimation_step", "output", "depth_results.json")
MEASURE_ALL = os.path.join(REPO, "measurement_extraction_step", "measure_all.py")
CARVE    = os.path.join(BASE, "pipeline", "build_silhouette_mesh.py")
GENERATE = os.path.join(BASE, "pipeline", "generate_interior.py")
TEXTURE  = os.path.join(BASE, "pipeline", "texture_hull.py")
MATERIAL = os.path.join(BASE, "pipeline", "material_pass.py")
ASSEMBLE = os.path.join(BASE, "pipeline", "assemble_container.py")
RENDER   = os.path.join(BASE, "tools", "render_preview.py")

# class -> which stages run + fixed knobs. "is it reflective?/object inside?"
# choose the class; the class chooses the flow.
PROFILES = {
    "opaque_solid":         {"glass": False, "assemble": False, "render_scene": "none"},
    "transparent_hollow":   {"glass": True,  "assemble": False, "render_scene": "none"},
    "transparent_container": {"glass": True, "assemble": True,  "render_scene": "none"},
}


# --------------------------------------------------------------------------- #
# wizard
# --------------------------------------------------------------------------- #
def ask(question, options=None, default=None, env=None):
    """Ask once. An ANS_* env var (or no-TTY + default) answers non-interactively."""
    if env and os.environ.get(env) is not None:
        return os.environ[env]
    if not sys.stdin.isatty():
        if default is not None:
            return default
        sys.exit(f"[build] non-interactive and no default — set {env}")
    tail = f" [{'/'.join(options)}]" if options else ""
    tail += f" (default: {default})" if default is not None else ""
    while True:
        a = input(f"  {question}{tail}: ").strip() or (default or "")
        if not options or a in options:
            return a
        print(f"    choose one of: {', '.join(options)}")


def run_wizard(subject):
    print(f"[build] no manifest for '{subject}' — quick setup:")
    prompt = ask("What is it? (segmentation prompt, e.g. 'perfume bottle.')",
                 default="product.", env="ANS_PROMPT")
    # NOTE: "is it transparent/reflective?" is NOT asked — it's auto-detected
    # from the captures (glassiness.py: depth see-through + specular) and written
    # back as the class. Set ANS_CLASS to force it; edit the manifest to
    # transparent_container for the rare shell-with-a-separate-interior case.
    klass = os.environ.get("ANS_CLASS", "auto")
    geometry = ask("Geometry: silhouette carve, or generative (clusters/thin/"
                   "hard-to-carve)?", ["carve", "generative"], default="carve",
                   env="ANS_GEOMETRY")
    multi = ask("Segment MULTIPLE instances and merge (a family/set)?",
                ["y", "n"], default="n", env="ANS_MULTI") == "y"

    manifest = {
        "subject": subject,
        "prompt": prompt,
        "class": klass,                    # 'auto' -> resolved by glassiness
        "geometry": geometry,
        "multi_instance": multi,
        "interior": {"mode": "none"},
        "frames_dir": os.environ.get("FRAMES_DIR", DEFAULT_FRAMES),
        "render_scene": os.environ.get("ANS_SCENE", "none"),
        "overrides": {},
    }
    os.makedirs(SUBJECTS_DIR, exist_ok=True)
    path = os.path.join(SUBJECTS_DIR, f"{subject}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    print(f"[build] wrote {os.path.relpath(path, REPO)}:\n"
          + "".join("    " + ln for ln in open(path).readlines()))
    return manifest


def load_manifest(subject):
    path = os.path.join(SUBJECTS_DIR, f"{subject}.yaml")
    if os.path.exists(path):
        print(f"[build] using {os.path.relpath(path, REPO)}")
        with open(path) as f:
            return yaml.safe_load(f)
    return run_wizard(subject)


# --------------------------------------------------------------------------- #
# auto-detection
# --------------------------------------------------------------------------- #
def auto_budgets(subject):
    """Scale face/texture budgets from the measured real-world size, if a
    measurements JSON exists. Best-effort: any problem -> no budget override
    (the stage defaults apply)."""
    try:
        mj = os.path.join(REPO, "measurement_extraction_step", "output",
                          f"measurements_{subject}.json")
        if not os.path.exists(mj):
            return {}
        d = json.load(open(mj))
        dims = d.get("dimensions_cm") or d.get("dimensions") or {}
        vals = [v for v in dims.values() if isinstance(v, (int, float))]
        if not vals:
            return {}
        mx = max(vals)
        faces = int(min(80000, max(20000, mx * 4000)))
        print(f"[build] auto budgets from measured {mx:.1f} cm -> "
              f"TARGET_FACES={faces}")
        return {"TARGET_FACES": str(faces), "TEXTURE_SIZE": "2048"}
    except Exception as e:  # noqa: BLE001
        print(f"[build] auto-budget skipped ({e})")
        return {}


# --------------------------------------------------------------------------- #
# step runner
# --------------------------------------------------------------------------- #
def step(script, env_extra, label):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_extra.items()})
    print(f"\n=== [build] {label} ===")
    r = subprocess.run([PY, script], env=env, cwd=REPO)
    if r.returncode != 0:
        sys.exit(f"[build] FAILED at: {label}")


def have(subject, suffix):
    return os.path.exists(os.path.join(WORK, subject + suffix))


def _front_src(frames):
    s = os.path.join(frames, "front")
    return s if os.path.isdir(s) else frames


def resolve_class(manifest):
    """Auto-detect the class when it is 'auto' (glassiness: depth see-through +
    specular). Ensures a front-view glass map exists (segment + depth the front
    frames if missing), scores it, and writes the resolved class back to the
    manifest. transparent_container stays a manual manifest choice."""
    if manifest.get("class", "auto") != "auto":
        return manifest["class"]
    subj = manifest["subject"]
    frames = manifest.get("frames_dir", DEFAULT_FRAMES)
    ov = {k: str(v) for k, v in (manifest.get("overrides") or {}).items()}
    front = os.path.join(MASKS, subj, "front")
    if not glob.glob(os.path.join(front, "*_glass.png")):
        src = _front_src(frames)
        step(SEG, {"SUBJECT": subj, "PRODUCT_PROMPT": manifest["prompt"],
                   "MULTI_INSTANCE": "1" if manifest["multi_instance"] else "0",
                   "FRAMES_DIR": src, "OUTPUT_DIR": SEG_OUT, **ov},
             "auto-detect: segment front")
        os.makedirs(front, exist_ok=True)
        for f in glob.glob(os.path.join(src, "*.*")):
            stem = os.path.splitext(os.path.basename(f))[0]
            m = os.path.join(SEG_OUT, f"{stem}_product_mask.png")
            if os.path.exists(m):
                shutil.copy(m, os.path.join(front, f"{stem}_product_mask.png"))
        depth_json = DEPTH_JSON
        try:                                    # depth = the strong see-through cue
            step(DEPTH, {**ov}, "auto-detect: depth front")
        except SystemExit:
            depth_json = None                   # no A4 -> specular-only fallback
        glassiness.file_view_glass_maps(src, front, depth_json)
    score, kind = glassiness.object_score(subj, MASKS)
    resolved = "transparent_hollow" if kind == "transparent" else "opaque_solid"
    print(f"[build] AUTO-DETECT: glassiness {score:.2f} → {kind} → "
          f"class={resolved}")
    manifest["class"] = resolved
    with open(os.path.join(SUBJECTS_DIR, f"{subj}.yaml"), "w") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False)
    return resolved


def build(manifest):
    manifest["class"] = resolve_class(manifest)
    subj = manifest["subject"]
    klass = manifest["class"]
    if klass not in PROFILES:
        sys.exit(f"[build] unknown class '{klass}' (have {list(PROFILES)})")
    prof = PROFILES[klass]
    frames = manifest.get("frames_dir", DEFAULT_FRAMES)
    ov = {k: str(v) for k, v in (manifest.get("overrides") or {}).items()}
    force = os.environ.get("FORCE") == "1"
    budgets = auto_budgets(subj)

    print(f"[build] {subj}: class={klass} geometry={manifest['geometry']} "
          f"multi={manifest['multi_instance']} glass={prof['glass']} "
          f"assemble={prof['assemble']}")

    multi = "1" if manifest["multi_instance"] else "0"
    # 1 + 2. SEGMENT + GEOMETRY (path depends on the geometry source)
    if manifest["geometry"] == "generative":
        # a single front view is enough for the learned prior. Source = the
        # front/ subfolder if the capture uses per-view folders, else flat.
        src = os.path.join(frames, "front")
        if not os.path.isdir(src):
            src = frames
        front = os.path.join(MASKS, subj, "front")
        if force or not (os.path.isdir(front) and os.listdir(front)):
            step(SEG, {"SUBJECT": subj, "PRODUCT_PROMPT": manifest["prompt"],
                       "MULTI_INSTANCE": multi, "FRAMES_DIR": src,
                       "OUTPUT_DIR": front, **ov}, "segment (front view)")
        else:
            print("[build] segment: masks present, skipping")
        if force or not have(subj, "_hull.glb"):
            step(GENERATE, {"SUBJECT": subj, "INTERIOR_VIEW": "front",
                            "FRAMES_DIR": frames, **ov},
                 "geometry — generative (TripoSR)")
    else:  # carve — ALL views from a SINGLE upload (per-view subfolders)
        mj = os.path.join(REPO, "measurement_extraction_step", "output",
                          f"measurements_{subj}.json")
        if force or not os.path.exists(mj):
            step(MEASURE_ALL, {"SUBJECT": subj, "FRAMES_ROOT": frames,
                               "PRODUCT_PROMPT": manifest["prompt"],
                               "MULTI_INSTANCE": multi, **ov},
                 "segment+depth+measure ALL views (single upload) + merge")
        else:
            print("[build] measure: measurements present, skipping")
        if force or not have(subj, "_hull.glb"):
            step(CARVE, {"SUBJECT": subj, "CROSS_SECTION": "silhouette",
                         **budgets, **ov}, "geometry — carve")

    # 3. TEXTURE
    if force or not have(subj, "_textured.glb"):
        step(TEXTURE, {"SUBJECT": subj, "FRAMES_DIR": frames,
                       "DESPECKLE": ov.get("DESPECKLE", "1.2"),
                       **budgets, **ov}, "texture")

    # 4. MATERIAL (glass regions) — transparent classes only
    if prof["glass"]:
        if force or not have(subj, "_final.glb"):
            step(MATERIAL, {"SUBJECT": subj, **ov}, "material — glass regions")

    # 5. ASSEMBLE (transparent_container: shell + separate interior)
    if prof["assemble"]:
        interior_glb = ov.get("INTERIOR_GLB") or os.path.join(
            WORK, f"{subj}_interior_textured.glb")
        if not os.path.exists(interior_glb):
            print(f"[build] NOTE: transparent_container needs a built interior "
                  f"at {os.path.basename(interior_glb)} — build the interior "
                  f"subject '{subj}_interior' first, or set overrides.INTERIOR_GLB. "
                  f"Skipping assemble.")
        elif force or not have(subj, "_assembled.glb"):
            step(ASSEMBLE, {"SUBJECT": subj, "INTERIOR_GLB": interior_glb, **ov},
                 "assemble")

    # 6. RENDER
    scene = manifest.get("render_scene", "none")
    step(RENDER, {"SUBJECT": subj, "SCENE": scene, **ov}, f"render (scene={scene})")

    out = "_assembled.glb" if prof["assemble"] and have(subj, "_assembled.glb") \
        else "_final.glb" if prof["glass"] and have(subj, "_final.glb") \
        else "_textured.glb"
    print(f"\n[build] DONE — deliverable: work/{subj}{out}, "
          f"render: work/{subj}_render.png")


def main():
    subject = os.environ.get("SUBJECT")
    if not subject:
        sys.exit("[build] set SUBJECT (e.g. SUBJECT=perfume-bottle make build)")
    build(load_manifest(subject))


if __name__ == "__main__":
    main()
