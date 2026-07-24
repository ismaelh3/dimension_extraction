
.PHONY: help setup camera-calibration instance-segmentation depth-estimation measurement-extraction merge-views accuracy-validation diagnostic-overlay measure-all build build-asset color-asset texture-asset material-asset setup-triposr generate-interior assemble-asset render-asset render-glass test-asset full-pipeline clean-images clean-frames clean-some-outputs clean-outputs clean-all

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

help:
	@echo "Available commands:"
	@echo "  make setup					* Create a local virtual environment and install dependencies"
	@echo "  SUBJECT=x make build				* ONE-COMMAND build: wizard asks a few questions, writes subjects/x.yaml, runs the whole pipeline"
	@echo "  make [specify_step]				* Run an individual step (advanced/manual)"
	@echo "  make full-pipeline				* Run the entire pipeline"
	@echo "  VIEW=side make measurement-extraction		* Measure a side-view capture set (default: front)"
	@echo "  make merge-views				* Combine front+side views into the final measurements JSON"
	@echo "  SUBJECT=x VIEW=side make diagnostic-overlay	* Render what the measurement step used (mask, bbox, endpoints, A4 quad)"
	@echo "  SUBJECT=x make build-asset			* Stage 3: build the real-world-scaled .glb from silhouette masks"
	@echo "  SUBJECT=x_interior make generate-interior	* Stage 3: GENERATE an interior extraction can't recover (TripoSR)"
	@echo "  SUBJECT=x make assemble-asset			* Stage 3: assemble a transparent-container deliverable GLB"
	@echo "  SUBJECT=x [SCENE=preset] make render-asset	* Stage 3: honest ray-traced product render (Blender/Cycles)"
	@echo "  make clean-all    				* Remove build artifacts and clear outputs"

setup: requirements.txt
	@echo "setting up virtual environment and installing dependencies..."
	@python3 -m venv $(VENV)
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

camera-calibration:
	@echo "running camera calibration step..."
	@$(PYTHON) camera_calibration_step/camera_calibration.py

instance-segmentation:
	@echo "running instance segmentation step..."
	@$(PYTHON) instance_segmentation_step/segmentation.py

depth-estimation:
	@echo "running depth estimation step..."
	@$(PYTHON) depth_estimation_step/depth_estimation.py

measurement-extraction:
	@echo "running measurement step..."
	@$(PYTHON) measurement_extraction_step/measurement_extraction.py

merge-views:
	@echo "merging per-view measurements..."
	@$(PYTHON) measurement_extraction_step/merge_views.py

measure-all:
	@echo "measuring ALL views from a single upload (per-view subfolders under FRAMES_ROOT)..."
	@echo "use case: SUBJECT=x FRAMES_ROOT=path/to/<subject> make measure-all   (expects <subject>/front/ <subject>/side/ ...)"
	@$(PYTHON) measurement_extraction_step/measure_all.py

diagnostic-overlay:
	@echo "rendering measurement diagnostic overlays..."
	@$(PYTHON) measurement_extraction_step/diagnostic_overlay.py

accuracy-validation:
	@echo "running accuracy validation step..."
	@$(PYTHON) accuracy_validation_step/accuracy_validation.py

build:
	@echo "one-command asset build (wizard writes subjects/<SUBJECT>.yaml, then runs the whole pipeline)..."
	@echo "use case: SUBJECT=perfume-bottle make build"
	@$(PYTHON) asset_generation_step/build.py

build-asset:
	@echo "building silhouette-hull mesh (Stage 3)..."
	@$(PYTHON) asset_generation_step/pipeline/build_silhouette_mesh.py

setup-triposr:
	@echo "vendoring + patching TripoSR (generative fallback)..."
	@$(PYTHON) asset_generation_step/tools/setup_triposr.py

generate-interior:
	@echo "generating an interior extraction can't recover (TripoSR, Stage 3)..."
	@echo "use case: SUBJECT=object_interior make generate-interior"
	@$(PYTHON) asset_generation_step/pipeline/generate_interior.py

color-asset:
	@echo "painting vertex colors from capture photos (Stage 3, M2)..."
	@$(PYTHON) asset_generation_step/pipeline/color_hull.py

texture-asset:
	@echo "baking UV texture from capture photos (Stage 3, M2 v2)..."
	@$(PYTHON) asset_generation_step/pipeline/texture_hull.py

material-asset:
	@echo "splitting PBR materials on the textured asset (Stage 3, M2 v3)..."
	@$(PYTHON) asset_generation_step/pipeline/material_pass.py

assemble-asset:
	@echo "assembling the transparent-container deliverable GLB (Stage 3)..."
	@echo "use case: SUBJECT=object make assemble-asset"
	@$(PYTHON) asset_generation_step/pipeline/assemble_container.py

render-asset:
	@echo "rendering an honest ray-traced product render in Blender/Cycles..."
	@echo "use case: SUBJECT=object [SCENE=snowglobe] make render-asset"
	@$(PYTHON) asset_generation_step/tools/render_preview.py

render-glass:
	@echo "rendering the snowglobe glass preset (shim -> render-asset SCENE=snowglobe)..."
	@echo "use case: SUBJECT=snowglobe make render-glass"
	@$(PYTHON) asset_generation_step/tools/glass_preview.py

test-asset:
	@echo "testing the generated asset"
	@echo "use case: SUBJECT=object_name FACE_BUDGETS=####,####,#### make test-asset"
	@$(PYTHON) asset_generation_step/analysis/fidelity_sweep.py

most-pipeline: instance-segmentation depth-estimation measurement-extraction merge-views
	@echo "running the majority of the pipeline"

full-pipeline: camera-calibration instance-segmentation depth-estimation measurement-extraction merge-views accuracy-validation
	@echo "running full pipeline complete"

clean-images:
	@echo "cleaning images directory..."
	@rm -rf camera_calibration_step/calibration_images/*

clean-frames:
	@echo "cleaning frames directory..."
	@rm -rf instance_segmentation_step/frames/*

clean-assets:
	@echo "cleaning assets directory..."
	@rm -rf asset_generation_step/work/lods/*

clean-some-outputs:
	@echo "cleaning some output directories..."
	@rm -rf instance_segmentation_step/output/*
	@rm -rf instance_segmentation_step/__pycache__
	@rm -rf depth_estimation_step/output/*
	@rm -rf depth_estimation_step/__pycache__
	@rm -rf measurement_extraction_step/output/*
	@rm -rf measurement_extraction_step/__pycache__
	@rm -rf accuracy_validation_step/__pycache__

clean-outputs:
	@echo "cleaning all output directories..."
	@rm -rf camera_calibration_step/output/*
	@rm -rf instance_segmentation_step/output/*
	@rm -rf depth_estimation_step/output/*
	@rm -rf measurement_extraction_step/output/*

light-clean: clean-frames clean-some-outputs

deep-clean: clean-images light-clean
	@echo "cleaning complete"