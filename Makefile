
.PHONY: help setup camera-calibration instance-segmentation depth-estimation measurement-extraction merge-views accuracy-validation diagnostic-overlay build-asset full-pipeline clean-images clean-frames clean-some-outputs clean-outputs clean-all

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

help:
	@echo "Available commands:"
	@echo "  make setup					* Create a local virtual environment and install dependencies"
	@echo "  make [specify_step]				* Run the specified step"
	@echo "  make full-pipeline				* Run the entire pipeline"
	@echo "  VIEW=side make measurement-extraction		* Measure a side-view capture set (default: front)"
	@echo "  make merge-views				* Combine front+side views into the final measurements JSON"
	@echo "  SUBJECT=x VIEW=side make diagnostic-overlay	* Render what the measurement step used (mask, bbox, endpoints, A4 quad)"
	@echo "  SUBJECT=x make build-asset			* Stage 3: build the real-world-scaled .glb from silhouette masks"
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

diagnostic-overlay:
	@echo "rendering measurement diagnostic overlays..."
	@$(PYTHON) measurement_extraction_step/diagnostic_overlay.py

accuracy-validation:
	@echo "running accuracy validation step..."
	@$(PYTHON) accuracy_validation_step/accuracy_validation.py

build-asset:
	@echo "building silhouette-hull mesh (Stage 3)..."
	@$(PYTHON) asset_generation_step/build_silhouette_mesh.py

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