
.PHONY: help setup camera-calibration instance-segmentation depth-estimation measurement-extraction merge-views accuracy-validation full-pipeline clean-images clean-frames clean-some-outputs clean-outputs clean-all

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

accuracy-validation:
	@echo "running accuracy validation step..."
	@$(PYTHON) accuracy_validation_step/accuracy_validation.py

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
	@rm -rf depth_estimation_step/output/*
	@rm -rf measurement_extraction_step/output/*

clean-outputs:
	@echo "cleaning all output directories..."
	@rm -rf camera_calibration_step/output/*
	@rm -rf instance_segmentation_step/output/*
	@rm -rf depth_estimation_step/output/*
	@rm -rf measurement_extraction_step/output/*

light-clean: clean-frames clean-some-outputs

deep-clean: clean-images light-clean
	@echo "cleaning complete"