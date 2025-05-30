# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] - 2025-05-30

### Added
- 04-05 - feat: add COCO annotation creation script and update config for dataset paths
- 04-04 - feat: add configuration files for SAM 2 model
- 04-03 - feat: add init file and dummy file
- 04-03 - feat: implement fine-tuning training loop with checkpointing and logging
- 04-03 - feat: update model configuration in config.yml with new checkpoint path
- 04-01 - feat: update fine-tuning parameters in config.yml
- 04-01 - feat: add new dataset processing scripts and configuration files
- 03-29 - feat: add example  image and mask files for dataset 3050
- 03-19 - feat: add git ignore

### Fixed
- 05-01 - fix: correct epochs value in fine-tuning configuration
- 04-03 - fix: update setuptools version range in pyproject.toml
- 03-14 - fix: agriculture counter and add split data program

### Changed
- 05-12 - Refactor: Remove unused file
- 05-01 - refactor: use utility function for saving model checkpoints
- 04-30 - refactor: change testing flow
- 04-03 - refactor: Re-implement fine-tuning script and utility functions for SAM 2 model
- 04-01 - refactor: change Sam 2 model
- 03-18 - Refactor: Remove deprecated requirements.txt on preprocessing folder
- 03-18 - Refactor: move requirement to root folder

### Changed
- 03-18 - chore: rename folder SAM 2 to SAM2

### Misc
- 05-30 - add: update configuration for fine-tuning parameters, create final testing setup, and implement directional point generation
- 05-25 - delete unused file
- 05-25 - add: implement inference scripts for image comparison and batch testing, update fine-tuning utilities
- 05-25 - add: implement npc hsv version
- 05-25 - add: implement HSV negative prompt augmentation for image processing
- 05-25 - add: refactor fine-tuning v3
- 05-25 - add: add new example images and masks to datasets
- 05-20 - add: cp to dir
- 05-20 - add: new npc and read single function
- 05-16 - add: implement new test script v4
- 05-16 - add: implement new test script v3
- 05-16 - add: implement new inference script for SAM2 model with segmentation and visualization
- 05-16 - update: refactor validate_model function to accept log file path and remove unused logging functionality
- 05-16 - update: refactor fine_tune_utils.py and add inference_utils.py for mask visualization functions
- 05-16 - add: fine tuning v3
- 05-16 - update: remove obsolete image files 3050.png and 3050_mask.png
- 05-12 - update: remove obsolete files including .gitignore, LICENSE, README.md, inference script, and requirements.txt
- 05-12 - update: implement new  model validation function with loss calculation and logging
- 05-12 - update: add new testting script version
- 05-12 - update: add new version read data
- 05-12 - update: add inference script for SAM2 model with visualization and evaluation features
- 05-12 - update: initialize neg_points and neg_labels arrays in neg_prompt_calibration function
- 05-12 - update: add fine-tuning v3 script with training and validation logic for SAM2 model
- 05-12 - update: correct checkpoint path and update dataset directories in config.yml
- 05-12 - update: add kagglehub to requirements
- 05-10 - update: refactor waterbodies_extraction function to accept image directly and modify testing file
- 05-10 - update: implement waterbodies extraction v2
- 05-09 - update: modify save_ckpts function to accept checkpoint path and refactor validate_model to use read_data
- 05-09 - update: add checkpoint loading functionality in main testing script
- 05-09 - update: add image_path parameter and import necessary modules for neg_prompt_calibration function
- 05-09 - update: enhance training logging and checkpointing in fine-tuning process
- 05-09 - delete: remove unused testnpc.py file
- 05-08 - Merge pull request #5 from SetiaBudii/feature-npc
- 05-05 - update: modify npc
- 05-04 - add: implement validation functionality and update configuration for validation datasets
- 05-04 - add: implement patch extraction and saving functionality in Jupyter notebook
- 05-01 - add: neg prompt calibration utils
- 05-01 - add: implement fine-tuning v2
- 04-30 - add: implement waterbodies extraction and mask refinement functions
- 04-30 - Merge pull request #4 from SetiaBudii/feature-samaug
- 04-30 - add: implement max distance and random point from samaug
- 04-30 - Merge pull request #3 from SetiaBudii/feature-npc
- 04-30 - add: implement testing script with IoU calculation and dataset loading (early)
- 04-30 - add: update requirements.txt to include additional dependencies
- 04-14 - add: new read data function
- 04-10 - Merge pull request #2 from SetiaBudii/feature-npc
- 04-07 - Create discord-notify.yml
- 04-05 - Initialize PointSAM for NPC
- 04-05 - add CHANGELOG.md and script to generate it from git log
- 04-01 - Merge pull request #1 from SetiaBudii/feature-npc
- 03-29 - Test implement NPC
- 03-12 - add preprocessing dataset
- 03-12 - Merge branch 'main' of https://github.com/SetiaBudii/SA307
- 03-12 - Add model preparation and optimizer functions for training
- 02-25 - Add model preparation and data handling functions for training
- 02-18 - Add utility functions for image and mask processing in fine-tuning
- 02-18 - Update download_ckpts.sh to enable checkpoint downloads for SAM 2
- 02-18 - Add Sam File
- 02-18 - first commit

