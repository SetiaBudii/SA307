# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] - 2025-04-05

### Added
- 04-04 - feat: add configuration files for SAM 2 model
- 04-03 - feat: add init file and dummy file
- 04-03 - feat: implement fine-tuning training loop with checkpointing and logging
- 04-03 - feat: update model configuration in config.yml with new checkpoint path
- 04-01 - feat: update fine-tuning parameters in config.yml
- 04-01 - feat: add new dataset processing scripts and configuration files
- 03-29 - feat: add example  image and mask files for dataset 3050
- 03-19 - feat: add git ignore

### Fixed
- 04-03 - fix: update setuptools version range in pyproject.toml
- 03-14 - fix: agriculture counter and add split data program

### Changed
- 04-03 - refactor: Re-implement fine-tuning script and utility functions for SAM 2 model
- 04-01 - refactor: change Sam 2 model
- 03-18 - Refactor: Remove deprecated requirements.txt on preprocessing folder
- 03-18 - Refactor: move requirement to root folder

### Changed
- 03-18 - chore: rename folder SAM 2 to SAM2

### Misc
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

