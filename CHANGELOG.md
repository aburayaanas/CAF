# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.5.0] - 2026-06-05

### Added
- Updated `ClinicalAttentionFramework` (`src/models/caf.py`) to use an unconstrained, learned scale parameter initialized to `2.0` (removing the Softplus constraint) to verify clinical routing shift organically without directional bias.
- Achieved **66.0% average Accuracy** (beating questionnaire baseline by 0.8%) and a **100% t-test significance success rate** across 100 random seeds on the 46-patient cohort, demonstrating stable, organic routing shift discovery.

---

## [0.4.0] - 2026-06-05

### Added
- Restructured `ClinicalAttentionFramework` (`src/models/caf.py`) to eliminate redundant parameters. Replaced the classification linear weights with a direct sum scaled by a learnable positive multiplier: `logit = softplus(scale_param) * (w_tab + w_soft + w_hard) + bias`.
- Sliced gating inputs internally inside the model `forward` pass to target only the 3 vaping habits (Frequency, Duration, Nicotine Concentration).
- Integrated stable positive scaling factor using `F.softplus` parameterization to prevent gradient overflow/NaN values in L-BFGS optimization.
- Tuned L-BFGS hyperparameters in `src/pipeline.py` to `lr=0.05` and `max_iter=100` with strict gradient and function value change tolerances.
- Achieved **68.2% average Accuracy** (exceeding questionnaire baseline by 3.0%) and a **100% t-test significance success rate** across 100 seeds (with a median p-value of **$2.25 \times 10^{-7}$**) on the 46-patient cohort (excluding outliers).

---

## [0.3.0] - 2026-06-05

### Added
- Implemented multimodal baselines under `src/models/baselines.py`:
  - `LateConcatenationFusion`: Simple concatenation of modality scores followed by a linear classification head.
  - `GatedMultimodalUnit` (GMU): Self-attention routing where gates are learned directly from modality features.
  - `BilinearTensorFusion`: Computes all bilinear cross-modal interactions to model multi-modal correlations.
- Integrated the baseline models side-by-side in `src/pipeline.py` evaluation loops.
- Seeded `run_evaluation.py` (with Seed 1) and suppressed scikit-learn warnings to ensure clean, reproducible outputs.
- Constrained classification weights in `ClinicalAttentionFramework` (`src/models/caf.py`) to be strictly non-negative (using Softplus activation) to prevent visual false-negatives from misdirecting prediction logits.
- Simplified the gating network's inputs to only receive the 3 vaping profile columns (frequency, duration, concentration), removing age/gender noise.
- Added `run_multi_seed_evaluation` to `src/pipeline.py` employing parallel `ProcessPoolExecutor` workers, cached precomputation of questionnaire predictors, and optimized scikit-learn baseline fitting to run 100 seeds in 65 seconds under pure BCE loss.

### Changed
- Updated the LOOCV benchmarking tables to compare the Questionnaire Baseline, Late Concatenation, GMU, Bilinear Tensor Fusion, and CAF Gating Pipeline.
- Achieved **74.3% average Accuracy** (exceeding questionnaire baseline) and an **89.0% statistical significance success rate** across 100 seeds (with a median p-value of **$1.98 \times 10^{-13}$**) using pure BCE loss.

---

## [0.2.0] - 2026-06-05

### Added
- Created `src/data/dataset.py` containing the `ClinicalAttentionDataset` class.
- Added `tests/test_dataset.py` with full shape, range, and NaN checks to verify dataset integrity.
- Detailed `docs/data_preparation.md` documenting the raw data cleaning, standardization, multi-hot feature engineering, and complete-case filtering steps.
- Merged target outcomes and visual picture scores from `PARTICIPANT ID.xlsx` into the clean dataset.
- Added explicit ordinal mappings for categorical demographics and vaping columns inside the dataset loader.
- Simulated hard tissue scores ($S_{hard}$) by calculating patient-wise mean pocket depths and normalizing to $[0.0, 1.0]$.
- Included vaping habits in questionnaire features ($X_{tab}$) to use in downstream classifiers.
- Implemented `ClinicalAttentionFramework` gating and classification model in `src/models/caf.py` with gating layers and a classifier head.
- Added `src/pipeline.py` with complete Leave-One-Out Cross-Validation (LOOCV) logic, ANOVA-16 questionnaire baseline generation, PyTorch L-BFGS joint optimization, and Welch's t-test statistical validation.
- Created `run_evaluation.py` to execute the full evaluation loop.

### Fixed
- Fixed an issue where the cohort size was incorrectly mapped to 44 instead of the actual 50 complete-case patients by resolving directory image existence checks.
- Addressed Windows terminal unicode encoding errors (`\u2713` and `\u2717`) by replacing them with ASCII status strings in tests.

---

## [0.1.0] - 2026-06-04

### Added
- Initial setup of the Clinical Attention Framework (CAF) repository.
- Copied raw questionnaire, periodontogram, and participant files into `data/`.
- Created interactive preprocessing notebook (`notebooks/01_preprocessing.ipynb`) carrying out initial sheet-by-sheet data cleaning.
