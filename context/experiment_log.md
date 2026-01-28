# Experiment Log

## Experiment 1: Data Standardization & Hyperparameter Tuning
**Date**: 2026-01-28
**Goal**: Improve finetuning performance by fixing data inconsistencies and increasing LoRA scaling.

### 1. Data Analysis (Pre-Fix)
- **Input**: `data_extraction/alpaca_physics_5k_cleaned.jsonl`
- **Issue**: Found mixed types.
    - ~4044 samples are plain strings.
    - ~353 samples were stringified JSONs.
- **Action**: Standardized all outputs to plain text format: `Explanation: ... \nAnswer: ...` (or similar).

### 2. Hyperparameter Updates
- **File**: `finetuning/train.py`
- **Change**:
    - `lora_alpha`: Increased from `16` to `128`.
    - `r`: Kept at `64`.
    - **Resulting Scaling**: $\alpha/r = 2.0$ (previously 0.25).
    - **Reasoning**: Stronger LoRA updates required to adapt to the new domain.

### 3. Execution Results
- **Dry Run Verification**: 
    - Ran `finetuning/train.py` with `--max_steps 5`.
    - **Status**: Success.
    - **Observations**: Model loaded, loss calculation worked ~1.10. Saved output to `results/`.
    - **Ready for Full Training**: Yes.

### 4. Backup
- **Uploaded to**: `Dilip-Ravi/mistral-7b-physics-finetuned`
- **Script**: `finetuning/upload_model.py`

### 5. Evaluation Fixes
- **JSON Error**: Fixed `Invalid \escape` in Sonnet response by using Regex score extraction (`r'"score":\s*([0-9.]+)'`) instead of strict JSON parsing.
- **Logging**: Added `evaluation/eval.log` to capture all evaluation outputs.
- **Resumability**: Updated `run_eval.py` to skip already completed questions if interrupted.

### 6. Robust Evaluation Results
- **Outcome**:
    - **MCQ**: 0.04 (Random noise).
    - **Numeric**: 0.14 (Improved due to regex).
    - **Explanation**: 0.33 (Low score with Sonnet 4.5).
- **Hypothesis**: The model checkpoint evaluated was likely the **5-step dry run** (overwriting any previous full run) or the model failed to learn.
- **Action**: Recommended re-running full training (3 epochs) on clean data.

### 7. Backup & Migration Prep
- **Versioning**: Enabled timestamped logging in `run_eval.py` (`eval_{timestamp}.log`, `results_table_{timestamp}.csv`) to preserve raw outputs.
- **Backup**: Created zip archive of workspace (Code + Data + Results/Adapter) for migration.

