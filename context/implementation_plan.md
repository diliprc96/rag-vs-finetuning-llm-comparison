# Implementation Plan: Unified Evaluation & Multi-Stage Training

### Phase 3: Evaluation Upgrades (COMPLETE)
- [x] Modify `scorers.py` to use pure Claude extraction.
- [x] Update `run_eval.py` to auto-version results.
- [x] Baseline Re-evaluation (50 Qs).
- [x] Data Analysis & Cleaning.

### Phase 4: Multi-Stage Training (COMPLETE)
- [x] Configure `train.py` to save checkpoints per epoch.
- [x] Train: 5 Epochs on **Cleaned Data**.
- [x] Evaluation: Epoch 3 (Poor scores due to format).
- [x] Evaluation: Epoch 5 (Poor scores due to format).

### Phase 5: Final Refinement (COMPLETE)
- [x] Create `prompts_documentation.md`.
- [x] Update scorers for relaxed grading (Focus on Correctness vs Format).
- [x] Upload Model to Hugging Face.
- [x] Final "Relaxed" Evaluation.

## User Review Required
> [!IMPORTANT]
> **Grading Logic**:
> - **MCQ/Numeric**: Binary scoring (0 or 1). All extraction and grading will be delegated to Claude (Haiku/Sonnet) to ignore formatting noise.
> - **Explanation**: 5-point normalized scale (0, 0.25, 0.5, 0.75, 1.0).
> - **Aggregation**: Final score will be the average of these normalized scores.

> [!WARNING]
> **Training**:
> - We will modify the training script to save checkpoints every epoch.
> - "Model A" will be the checkpoint at Epoch 3.
> - "Model B" will be the checkpoint at Epoch 5.

## Proposed Changes

### Evaluation Framework
#### [MODIFY] [scorers.py](file:///workspace/rag-vs-finetuning-llm-comparison/evaluation/scorers.py)
- **`grade_mcq`**: USE CLAUDE ONLY. Prompt: Extract answer, compare with key, return 0 or 1.
- **`grade_numeric`**: USE CLAUDE ONLY. Prompt: Extract value, compare with key (5% tolerance), return 0 or 1.
- **`grade_explanation`**: USE CLAUDE ONLY. Prompt: Grade on 5-point scale (0, 0.25, 0.5, 0.75, 1.0).
- **Versioning**: Ensure all outputs (`eval.log`, `results.csv`) are timestamped and preserved.

#### [MODIFY] [run_eval.py](file:///workspace/rag-vs-finetuning-llm-comparison/evaluation/run_eval.py)
- Integrate new scorers.
- Ensure outputs are saved to `evaluation/run_logs/` (new dir) to avoid clutter and ensure history is kept.

### Base Comparison Run (Immediate Action)
- **Goal**: Re-evaluate the *existing* "Base" and "Finetuned" (epoch 3/dry-run?) models using the new Claude-based metrics.
- **Scope**: All 50 Questions.
- **Comparison**: Generate a comparison table between `Old Metrics` vs `New Metrics`.

### Data Quality (Analysis Phase)
- **Action**: Inspect `data_extraction/alpaca_physics_5k_cleaned.jsonl` for specific patterns (e.g., JSON string vs plain text).
- **Deliverable**: A "Data Issues Report" listing specific examples of bad data.
- **Constraint**: **STOP** and ask user for approval on cleaning logic before running any cleaning script.

### Training Pipeline
#### [MODIFY] [train.py](file:///workspace/rag-vs-finetuning-llm-comparison/finetuning/train.py)
- Change `SFTConfig`:
    - Set `save_strategy="epoch"`.
    - Set `save_total_limit=10`.
    - Train for 5 epochs.
- We will evaluate checkpoints at Epoch 3 and Epoch 5.

## Verification Plan
### Step 1: Baseline Re-Evaluation
- Run `python -m evaluation.run_eval --mode all` with new metrics.
- Check `evaluation/run_logs/` for results.
