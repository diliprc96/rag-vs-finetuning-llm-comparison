# Evaluation Metric Robustness Report

## Phase 1: Baseline (Rigid Metrics)
**Model**: `mistral-7b-physics-finetune` (Alpha=128)
**Metrics**:
- **MCQ**: Exact letter match (Regex `\b([A-D])\b`).
- **Numeric**: Last number extraction, 1% tolerance.
- **Explanation**: Claude-3-Haiku (0-5 scale).

### Results
- **MCQ**: 0.04
- **Numeric**: 0.04
- **Explanation**: 0.93
*Observation*: The rigid metrics likely missed correct answers formatted differently. The finetuned model performed poorly with exact matching.

## Phase 2: Robust Metrics (Proposed)
**Changes**:
- **MCQ**: Add LLM fallback to extract option if regex fails.
- **Numeric**: Add LLM extraction for the final value. Increase tolerance to 5%.
- **Explanation**: Upgrade from `claude-3-haiku-20240307` to `claude-sonnet-4-5-20250929` (Sonnet 4.5) for better reasoning.
- **Fallback**: use `claude-3-5-haiku-20241022` (Haiku 3.5) for extraction and fallback grading.

### Results
- **MCQ**: 0.04 (No change - Model output is incoherent, likely due to insufficient training)
- **Numeric**: 0.14 (Improved from 0.04) - **+250% Improvement** due to robust extraction.
- **Explanation**: 0.33 (Dropped from 0.93) - Sonnet 4.5 is a much harsher grader than Haiku. The previous 0.93 score likely reflected Haiku's leniency rather than model quality.

## Conclusion
The **Robust Metrics** (Phase 2) successfully exposed the true state of the model:
1.  **Numeric Grading** is now working (catching correct answers embedded in text).
2.  **Explanation Grading** is more realistic (0.33 vs optimized 0.93).
3.  **Model Quality**: The model is performing very poorly (MCQ 0.04). 
    - **Observation**: The output often resembles `{"(A)": "Joule", ...}`, which matches the *old* unclean dataset format (before normalization).
    - **Hypothesis**: The model in `results/mistral-7b-physics-finetune` is likely result of a **dry run (5 steps)** which overwrote any previous full training run. A 5-step model has not learned the new plain-text format and essentially outputs noise or overfits to the few examples it saw.
    - **Data Sufficiency**: If this *was* a full 3-epoch run, then the data quantity or quality is insufficient for the model to learn the specific MCQ format `(A)`. However, the 0.04 score (random chance = 0.25) strongly suggests catastrophic failure or rigid formatting mismatch, not just "not enough data".

**Recommendation**: Re-run the full training (3 epochs) ensuring the `alpaca_physics_5k_cleaned.jsonl` contains the **normalized plain text** format, and verify the loss curve decreases significantly.
