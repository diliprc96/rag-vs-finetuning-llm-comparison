# Analysis Report: RAG vs Fine-tuning Comparison

## Executive Summary
The latest evaluation reveals that the **Finetuned model** is performing significantly worse than the **Base Mistral-7B model** across all metrics (MCQ, Numeric, Explanation). The results strongly suggest that the model currently being evaluated is **underfitted** or **defective**, likely due to being a "dry run" checkpoint (5 steps) rather than a fully trained model.

## Quantitative Results (Latest Run)

| Configuration | MCQ Score | Numeric Score | Explanation Score |
| :--- | :--- | :--- | :--- |
| **Base** | **0.26** | **0.20** | **1.43** |
| Base + RAG | 0.24 | 0.22 | 1.29 |
| Finetuned | 0.04 | 0.14 | 0.33 |
| Finetuned + RAG | 0.12 | 0.08 | 1.23* |

*Note: Older result from `results_summary.csv` for Finetuned+RAG. The latest `eval.log` focused on "finetuned" config with robust metrics.*

### Robustness Check (Phase 2)
The latest evaluation (from `eval.log` and `metric_robustness_report.md`) used robust metrics (REGEX extraction, Sonnet 4.5 grading) to rule out formatting errors:
- **MCQ**: **0.04** (Random chance is 0.25). The model is failing to output valid options entirely.
- **Numeric**: Improved to **0.14** due to better extraction, but still lags behind Base model (0.20).
- **Explanation**: **0.33**. The score dropped significantly with the stricter Sonnet 4.5 grader (previous 0.93 was likely inflated by Haiku).

## Diagnosis
1.  **Dry Run Hypothesis**: The `experiment_log.md` notes that a "Dry Run Verification" (max_steps 5) was performed. It is highly probable that this 5-step checkpoint returned to `results/` and was subsequently used for evaluation, overwriting any previous full training run. A 5-step model acts effectively as a random initialized adapter or worse, confusing the base model.
2.  **Formatting Issues**: The model outputs often resemble raw JSON or older dataset formats (e.g., `{"Answer": ...}`), indicating it has not learned the target plain-text format (`Explanation: ...\nAnswer: ...`) enforced in the cleaned dataset.
3.  **Data Quality**: While the dataset was cleaned (`alpaca_physics_5k_cleaned.jsonl`), the model likely didn't see enough of it (or any of it, if 5 steps) to adapt.

## Recommendations
1.  **Re-run Training**: Perform a full training run (3 epochs) using `finetuning/train.py`. Ensure `max_steps` is NOT set to 5.
2.  **Verify Loss**: Monitor the training loss to ensure it converges (starts ~1.10 and should drop).
3.  **Verify Checkpoint**: Ensure the `results/` directory contains the fully trained adapter before running evaluation.
4.  **Re-evaluate**: Run `evaluation/run_eval.py` again on the fully trained model.
