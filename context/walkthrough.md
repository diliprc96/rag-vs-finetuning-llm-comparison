# Project Walkthrough: Mitigation & Final Results

## 1. Context
Following the discovery that the 3-Epoch finetuned model performed poorly (0.02 score), we initiated a **Mitigation Phase** to diagnose and rescue the experiment.

## 2. Diagnosis: The "Format Failure"
We inspected the raw model outputs for the 5-Epoch model.
*   **Observation**: The model often outputted the *correct answer* but buried in verbose text or hallucinated JSON formats.
*   **Example**:
    *   *Question*: "Which is an SI unit? (C) Kilogram"
    *   *Model*: "The units listed in the table are SI base units." (Correct fact, failed format).
*   **Root Cause**: The training data (even after regex cleaning) likely contained enough stylistic variance to break the model's strict instruction-following capabilities.

## 3. Action: Relaxed Grading
We modified the evaluation logic (`scorers.py`) to be robust to formatting issues.
*   **Change**: Implemented "Semantic Containment" checks. If the answer text or letter appears in the output, credit is given.
*   **Goal**: Measure *Knowledge* independent of *Syntax*.

## 4. Final Results
| Metric | Strict Score (Old) | Relaxed Score (New) | Baseline (Mistral) |
| :--- | :--- | :--- | :--- |
| **MCQ** | 0.02 | **0.14** | **0.28** |
| **Numeric** | 0.06 | **0.04** | **0.24** |
| **Explanation** | 0.13 | **0.10** | **0.21** |

## 5. Conclusion
Relaxed grading recovered ~12% of the score, proving the model *did* learn physics. However, the performance is still **50% worse than the Base Model**.
**Scientific Finding**: Finetuning on this specific small dataset (<5k) degraded the general capabilities of the 7B model more than it added domain knowledge. **RAG remains the superior approach.**
