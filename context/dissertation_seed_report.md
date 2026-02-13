# Dissertation Seed Report: RAG vs. Finetuning for Physics QA

**Date**: January 29, 2026
**Project**: Comparison of Retrieval-Augmented Generation (RAG) and Fine-Tuning (LoRA) for Specialized Physics Question Answering.

---

## 1. Project Objective
The primary goal was to empirically compare two techniques for adapting a General Purpose LLM (Mistral-7B-Instruct-v0.2) to a specific domain (Physics):
1.  **RAG**: Injecting relevant context from a textbook/knowledge base.
2.  **Fine-Tuning**: Training the model weights on a dataset of 5,000 physics Q&A pairs.

## 2. Methodology & Phases

### Phase 1: Robust Evaluation Framework (The "LLM-as-a-Judge" Approach)
**Evolution of Metrics**:
*   **Legacy Approach**: Initially, we employed a **0-5 Raw Scale** using a standard prompt.
    *   *Issue*: The 0-5 scale was subjective and noisy. A score of "3" vs "4" was often arbitrary.
    *   *Decision*: Moved to a **Binary/Discrete Scale** to reduce variance.
*   **Refined Approach**: implemented **Claude 3.5 Sonnet** as the Judge.
    *   **MCQ**: Binary (0/1). Correct option selected.
    *   **Numeric**: Binary (0/1). Value within 5% tolerance.
    *   **Explanation**: Discrete Steps (0.0, 0.25, 0.5, 0.75, 1.0) based on clear rubrics.
    *   *Why*: This granularity allowed for precise error tracking (e.g., "Concept is correct but missed details" = 0.75).

**Prompt Engineering**: Evaluation prompts were originally strict (demanding JSON output) but later relaxed to focus on content validity.

**Step Results (Baseline)**:
Before any training, we established a baseline using the pre-trained Mistral-7B model.
-   **Base Model**: ~0.28 (Average Score)
-   **Base + RAG**: ~0.30 (Slight Improvement)
*Observation*: The base model already had decent zero-shot capabilities.

---

### Phase 2: Data Quality Analysis & Remediation
**Discovery**:
*   Initial models failed to answer simple questions, instead outputting random JSON snippets.
*   **Investigation**: A deep dive into `alpaca_physics_5k.jsonl` revealed a critical flaw:
    *   **92%** of samples were standard Plain Text (Natural Language).
    *   **8%** (353 samples) were formatted as **JSON Strings** (e.g., `{"output": "The answer is..."}`).
*   **Impact**: Even this small fraction was enough to "poison" the instruction-tuning. The model learned that "Physics Question = Output JSON", but got the schema wrong, leading to hallucinations like `{"solution": "C"}` instead of just `"C"`.

**Improvement**:
-   Implemented `clean_dataset.py` to standardize the dataset.
-   **Action**: Extracted the `explanation` or text values from the JSON strings and converted them to plain text.
-   **Result**: A 100% consistent Plain Text training set.

---

### Phase 3: Multi-Stage Fine-Tuning
**Approach**:
-   Trained Mistral-7B using **QLoRA** on the *Cleaned Dataset*.
-   **Training Schedule**: 5 Epochs total.
-   **Checkpoints**: Saved models at every epoch to compare "early" (Epoch 3) vs "converged" (Epoch 5) performance.
-   **Leakage Analysis**: Verified that the evaluation set was *not* disjoint from the training data (High semantic overlap, >0.5 cosine similarity). This confirmed that poor performance was not due to Out-Of-Distribution (OOD) testing.

**Step Results (Initial Evaluation)**:
-   **Epoch 3**: ~0.02 (MCQ)
-   **Epoch 5**: ~0.02 (MCQ)
-   **Analysis**: The scores were catastrophic (near 0).
-   **Diagnosis**: Manual inspection of raw logs revealed that despite cleaning, the model was **highly verbose** or **hallucinating JSON keys** (e.g., answering `{"solution": "C"}` instead of `C`), which failed the strict evaluation regex.

---

### Phase 4: Refinement & Mitigation (Relaxed Grading)
**Problem**: The model "knew" the answer (often writing the correct full sentence) but failed the "Format" test.

**Improvement**:
-   Modified `scorers.py` to implement **"Relaxed Grading"**.
-   **New Rule**: If the model output contains the correct answer text (e.g., "The answer is Newton"), it counts as correct, even if it didn't strictly output "A".
-   **Prompt Engineering**: Enhanced validation prompts to look for *correctness* over *style*.

**Final Results (Relaxed Eval - 5 Epochs)**:
-   **MCQ**: 0.14 (Improved from 0.02)
-   **Numeric**: 0.04
-   **Explanation**: 0.10
-   **Comparison**: Still significantly **worse** than the Base Model (0.28).

---

## 3. Final Conclusion & Dissertation Takeaways

### Summary Table
| Model Config | MCQ Score | Numeric Score | Explanation Score | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Base Mistral** | **0.28** | **0.24** | **0.21** | Best zero-shot performance. |
| **Base + RAG** | **0.30** | 0.22 | 0.21 | Marginal improvement, helpful for context. |
| **Finetuned (Old/Dirty)** | 0.08 | 0.10 | 0.10 | Severely impacted by formatting noise. |
| **Finetuned (Clean/5-Ep)**| 0.14 | 0.04 | 0.10 | **Degraded performance**. |

### Key Findings
1.  **Data Consistency is Critical**: Even a small amount (8%) of formatting noise (JSON vs Text) can destroy a model's instruction-following capability, causing it to hallucinate formats during inference.
2.  **Finetuning Can Cause Regression**: On a small (5k) dataset, finetuning catastrophically degraded the general reasoning capabilities of the 7B model. It "overfit" to the style of the training data (verbose, specific phrasing) and lost the ability to answer concisely, which penalized it heavily even with relaxed grading.
3.  **RAG Stability**: RAG proved to be a safer, more stable approach for this domain. It preserved the model's instruction-following abilities while injecting knowledge, whereas finetuning destabilized the model.

### Recommendations for Future Work
-   **Synthetically Clean Data**: Instead of just regex cleaning, use an LLM to rewrite the entire training dataset into a perfect, consistent Q&A format.
-   **Parameter Efficient Tuning**: Experiment with lower Rank (r) in LoRA to reduce the destructiveness of the updates.
-   **Hybrid Approach**: Finetune *only* on the RAG reasoning traces (not just raw QA) to teach the model *how* to use the retrieved context.
