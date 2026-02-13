# Project Chronological Log: RAG vs. Finetuning (Physics Domain)

**Purpose**: This document serves as a detailed, linear history of all technical decisions, experiments, failures, and improvements made during the project. It provides the raw material for the dissertation by documenting the *process* rather than just the final result.

---

## 0. Phase 0: Data Sourcing & Pre-processing (The Origin Story)
**Goal**: Create a domain-specific dataset from a Physics Textbook.
**Process**:
*   **Source**: OpenStax Physics Vol 1 (Chapters 1-6).
*   **Extraction**: Used `chunk_and_generate.py` to chunk the textbook text.
*   **Generation**: Used Claude to generate Q&A pairs from each chunk (`raw_claude_responses.jsonl`).
*   **Refinement (LaTeX Cleaning)**:
    *   **Issue**: The raw extraction contained broken LaTeX formatting (e.g., `$(1.4Ã—{10}^{21}\text{mi})`).
    *   **Action**: Implemented `latex_refiner.py` to normalize equations and text encoding.
    *   **Result**: Produces `alpaca_physics_5k_no_latex.jsonl`.
    *   *Note*: This step fixed the *input* text but left some *output* format inconsistencies (JSON vs Text) which were discovered later in Phase 2.

## 1. Initial State & Legacy Evaluation
**Approach**: 
We started with a standard "LLM-as-a-Judge" setup using a 0-5 scalar metric.
*   **Model**: Mistral-7B-Instruct-v0.2 (Base).
*   **Grader**: Early prompt asking for a score from 0 to 5.
*   **Dataset**: `alpaca_physics_5k.jsonl`.

**Result**: 
*   **Issue**: The 0-5 scores were highly inconsistent. A "3" and a "4" were often indistinguishable. The prompt instructions were too vague, leading to noise in the baseline.
*   **Assessment**: We could not reliably measure improvement if the ruler itself was broken.

**Improvement**: 
*   **Action**: Pivoted to a **Binary/Discrete Framework** using a more powerful judge (**Claude 3.5 Sonnet**).
*   **New Logic**:
    *   **MCQ**: 0 or 1 (Strict selection).
    *   **Numeric**: 0 or 1 (Tolerance +/- 5%).
    *   **Explanation**: 0.0, 0.25, 0.5, 0.75, 1.0 (Rubric-based).

---

## 2. Baseline Evaluation & Data Discovery
**Approach**: 
We re-ran the baseline using the new Claude Sonnet scorers.
*   **Configuration**: Base Mistral + RAG (Retrieve top-3 docs).

**Result**: 
*   **Base Scores**: ~0.28 MCQ / ~0.24 Numeric.
*   **RAG Scores**: ~0.30 MCQ (Slight improvement).
*   **Observation**: The RAG model was stable and followed instructions well.

**Unexpected Discovery (The "Bad Data" Event)**:
While preparing for finetuning, we analyzed the training data `alpaca_physics_5k.jsonl`.
*   **Finding**: A script `analyze_data.py` revealed that **8% (353 samples)** of the output fields were formatted as **JSON strings** (e.g., `{"explanation": "..."}`), while 92% were plain text.
*   **Why it matters**: This inconsistency confuses the model. It learns that "sometimes I should speak JSON" but doesn't know when.

**Improvement**: 
*   **Action**: Created `clean_dataset.py`.
*   **Logic**: Regex parsing to extract the `explanation` or text value from the JSON strings and replace the original field.
*   **Outcome**: A "Cleaned" dataset where 100% of outputs were natural language.

---

## 3. Finetuning Experiment (Multi-Stage)
**Approach**: 
We hypothesized that correcting the data format would lead to a high-performing model.
*   **Method**: QLoRA Finetuning on `alpaca_physics_5k_cleaned.jsonl`.
*   **Schedule**: Train for 5 Epochs, saving checkpoints at every epoch to monitor convergence.
*   **Evaluation Plan**: Evaluate "Early" (Epoch 3) and "Converged" (Epoch 5) models.

**Result**: 
*   **Epoch 3 Scores**: MCQ ~0.02.
*   **Epoch 5 Scores**: MCQ ~0.02.
*   **Diagnosis**: The results were significantly **worse** than the Base model (0.28).
    *   **Manual Inspection**: We looked at the raw prediction logs.
    *   **Failure Mode**: The model was outputting responses like:
        > *Question*: "Which is an SI unit? (C) Kilogram"
        > *Answer*: "The units listed in the table are SI base units." (Correct content, but verbose).
        > *Answer*: `{"solution": "C"}` (JSON Hallucination).
    *   **Conclusion**: Despite cleaning the data, the model had "overfit" to the verbose/specific style of the training data and lost the zero-shot ability to follow "Answer with only the letter" instructions.

---

## 4. Mitigation & Final "Relaxed" Evaluation
**Approach**: 
Since the model *knew* the physics (as seen in the logs) but failed the format, we needed to adjust the ruler.
*   **Action**: Implemented **Relaxed Grading** in `scorers.py`.
*   **New Logic**: 
    *   If the model output contains the correct option text (e.g. "Kilogram") or the letter ("C") *anywhere* in the response, count it as Correct (1.0).
    *   Do not penalize for verbosity.

**Result (Final)**: 
*   **MCQ**: Jumped from 0.02 -> **0.14**.
*   **Numeric**: 0.04.
*   **Explanation**: 0.10.

**Final Conclusion**: 
Even with relaxed grading, the Finetuned model (0.14) performed significantly worse than the Base Model (0.28) and RAG (0.30).
*   **Takeaway**: For this codebase/dataset, **RAG is superior**. Finetuning caused "catastrophic forgetting" of instruction-following capabilities, likely because the dataset (even after cleaning) was too small or stylistically different from the validation questions.

---

## 5. Artifacts Created & Stored
All relevant files have been moved to the `context/` folder for your repository:
*   `prompts_documentation.md`: The exact system prompts used.
*   `dissertation_seed_report.md`: The summary report.
*   `model_behavior_report.md`: Specific examples of the model's failures.
