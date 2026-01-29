# Prompt Documentation

This document records all prompts used throughout the data extraction, training, and evaluation pipeline.

## 1. Data Processing
*No LLM prompts were used for data cleaning; it was done via Python scripts/Regex.*

## 2. Training
**Instruction Format**: Mistral Instruct v0.2
```text
[INST] {instruction}

{input} [/INST] {output}
```
*Note: If `input` is empty, it is omitted.*

## 3. Evaluation (Inference)
**File**: `evaluation/run_eval.py`
**Function**: `generate_answer`
**Model**: Mistral-7B-Instruct-v0.2 (Base & Finetuned)

**Prompt Template**:
```text
[INST] Question: {question}

Answer concisely. If MCQ, output only the option letter. If Numeric, output only the number. [/INST]
```
*(Updated in Step 307 to force conciseness)*

## 4. Grading (Scorers)
**File**: `evaluation/scorers.py`
**Model**: Claude 3.5 Sonnet / Haiku

### MCQ Grader (`grade_mcq`)
**Goal**: Check if the student selected the correct option, ignoring verbosity.
**Prompt**:
```text
You are an impartial grader.
Task: Identify the selected option (A, B, C, or D) from the Student's Answer and compare it to the Correct Answer.

Student's Answer: "{predicted_text}"
Correct Answer: "{reference_answer}"

Instructions:
1. If the Student's Answer matches the Correct Answer (A, B, C, or D) or clearly states the text of the correct option, score 1.
2. If the answer is correct in concept but verbose, score 1.
3. Only score 0 if the answer is fundamentally wrong or selects a different option.
4. Output ONLY a valid JSON object with the format: {{"score": 0}} or {{"score": 1}}.
```

### Numeric Grader (`grade_numeric`)
**Goal**: Extract numeric value and check tolerance.
**Prompt**:
```text
You are an impartial grader.
Task: Extract the numeric value from the Student's Answer and determine if it matches the Correct Answer.

Student's Answer: "{predicted_text}"
Correct Answer: "{reference_answer}"
Tolerance: +/- {tolerance*100}%

Instructions:
1. Identify the final numeric answer in the Student's Answer. Ignore units unless they are fundamentally wrong.
2. Compare it to the Correct Answer value.
3. If the value is within {tolerance*100}% of the Correct Answer, score 1. Otherwise, score 0.
4. Output ONLY a valid JSON object with the format: {{"score": 0}} or {{"score": 1}}.
```

### Explanation Grader (`grade_explanation`)
**Goal**: Check semantic correctness.
**Prompt**:
```text
You are an expert Physics grader.
Task: Grade the Student's Explanation against the Reference Explanation.

Student's Answer: "{predicted_text}"
Reference Answer: "{reference_text}"

Rubric:
- 1.0: Correct. Captures the core physical concept. Ignore formatting/length.
- 0.0: Incorrect. Misses the main point or contains false statements.

Instructions:
1. Compare the core physical meaning.
2. Do not penalize for verbosity or formatting.
3. Output ONLY a valid JSON object with: {{"score": <float>, "reasoning": "<short text>"}}.
```
