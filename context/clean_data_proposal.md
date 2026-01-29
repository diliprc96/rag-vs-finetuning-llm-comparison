# Data Cleaning Proposal

## Findings
Analysis of `data_extraction/alpaca_physics_5k_cleaned.jsonl` reveals inconsistent output formatting:
- **Total Samples**: 4397
- **Plain Text Outputs**: 4044 (92.0%)
- **JSON String Outputs**: 353 (8.0%)

### Example of Inconsistency
**Text Sample**:
```json
{"output": "The law of conservation of energy states..."}
```

**JSON Sample**:
```json
{"output": "{\"explanation\": \"The text explains that...\", \"commonality\": \"Both contain energy...\"}"}
```

## Proposed Logic
We will standardize all outputs to **Plain Text**. This matches the majority of the dataset and is better for a standard QA/Instruction model.

**Transformation Rules**:
1. Iterate through all samples.
2. Attempt to parse `output` as JSON.
3. If valid JSON:
    - If `explanation` key exists exists -> Use `explanation` value.
    - If `answer` key exists -> Use `answer` value.
    - If both/others -> Join values with newlines or prioritize `explanation`.
4. If not JSON (already text) -> Keep as is.

## Expected Outcome
- 100% of samples will have `output` as a plain string.
- This will stabilize the training target and likely improve model performance significantly (model heavily penalized currently for bad formatting).
