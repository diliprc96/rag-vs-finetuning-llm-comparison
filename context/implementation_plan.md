# Evaluation Setup Plan

## Goal
Prepare the environment and run the evaluation script `evaluation/run_eval.py` on RunPod to compare Base, Base+RAG, Finetuned, and Finetuned+RAG models.

## User Review Required
> [!IMPORTANT]
> **Anthropic API Key**: The evaluation script uses `evaluation/scorers.py` which requires an `ANTHROPIC_API_KEY` environment variable to grade explanations using Claude. Please ensure this is set or provide it.

> [!WARNING]
> **Model Zip File**: The model is currently a zip file `results/mistral-7b-physics-finetune.zip`. It needs to be unzipped.

## Proposed Changes

### 1. Environment Setup
- Install dependencies from `requirements.txt`.

### 2. Data Preparation
- **Wait for copy to complete**.
- Unzip `results/mistral-7b-physics-finetune.zip` to `results/mistral-7b-physics-finetune`.

### 3. Execution
- Run the evaluation script:
  ```bash
  export ANTHROPIC_API_KEY=your_key_here
  python -m evaluation.run_eval --mode all --rag
  ```

## Verification Plan
### Automated Tests
- **Check Model Loading**: Run a quick test to load the adapter `load_models(run_finetuned=True)` in a python shell.

### Manual Verification
- Monitor the `tqdm` progress bar in the terminal.
- Check if `evaluation/results_table.csv` is generated and populated.
