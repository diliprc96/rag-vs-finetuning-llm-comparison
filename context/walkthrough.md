# Evaluation Walkthrough

## Goal
Compare the performance of:
1. **Base Model**: Mistral-7B-Instruct-v0.2
2. **Base + RAG**: Using OpenStax Physics context
3. **Finetuned Model**: LoRA adapter on Physics dataset
4. **Finetuned + RAG**: Combined approach

## Setup
- **Environment**: RunPod (Linux), Python 3.11, Nvidia GPU (24GB VRAM)
- **Dependencies**: `transformers`, `peft`, `langchain`, `faiss-cpu`, `anthropic`, `bitsandbytes`
- **Data**: 50 Physics Questions (35 MCQ/Numeric, 15 Explanation)
  - Source: Generated based on OpenStax University Physics Vol 1 criteria.
- **Scoring**:
  - MCQ: Exact match (A/B/C/D)
  - Numeric: Within 1% tolerance
  - Explanation: Graded by Claude (0-5 scale)

## Execution
Command (Sequential Execution due to Memory Constraints):
```bash
python -m evaluation.run_eval --mode base --rag
python -m evaluation.run_eval --mode finetuned
# Optimized for memory
python -m evaluation.run_eval --mode finetuned --rag
```

## Results

### Summary Table
| Configuration | MCQ Score | Numeric Score | Explanation Score |
| :--- | :--- | :--- | :--- |
| **Base** | **0.26** | 0.20 | **1.43** |
| Base + RAG | 0.24 | **0.22** | 1.29 |
| Finetuned | 0.14 | 0.12 | 1.22 |
| Finetuned + RAG | 0.12 | 0.08 | 1.23 |

### Key Observations
- **Base Model Superiority**: surprisingly, the Base Mistral-7B model outperformed the Finetuned model across almost all metrics. This strongly suggests that either:
  - The finetuning dataset size or quality was insufficient.
  - The LoRA parameters (rank, alpha) were not optimal.
  - The "catastrophic forgetting" phenomenon occurred where the model lost general reasoning abilities.
- **RAG Impact**: RAG provided a slight boost to **Numeric** questions for the Base model (0.20 -> 0.22) but generally did not drastically improve performance, and in some cases added noise.
- **Complexity**: The Finetuned + RAG configuration was the most resource-intensive, requiring memory optimization ($k=2$ retrieval) to run on a 24GB card.

### Next Steps 
- **Investigate Finetuning Data**: Review the training loss curves and the quality of the dataset used for the adapter.
- **Hyperparameter Tuning**: Experiment with different LoRA ranks or learning rates.
