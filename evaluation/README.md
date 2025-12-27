Evaluation scaffolding for OpenStax-based physics QA experiments

Overview
- Implements RAG indexing and retrieval with sentence-transformers + FAISS.
- Provides numeric scoring utilities and a rubric-based scoring harness.
- Provides a small `harness.py` to run four experimental setups (placeholders for model calls).

Quick start
1. Create a Python venv and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r evaluation/requirements.txt
```

2. Build FAISS index (see `rag.py`), populate `evaluation/questions_sample.jsonl`.
3. Plug your model inference functions into `harness.py` and run:

```bash
python evaluation/harness.py --questions evaluation/questions_sample.jsonl
```

Files
- `rag.py` — index builder & retriever utilities
- `score.py` — numeric and rubric scoring helpers (includes LLM judge wrapper)
- `harness.py` — orchestrates experiment runs and metrics
- `questions_sample.jsonl` — sample questions to test the pipeline

Notes
- The harness contains placeholders for model inference. Replace `run_base_model` and `run_finetuned_model` with your inference code (local or API-based).
- The judge function uses OpenAI if an API key is provided; otherwise a simple heuristic fallback is used for automated testing.