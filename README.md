# RAG vs Fine-Tuning: A Comparative Study for Domain-Specific LLM Adaptation

> **Can retrieval-augmented generation match or outperform fine-tuning for specializing a general-purpose LLM?**

This study compares **Retrieval-Augmented Generation (RAG)** and **Fine-Tuning (FT)** as strategies for adapting [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) to a domain-specific Physics Question Answering task. Using the [OpenStax College Physics](https://openstax.org/details/books/college-physics-2e) textbook as the knowledge source, I find that **RAG consistently outperforms fine-tuning**, and that even small data-quality issues can cause catastrophic regression during fine-tuning.

---

## Table of Contents

- [Key Findings](#key-findings)
- [Results](#results)
- [Methodology](#methodology)
- [Evaluation Framework](#evaluation-framework)
- [Critical Discovery: Data Quality](#critical-discovery-data-quality)
- [Repository Structure](#repository-structure)
- [Technologies](#technologies)
- [How to Reproduce](#how-to-reproduce)
- [Future Work](#future-work)
- [License](#license)

---

## Key Findings

| # | Finding | Detail |
|---|---------|--------|
| 1 | **RAG outperforms Fine-Tuning** | RAG preserves the base model's instruction-following ability while injecting domain knowledge at inference time. Fine-tuning degraded both formatting and accuracy. |
| 2 | **Data quality is paramount** | An 8% contamination of JSON-formatted strings in an otherwise plain-text training set caused *catastrophic forgetting* — the fine-tuned model began hallucinating JSON and lost instruction-following capabilities. |
| 3 | **Syntax ≠ Semantics** | The fine-tuned model often *knew* the correct physics but couldn't express it in the required format. Relaxed grading (ignoring formatting) recovered some scores, but RAG still won. |
| 4 | **Small-data fine-tuning is risky** | With only ~5,000 training samples and one domain, QLoRA fine-tuning caused regression — the model forgot general capabilities faster than it learned specialized ones. |

---

## Results

### Aggregate Scores (50-question benchmark, Claude 3.5 Sonnet judge)

| Configuration | MCQ Score | Numeric Score | Explanation Score |
|---------------|:---------:|:-------------:|:-----------------:|
| **Base (Mistral-7B)** | **0.26** | **0.20** | **1.43** |
| **Base + RAG** | 0.24 | **0.22** | 1.29 |
| Fine-Tuned (relaxed grading) | 0.14 | 0.12 | 1.22 |
| Fine-Tuned + RAG | 0.12 | 0.08 | 1.23 |

> **Reading the table:** MCQ and Numeric scores are on a 0–1 scale (proportion correct). Explanation scores are on a 0–5 scale (semantic quality).

### Key Observations

- **Base + RAG** achieved the best numeric score (0.22), demonstrating that retrieved context helps with calculation-heavy questions.
- **Fine-tuning degraded performance across all metrics**, even with relaxed grading that strips formatting requirements.
- Adding RAG to the fine-tuned model did *not* recover performance — the underlying instruction-following damage could not be compensated for.

---

## Methodology

The study was conducted in five phases:

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4
 Data        Eval       Data        Fine-       Relaxed
 Pipeline    Framework  Quality     Tuning      Evaluation
```

### Phase 0 — Data Extraction Pipeline

Built an automated pipeline to extract and transform physics content:
1. **Web Crawler** (`data_extraction/data_crawler.py`) — scrapes OpenStax textbook (Chapters 1–6)
2. **Chunk & Generate** (`data_extraction/chunk_and_generate.py`) — uses Claude API to convert text chunks into Alpaca-format instruction/output pairs
3. **Output** — ~5,000 QA pairs in `alpaca_physics_5k.jsonl`

### Phase 1 — Evaluation Framework

Created a 50-question benchmark (`evaluation/physics_questions_50.json`) covering three question types:

| Type | Description | Scoring |
|------|-------------|---------|
| **MCQ** | Multiple choice (A/B/C/D) | Binary (0 or 1) |
| **Numeric** | Calculation with ±5% tolerance | Binary (0 or 1) |
| **Explanation** | Free-form physics explanation | Graded (0.0 – 5.0) |

All scoring is performed by **Claude 3.5 Sonnet** as an LLM judge, using carefully designed grading prompts (documented in `context/prompts_documentation.md`).

### Phase 2 — Data Quality Investigation

Analysis revealed that **8% of training samples** had JSON-formatted output strings while the remaining 92% were plain text. This inconsistency was traced to the Claude API responses during data generation.

**Impact:** Mixed formats forced the model to learn two incompatible output distributions simultaneously, leading to erratic behavior and JSON hallucinations.

**Fix:** `clean_dataset.py` normalizes all outputs to plain text → `alpaca_physics_5k_cleaned.jsonl`

### Phase 3 — Fine-Tuning

Fine-tuned Mistral-7B using **QLoRA** (4-bit quantization + Low-Rank Adaptation):

| Parameter | Value |
|-----------|-------|
| Base Model | Mistral-7B-Instruct-v0.2 |
| Method | QLoRA (4-bit NF4) |
| LoRA Rank | 64 |
| LoRA Alpha | 128 |
| Epochs | 5 |
| Learning Rate | 2e-4 |
| Batch Size | 4 (effective 16 with gradient accumulation) |
| Platform | RunPod (NVIDIA RTX4090 24GB) |

Training used the cleaned dataset with epoch-level checkpointing and best-model selection based on validation loss.

### Phase 4 — Relaxed Evaluation

After the fine-tuned model showed poor results, I introduced **relaxed grading** to separate formatting failures from knowledge failures:
- MCQ: Accept correct answer even without standard A/B/C/D format
- Numeric: Extract numeric values from verbose responses
- Explanation: Grade semantic content regardless of structure

**Result:** Relaxed grading improved the fine-tuned model's MCQ score from 0.02 to 0.14 — confirming it had *some* physics knowledge but had lost the ability to format responses correctly. RAG still outperformed.

---

## Evaluation Framework

```
┌────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌───────────┐
│  50-Question │────►│  Model Under │────►│  Claude 3.5      │────►│  Scored   │
│  Benchmark   │     │  Test (±RAG) │     │  Haiku (Judge)  │     │  Results  │
└────────────┘     └──────────────┘     └──────────────────┘     └───────────┘
```

- **Question bank:** Hand-curated from OpenStax Chapters 1–6 (`evaluation/physics_questions_50.json`)
- **RAG retrieval:** FAISS index over textbook chunks (`rag_pipeline/`)
- **Judge model:** Claude 3.5 Haiku scores each response against the reference answer
- **Outputs:** Per-question CSVs (`evaluation/results_*.csv`) and an aggregate summary

---

## Critical Discovery: Data Quality

The most impactful finding was not about model architecture, but about **training data quality**.

```
Training Data (alpaca_physics_5k.jsonl)
├── 92% samples → Plain text output     ✓ Consistent
└──  8% samples → JSON-formatted output  ✗ Contamination
```

**What happened:**
- The Claude API occasionally returned responses wrapped in JSON objects during data generation
- These 8% of samples created a competing output distribution
- During fine-tuning, the model tried to learn both formats, resulting in:
  - ❌ JSON hallucinations in free-text responses
  - ❌ Loss of instruction-following capability
  - ❌ Catastrophic forgetting of general language abilities

**Lesson:** For small datasets (~5K samples), even minor format inconsistencies can dominate the learning signal and cause catastrophic failure. Data quality auditing should be a mandatory step before any fine-tuning run.

---

## Repository Structure

```
rag-vs-finetuning-llm-comparison/
├── data_extraction/              # Phase 0: Data pipeline
│   ├── data_crawler.py           # OpenStax web scraper
│   ├── chunk_and_generate.py     # Claude-based QA pair generation
│   ├── alpaca_physics_5k.jsonl   # Original dataset (with contamination)
│   ├── alpaca_physics_5k_cleaned.jsonl  # Cleaned dataset
│   └── latex_refiner.py          # LaTeX normalization utility
│
├── evaluation/                   # Phase 1 & 4: Evaluation system
│   ├── physics_questions_50.json # 50-question benchmark
│   ├── scorers.py                # Claude-based grading (strict + relaxed)
│   ├── run_eval.py               # Evaluation orchestrator
│   ├── rag_utils.py              # RAG integration for evaluation
│   ├── results_summary.csv       # Aggregate results table
│   ├── results_base.csv          # Per-question base model results
│   ├── results_final.csv         # Per-question all-config results
│   └── run_logs/                 # Timestamped experiment logs
│
├── finetuning/                   # Phase 3: Fine-tuning
│   ├── train.py                  # QLoRA training script (with checkpointing)
│   ├── upload_model.py           # HuggingFace Hub upload utility
│   └── requirements_finetune.txt # Training dependencies
│
├── rag_pipeline/                 # RAG infrastructure
│   ├── indexer.py                # FAISS index builder
│   ├── retriever.py              # Retrieval interface
│   └── faiss_index/              # Pre-built vector index
│
├── context/                      # Study documentation & analysis
│   ├── final_conclusion_notes.md # Core scientific findings
│   ├── dissertation_seed_report.md # Comprehensive study report
│   ├── project_chronological_log.md # Full project timeline
│   ├── prompts_documentation.md  # All prompts used in the study
│   └── ...                       # Additional analysis documents
│
├── clean_dataset.py              # Data cleaning script (Phase 2)
├── analyze_data.py               # Dataset analysis utility
└── requirements.txt              # Project dependencies
```

---

## Technologies

| Category | Technology |
|----------|-----------|
| **Base Model** | [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |
| **Fine-Tuning** | QLoRA via [PEFT](https://github.com/huggingface/peft) + [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) (4-bit NF4) |
| **RAG** | [FAISS](https://github.com/facebookresearch/faiss) (vector similarity search) |
| **LLM Judge** | [Claude 3.5 Sonnet](https://www.anthropic.com/claude) (evaluation scoring) |
| **Data Generation** | Claude API (synthetic QA pair generation from textbook) |
| **Training Platform** | [RunPod](https://www.runpod.io/) (NVIDIA A40 48GB GPU) |
| **Data Source** | [OpenStax College Physics 2e](https://openstax.org/details/books/college-physics-2e) (Chapters 1–6) |
| **Libraries** | Transformers, Datasets, PyTorch, Sentence-Transformers |

---

## How to Reproduce

### Prerequisites
- Python 3.10+
- Anthropic API key (for Claude-based scoring)
- GPU with ≥24GB VRAM (for inference and fine-tuning)

### 1. Setup

```bash
git clone https://github.com/diliprc96/rag-vs-finetuning-llm-comparison.git
cd rag-vs-finetuning-llm-comparison
pip install -r requirements.txt
```

### 2. Data Extraction (optional — dataset is included)

```bash
cd data_extraction
pip install -r requirements.txt
python data_crawler.py          # Scrape OpenStax textbook
python chunk_and_generate.py    # Generate QA pairs via Claude
cd ..
python clean_dataset.py         # Normalize output formats
```

### 3. Build RAG Index

```bash
cd rag_pipeline
python indexer.py               # Build FAISS index from textbook chunks
```

### 4. Run Evaluation

```bash
cd evaluation
pip install -r requirements.txt

# Base model evaluation
python run_eval.py --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2

# Base + RAG evaluation
python run_eval.py --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 --use_rag

# Fine-tuned model evaluation
python run_eval.py --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --use_finetuned --finetuned_model_path <path-to-qlora-adapter>
```

### 5. Fine-Tuning (optional — requires RunPod or equivalent GPU)

```bash
cd finetuning
pip install -r requirements_finetune.txt
python train.py
```

---

## Future Work

- **Synthetic data normalization** — Generate a fully clean dataset from scratch with strict format constraints
- **Hybrid RAG + Instruction Tuning** — Fine-tune the model to better *use* retrieved context rather than memorize facts
- **LoRA rank experiments** — Sweep rank/alpha to find the optimal capacity-stability trade-off
- **Larger evaluation set** — Expand beyond 50 questions for more statistically robust comparisons
- **Multi-domain generalization** — Test whether findings hold across other STEM subjects

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The OpenStax textbook content is used under the Creative Commons Attribution License 4.0. See [LICENSE_DATA.md](LICENSE_DATA.md) for details.
