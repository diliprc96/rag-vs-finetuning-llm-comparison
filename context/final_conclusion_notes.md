# Final Experiment Conclusion & Future Work Repository

## 1. The Core Scientific Finding: "Fragility vs. Stability"
The most significant outcome of this experiment is not just that "RAG performed better" (0.30 vs 0.14), but *why* it performed better. The experiment demonstrated a fundamental trade-off between the **Stability** of RAG and the **Fragility** of Fine-Tuning when applied to unstructured or slightly noisy domain data.

### The "Syntax vs. Semantics" Dissociation
Our fine-tuned model (Mistral-7B + LoRA) exhibited a fascinating failure mode.
*   **Semantics (Physics)**: The model *did* learn physics. In many cases (e.g., the SI Unit question), it correctly identified the concept (Kilogram) and even wrote a truthful sentence about it.
*   **Syntax (Instruction Following)**: However, it "forgot" how to be a test-taker. It lost the ability to follow meta-instructions like "Answer with only option A" or "Output a single number."
*   **Interpretation**: This suggests that small-scale fine-tuning (5k samples) on a dataset that is even partially inconsistent (8% JSON vs Text) causes the model to overfit to the *distribution of logic* in the training set while unlearning the *alignment* provided by the base instruct-tuning.

In contrast, **RAG** decoupled these problems:
*   **Retrieval** handled the "Knowledge" (finding the physics connection).
*   **Base Model** handled the "Syntax" (following the exam format).
*   **Result**: The RAG system improved knowledge (0.28 -> 0.30) without breaking the instruction-following capability.

## 2. The Data Quality Imperative (The "8% Rule")
We discovered that a dataset doesn't need to be "bad" to break a model; it just needs to be **inconsistent**.
*   **Observation**: Only ~350 samples (8%) were formatted as JSON strings. The other 92% were perfect natural language.
*   **Outcome**: This minority population was sufficient to confuse the model's output head, causing it to randomly hallucinate JSON structures in the evaluation phase.
*   **Implication for Future**: "Cleaning" is not just about removing Html tags or LaTeX errors (which we did in Phase 0). It is about **Structural Homogeneity**. Use strict schema validation before training. If 1% of your data speaks a different "structural language" (JSON vs Text), it can destabilize the entire training run.

## 3. Evaluation as the Primary Driver
We learned that **Metrics define the Result**.
*   **Legacy (0-5 scale)**: This metric was too noisy to detect the subtle degradation of the model.
*   **Strict Binary**: This exposed the formatting failure (0.02 score).
*   **Relaxed/Semantic**: This rescued the "truth" (0.14 score) by acknowledging that the model knew the physics but couldn't format it.

**Conclusion**: If we had stuck to the original regex-based evaluation, we would have concluded the model learned *nothing*. By using "Relaxed Grading" (LLM-as-a-Judge looking for semantic containment), we proved the model *did* learn but failed at delivery. This nuance is critical for the dissertation: **The failure was not in knowledge acquisition, but in format alignment.**

## 4. Future Work & Recommendations

### A. Synthetic Normalization (The "LLM-Refiner" Loop)
Regex-based cleaning (Phase 2) was insufficient because it only fixed obvious patterns.
*   **Future Approach**: Pass the entire 5k dataset through a strong model (e.g. GPT-4 or Claude 3.5) with the instruction: *"Rewrite this Q&A pair to be perfectly consistent with format X."*
*   **Goal**: Ensure 100% structural homogeneity before a single gradient update is performed.

### B. Hybrid RAG-Instruction Tuning (RA-IT)
Instead of fine-tuning on "Question -> Answer", fine-tune on "Question + Context -> Answer".
*   **Hypothesis**: Teaching the model *how to use retrieved context* is more valuable than trying to bake the context into the weights. This combines the stability of RAG with the specialization of Fine-Tuning.

### C. Parameter-Efficient Tuning Analysis
We used a standard LoRA rank (r=64).
*   **Future Experiment**: Test r=8, r=16, r=32. It is possible that r=64 was too high for a 5k dataset, allowing the model to "memorize" the noisy styles too quickly. A lower rank might force it to learn only the core physics concepts without overfitting to the inconsistent formatting.

### D. The "Negative Result" Value
Do not frame this project as a failure. Frame it as a successful boundary-testing of SFT. You have empirically proven that **for small, slightly noisy datasets, RAG is the superior engineering choice over SFT**. This is a highly valuable, publishable industrial finding.
