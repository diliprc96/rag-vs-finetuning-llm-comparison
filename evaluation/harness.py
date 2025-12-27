import json
import argparse
import os
from statistics import mean
import numpy as np
from scipy import stats
from score import score_objective, llm_judge_score
from rag import RAGIndex


# --- PLACEHOLDER model inference functions ---
# Replace these with real inference code (local model, API, or pipeline)

def run_base_model(question: str, context: str = None) -> str:
    # Minimal placeholder: echo question or include context
    if context:
        return f"Using context: {context[:200]} ... Answer: [placeholder answer for] {question}"
    return "[BASE MODEL ANSWER]"


def run_finetuned_model(question: str, context: str = None) -> str:
    if context:
        return f"[FINETUNED ANSWER using context] {question}"
    return "[FINETUNED ANSWER]"


def evaluate_run(setup_name: str, questions: list, retriever: RAGIndex = None, openai_client=None):
    mcq_scores = []
    expl_scores = []
    runtimes = []
    for q in questions:
        context = None
        if retriever:
            hits = retriever.retrieve(q['question'], k=5)
            context = "\n".join([h[2] for h in hits])
        if setup_name == 'Base':
            pred = run_base_model(q['question'], context)
        elif setup_name == 'Finetuned':
            pred = run_finetuned_model(q['question'], context)
        elif setup_name == 'Base+RAG':
            pred = run_base_model(q['question'], context)
        elif setup_name == 'Finetuned+RAG':
            pred = run_finetuned_model(q['question'], context)
        else:
            pred = ""
        if q['type'] == 'objective':
            s = score_objective(pred, q['gold'], q.get('gold_numeric'))
            mcq_scores.append(s)
        else:
            res = llm_judge_score(q['question'], pred, q.get('reference', q.get('gold', '')), openai_client=openai_client)
            expl_scores.append(res['score'])
    mcq_acc = mean(mcq_scores) if mcq_scores else 0.0
    expl_mean = mean(expl_scores) if expl_scores else 0.0
    total = 0.7 * mcq_acc + 0.3 * (expl_mean / 5.0)  # normalize expl to 0-1 before combining
    return {'setup': setup_name, 'mcq_acc': mcq_acc, 'expl_mean': expl_mean, 'total': total}


def load_questions(path: str):
    qs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            qs.append(json.loads(line))
    return qs


def main(args):
    questions = load_questions(args.questions)
    # Build or load RAG index if requested
    rag = None
    if args.use_rag:
        # for demo, build index from reference text snippets in questions
        texts = [q.get('reference', '') for q in questions]
        rag = RAGIndex()
        rag.build(texts)
    setups = ['Base', 'Finetuned']
    results = []
    for s in setups:
        results.append(evaluate_run(s, questions, retriever=None))
        if args.use_rag:
            results.append(evaluate_run(s + '+RAG', questions, retriever=rag))
    # Print table-like output
    print("Setup\tMCQ Acc\tExpl Mean\tTotal")
    for r in results:
        print(f"{r['setup']}\t{r['mcq_acc']:.2f}\t{r['expl_mean']:.2f}\t{r['total']:.2f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--questions', required=True)
    p.add_argument('--use-rag', action='store_true')
    args = p.parse_args()
    main(args)
