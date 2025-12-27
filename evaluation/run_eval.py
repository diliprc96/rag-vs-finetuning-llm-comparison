import os
import json
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from rag_pipeline.retriever import load_index, retrieve, format_docs
from evaluation.scorers import grade_mcq, grade_numeric, grade_explanation

# Configuration
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = "results/mistral-7b-physics-finetune" # Path to local adapter after training
EVAL_DATA_PATH = "evaluation/physics_questions_50.json" # User provided
OUTPUT_FILE = "evaluation/results_table.csv"

def load_models(run_finetuned=False):
    print(f"Loading Base Model: {BASE_MODEL_ID}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    if run_finetuned:
        if not os.path.exists(ADAPTER_PATH):
            raise FileNotFoundError(f"Adapter not found at {ADAPTER_PATH}. Run training first.")
        print(f"Loading LoRA Adapter from {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    
    return model, tokenizer

def generate_answer(model, tokenizer, question, context=None):
    if context:
        prompt = f"[INST] Context:\n{context}\n\nQuestion: {question} [/INST]"
    else:
        prompt = f"[INST] Question: {question} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the output part (remove prompt)
    # Mistral output usually includes prompt? 
    # Yes, decode returns full text.
    # Simple split by [/INST]
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "finetuned", "all"], default="all")
    parser.add_argument("--rag", action="store_true", help="Enable RAG")
    parser.add_argument("--eval_file", default=EVAL_DATA_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.eval_file):
        print(f"Error: Eval file {args.eval_file} not found. Please provide the 50 questions.")
        return

    with open(args.eval_file, "r") as f:
        questions = json.load(f)

    # Load RAG if needed
    db = None
    if args.rag or args.mode == "all":
        try:
            db = load_index()
            print("RAG Index loaded.")
        except Exception as e:
            print(f"RAG Load Error: {e}")
            if args.rag: return

    results = []

    # Define configurations to run
    configs = []
    if args.mode == "all":
        configs = [
            ("Base", False, False),
            ("Base+RAG", False, True),
            ("Finetuned", True, False),
            ("Finetuned+RAG", True, True)
        ]
    else:
        is_ft = (args.mode == "finetuned")
        configs.append((args.mode + ("+RAG" if args.rag else ""), is_ft, args.rag))

    # Iterate configs
    # Note: Loading models takes time, so we might want to group by model type
    # But for simplicity, we reload or manage memory? 
    # Better to load Base, run Base tasks. Then load Adapter, run FT tasks.
    
    # 1. Run Base Model Tasks
    base_tasks = [c for c in configs if not c[1]]
    if base_tasks:
        model, tokenizer = load_models(run_finetuned=False)
        for name, _, use_rag in base_tasks:
            print(f"Running Configuration: {name}")
            for q in tqdm(questions):
                context = ""
                if use_rag and db:
                    docs = retrieve(db, q['question'])
                    context = format_docs(docs)
                
                ans = generate_answer(model, tokenizer, q['question'], context)
                
                # Grade
                score_mcq = 0.0
                score_num = 0.0
                score_exp = 0.0
                
                if q['type'] == 'mcq':
                    score_mcq = grade_mcq(ans, q['answer'])
                elif q['type'] == 'numeric':
                    score_num = grade_numeric(ans, q['answer'])
                elif q['type'] == 'explanation':
                    score_exp = grade_explanation(ans, q['answer']) # Uses Claude
                
                results.append({
                    "config": name,
                    "question_id": q.get('id'),
                    "type": q['type'],
                    "question": q['question'],
                    "predicted": ans,
                    "correct": q['answer'],
                    "score_mcq": score_mcq,
                    "score_numeric": score_num,
                    "score_explanation": score_exp
                })
        del model
        torch.cuda.empty_cache()

    # 2. Run Finetuned Model Tasks
    ft_tasks = [c for c in configs if c[1]]
    if ft_tasks:
        try:
            model, tokenizer = load_models(run_finetuned=True)
            for name, _, use_rag in ft_tasks:
                print(f"Running Configuration: {name}")
                for q in tqdm(questions):
                    context = ""
                    if use_rag and db:
                        docs = retrieve(db, q['question'])
                        context = format_docs(docs)
                    
                    ans = generate_answer(model, tokenizer, q['question'], context)
                     
                    score_mcq = 0.0
                    score_num = 0.0
                    score_exp = 0.0
                    
                    if q['type'] == 'mcq':
                        score_mcq = grade_mcq(ans, q['answer'])
                    elif q['type'] == 'numeric':
                        score_num = grade_numeric(ans, q['answer'])
                    elif q['type'] == 'explanation':
                        score_exp = grade_explanation(ans, q['answer'])
                    
                    results.append({
                        "config": name,
                        "question_id": q.get('id'),
                        "type": q['type'],
                        "question": q['question'],
                        "predicted": ans,
                        "correct": q['answer'],
                        "score_mcq": score_mcq,
                        "score_numeric": score_num,
                        "score_explanation": score_exp
                    })
        except Exception as e:
            print(f"Skipping Finetuned runs: {e}")

    # Save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")
    print(df.groupby("config")[["score_mcq", "score_numeric", "score_explanation"]].mean())

if __name__ == "__main__":
    main()
