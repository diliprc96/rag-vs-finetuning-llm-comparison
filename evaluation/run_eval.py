import os
import json
import argparse
import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()
from peft import PeftModel, PeftConfig
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag_pipeline.retriever import load_index, retrieve, format_docs
from evaluation.scorers import grade_mcq, grade_numeric, grade_explanation

import logging
import datetime

# Configuration
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = "results/mistral-7b-physics-finetune" 
EVAL_DATA_PATH = "evaluation/physics_questions_50.json" 

# Versioned Output
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"evaluation/results_table_{timestamp}.csv"
LOG_FILE = f"evaluation/eval_{timestamp}.log"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Redirect print to logger (optional, or just use logger)
# Ideally replace print with logger.info, but for quick fix:
def log_print(*args, **kwargs):
    msg = " ".join(map(str, args))
    logger.info(msg)
    print(*args, **kwargs)

def load_models(run_finetuned=False, adapter_id=ADAPTER_PATH):
    log_print(f"Loading Base Model: {BASE_MODEL_ID}")
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
        if not adapter_id:
            raise ValueError("Adapter ID/Path must be provided for finetuned mode.")
        log_print(f"Loading LoRA Adapter from {adapter_id}")
        # Note: If adapter_id is a remote Hub ID (private), ensure HF_TOKEN is set.
        model = PeftModel.from_pretrained(model, adapter_id)
    
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
    parser.add_argument("--adapter_id", default=ADAPTER_PATH, help="Path or HF ID of the adapter to load")
    args = parser.parse_args()

    if not os.path.exists(args.eval_file):
        log_print(f"Error: Eval file {args.eval_file} not found. Please provide the 50 questions.")
        return

    with open(args.eval_file, "r") as f:
        questions = json.load(f)

    # Load RAG if needed
    db = None
    if args.rag or args.mode == "all":
        try:
            db = load_index()
            log_print("RAG Index loaded.")
        except Exception as e:
            log_print(f"RAG Load Error: {e}")
            if args.rag: return

    # Load existing results if they exist to support resume
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            results = existing_df.to_dict('records')
            log_print(f"Loaded {len(results)} existing results.")
        except:
            log_print("Could not load existing results. Starting fresh.")
            results = []
    else:
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
            log_print(f"Running Configuration: {name}")
            for q in tqdm(questions):
                # Check if already done
                if any(r['question_id'] == q['id'] and r['config'] == name for r in results):
                    continue
                
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
                # Incremental Save
                pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

        del model
        torch.cuda.empty_cache()
 
    # 2. Run Finetuned Model Tasks
    ft_tasks = [c for c in configs if c[1]]
    if ft_tasks:
        torch.cuda.empty_cache()
        try:
            model, tokenizer = load_models(run_finetuned=True, adapter_id=args.adapter_id)
            for name, _, use_rag in ft_tasks:
                log_print(f"Running Configuration: {name}")
                for q in tqdm(questions):
                    # Check if already done
                    if any(r['question_id'] == q['id'] and r['config'] == name for r in results):
                        continue

                    context = ""
                    if use_rag and db:
                        docs = retrieve(db, q['question'], k=2)
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
                    # Incremental Save
                    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        except Exception as e:
            log_print(f"Skipping Finetuned runs: {e}")
            # traceback.log_print_exc()

    # Save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    log_print(f"Results saved to {OUTPUT_FILE}")
    log_print(df.groupby("config")[["score_mcq", "score_numeric", "score_explanation"]].mean())

if __name__ == "__main__":
    main()
