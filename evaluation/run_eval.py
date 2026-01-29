import os
import json
import argparse
import pandas as pd
import torch
import logging
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from evaluation.scorers import grade_mcq, grade_numeric, grade_explanation
# Assumes rag_utils is available 
try:
    from evaluation.rag_utils import load_index, retrieve, format_docs
except ImportError:
    print("Warning: rag_utils not found or failed to import. RAG will not work.")

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = "mistral-7b-physics-finetuned"
EVAL_DATA_PATH = "evaluation/physics_questions_50.json"
LOG_DIR = "evaluation/run_logs"

def setup_output_file():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(LOG_DIR, f"eval_results_{timestamp}.csv")

def load_models(run_finetuned=False, adapter_id=ADAPTER_PATH):
    logger.info(f"Loading Base Model: {BASE_MODEL_ID}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    if run_finetuned:
        if not adapter_id:
            raise ValueError("Adapter ID/Path must be provided for finetuned mode.")
        logger.info(f"Loading LoRA Adapter from {adapter_id}")
        model = PeftModel.from_pretrained(model, adapter_id)
    
    return model, tokenizer

def generate_answer(model, tokenizer, question, context=None):
    if context:
        prompt = f"[INST] Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely. If MCQ, output only the option letter. If Numeric, output only the number. [/INST]"
    else:
        prompt = f"[INST] Question: {question}\n\nAnswer concisely. If MCQ, output only the option letter. If Numeric, output only the number. [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1 # Low temp for deterministic evaluation
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
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

    output_file = setup_output_file()
    logger.info(f"Results will be saved to {output_file}")

    if not os.path.exists(args.eval_file):
        logger.error(f"Error: Eval file {args.eval_file} not found.")
        return

    with open(args.eval_file, "r") as f:
        questions = json.load(f)

    # Load RAG if needed
    db = None
    if args.rag or args.mode == "all":
        try:
            db = load_index()
            logger.info("RAG Index loaded.")
        except Exception as e:
            logger.warning(f"RAG Load Error: {e}")

    results = []

    # Define configurations
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

    # --- Run Base Tasks First ---
    base_confs = [c for c in configs if not c[1]]
    if base_confs:
        model, tokenizer = load_models(run_finetuned=False)
        for name, _, use_rag in base_confs:
            logger.info(f"Running Configuration: {name}")
            for q in tqdm(questions):
                context = ""
                if use_rag and db:
                    docs = retrieve(db, q['question'])
                    context = format_docs(docs)
                
                ans = generate_answer(model, tokenizer, q['question'], context)
                
                # Grading
                score_mcq = 0.0
                score_num = 0.0
                score_exp = 0.0
                reasoning = ""
                
                if q['type'] == 'mcq':
                    score_mcq = grade_mcq(ans, q['answer'])
                elif q['type'] == 'numeric':
                    score_num = grade_numeric(ans, q['answer'])
                elif q['type'] == 'explanation':
                    res = grade_explanation(ans, q['answer'])
                    score_exp = res['score']
                    reasoning = res['reasoning']
                
                results.append({
                    "config": name,
                    "question_id": q.get('id'),
                    "type": q['type'],
                    "question": q['question'],
                    "predicted": ans,
                    "correct": q['answer'],
                    "score_mcq": score_mcq,
                    "score_numeric": score_num,
                    "score_explanation": score_exp,
                    "reasoning": reasoning
                })
                pd.DataFrame(results).to_csv(output_file, index=False)
        
        del model
        torch.cuda.empty_cache()

    # --- Run Finetuned Tasks ---
    ft_confs = [c for c in configs if c[1]]
    if ft_confs:
        try:
            logger.info("Loading Finetuned Model...")
            model, tokenizer = load_models(run_finetuned=True, adapter_id=args.adapter_id)
            for name, _, use_rag in ft_confs:
                logger.info(f"Running Configuration: {name}")
                for q in tqdm(questions):
                    context = ""
                    if use_rag and db:
                        docs = retrieve(db, q['question'])
                        context = format_docs(docs)
                    
                    ans = generate_answer(model, tokenizer, q['question'], context)
                    
                    # Grading
                    score_mcq = 0.0
                    score_num = 0.0
                    score_exp = 0.0
                    reasoning = ""
                    
                    if q['type'] == 'mcq':
                        score_mcq = grade_mcq(ans, q['answer'])
                    elif q['type'] == 'numeric':
                        score_num = grade_numeric(ans, q['answer'])
                    elif q['type'] == 'explanation':
                        res = grade_explanation(ans, q['answer'])
                        score_exp = res['score']
                        reasoning = res['reasoning']
                    
                    results.append({
                        "config": name,
                        "question_id": q.get('id'),
                        "type": q['type'],
                        "question": q['question'],
                        "predicted": ans,
                        "correct": q['answer'],
                        "score_mcq": score_mcq,
                        "score_numeric": score_num,
                        "score_explanation": score_exp,
                        "reasoning": reasoning
                    })
                    pd.DataFrame(results).to_csv(output_file, index=False)
        except Exception as e:
            logger.error(f"Finetuned run failed: {e}")

    # Final Save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    logger.info(f"Final results saved to {output_file}")
    print(df.groupby("config")[["score_mcq", "score_numeric", "score_explanation"]].mean())

if __name__ == "__main__":
    main()
