import json
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Configuration
INPUT_FILE = "data_extraction/alpaca_physics_5k_no_latex.jsonl"
OUTPUT_FILE = "data_extraction/alpaca_physics_5k.jsonl"
API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_NAME = "claude-3-haiku-20240307"

SYSTEM_PROMPT = "You are a helpful assistant. Output ONLY valid JSON."

USER_PROMPT_TEMPLATE = """
You are acting as a data refiner. Your task is to take the provided Physics Instruction/Input/Output triplet and apply LaTeX formatting ONLY to mathematical formulas, variables, units, and numbers.

Rules:
1. ONLY apply LaTeX to math (e.g., "F = ma" -> "$F=ma$", "10 kg" -> "$10$ kg").
2. DO NOT apply LaTeX to regular text, words, or descriptions. (e.g., DO NOT write $\\text{{energy}}$, $energy$, $\\mathcal{{E}}nergy$ or $trains$. Keep them as plain text).\n3. Do not LaTeX-ify single words like "Force", "Energy", "Mass" unless they are single-letter variables (e.g. $F$, $E$, $m$).
3. DO NOT change the JSON structure. `output` should be a STRING, not a nested JSON object unless the original was one.
4. If the generic `input` field was empty, keep it empty.

Original Data:
Instruction: {instruction}
Input: {input_val}
Output: {output_val}

Return the refined JSON object:
{{
  "instruction": "...",
  "input": "...",
  "output": "..."
}}
"""

def refine_record(client, record):
    max_retries = 5
    base_delay = 5  # Start with 5s delay

    instruction = record.get("instruction", "")
    output_val = record.get("output", "")
    input_val = record.get("input", "")

    # Optimization: If no numbers or math symbols, maybe skip? 
    # But safer to just run all or do a quick regex check check if strict. 
    # For now, we process all.

    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=MODEL_NAME,
                max_tokens=1000,
                temperature=0.0, # Deterministic for formatting
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                        instruction=instruction, 
                        output_val=output_val,
                        input_val=input_val
                    )}
                ]
            )
            response_text = message.content[0].text.strip()
            
            # Basic cleanup if it wraps in markdown code block
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()

            try:
                data = json.loads(response_text)
                return data
            except json.JSONDecodeError:
                # Fallback: keep original if parsing fails
                # print(f"WARN: JSON parse error. Keeping original.")
                return record
                
        except Exception as e:
            # Rate limit handling
            if "rate_limit_error" in str(e) or "429" in str(e):
                delay = base_delay * (2 ** attempt)
                # print(f"Rate limit. Retrying in {delay}s...")
                time.sleep(delay)
            elif "overloaded" in str(e):
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                print(f"Error: {e}")
                return record # Return original on error
    
    return record # Return original if retries exhausted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent threads")
    parser.add_argument("--limit", type=int, default=0, help="Test limit")
    args = parser.parse_args()

    if not API_KEY:
        print("Error: ANTHROPIC_API_KEY not set.")
        return

    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"Total records: {len(lines)}")
    
    if args.limit > 0:
        lines = lines[:args.limit]
        print(f"Limiting to first {args.limit} records.")

    records = [json.loads(line) for line in lines]
    client = Anthropic(api_key=API_KEY)
    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_record = {executor.submit(refine_record, client, r): r for r in records}
        
        for future in tqdm(as_completed(future_to_record), total=len(records)):
            res = future.result()
            results.append(res)
            
    print(f"Processed {len(results)} records.")
    
    # Sort or keep order? Threading messes up order. 
    # We might want to try to preserve order if possible, but datasets are usually shuffled anyway.
    # Results list is appended in completion order.
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
