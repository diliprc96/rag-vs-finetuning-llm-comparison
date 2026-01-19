import json
import os
import random
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()
# Configuration
INPUT_FILE = "data_extraction/openstax_physics_vol1_ch1_6.json"
OUTPUT_FILE = "data_extraction/alpaca_physics_5k.jsonl"
API_KEY = os.getenv("ANTHROPIC_API_KEY") # Ensure this is set
MODEL_NAME = "claude-3-haiku-20240307" # Cost-effective, fast
NUM_PAIRS_TARGET = 5000

# Templates
# Templates
SYSTEM_PROMPT = "You are a physics expert. Output ONLY valid JSON."

USER_PROMPT_TEMPLATE = """
Here is a section from a Physics textbook:
{text}

Based on this text, generate {num_pairs} diverse instruction-response pairs.
Mix Explanation, Problem-Solving, and Concept Q&A.

IMPORTANT: Use LaTeX formatting for ALL math formulas, equations, and variables. Enclose them in dollar signs (e.g., $F=ma$, $v$, $10^{{2}}$).

Format:
[
  {{"instruction": "Question?", "input": "", "output": "Answer..."}},
  ...
]

Return ONLY the JSON list. No other text.
"""

RAW_OUTPUT_FILE = "data_extraction/raw_claude_responses.jsonl"

def generate_batch(client, text, num_pairs=3):
    max_retries = 5
    base_delay = 20
    
    for attempt in range(max_retries):
        try:
            # Pre-fill the assistant response with "[" to force JSON mode
            message = client.messages.create(
                model=MODEL_NAME,
                max_tokens=2000, 
                temperature=0.5,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text, num_pairs=num_pairs)},
                    {"role": "assistant", "content": "["}
                ]
            )
            response_text = message.content[0].text.strip()
            
            # Save raw response immediately!
            try:
                with open(RAW_OUTPUT_FILE, "a", encoding="utf-8") as f:
                    log_entry = {
                        "timestamp": time.time(),
                        "chunk_snippet": text[:100],
                        "raw_response": response_text
                    }
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"WARN: Failed to save raw log: {e}")

            if response_text.startswith("["):
                clean_text = response_text
            else:
                clean_text = "[" + response_text
            
            try:
                data = json.loads(clean_text, strict=False)
                return data
            except json.JSONDecodeError as e:
                print(f"WARN: JSON Parse Error {e} in chunk. Raw response saved to {RAW_OUTPUT_FILE}.")
                return []
                
        except Exception as e:
            # Only retry on network/rate limit, NOT on code errors
            if "rate_limit_error" in str(e) or "429" in str(e):
                delay = base_delay * (2 ** attempt)
                print(f"Rate limit hit. Retrying in {delay}s...")
                time.sleep(delay)
            elif "overloaded" in str(e):
                delay = base_delay * (2 ** attempt)
                print(f"API Overloaded. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"Error generating batch: {e}")
                return []
    
    print("Max retries exceeded for batch.")
    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Test run with limited number of chunks")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent threads")
    args = parser.parse_args()

    if not API_KEY:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        return

    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    all_chunks = []
    for section in raw_data:
        chunks = text_splitter.split_text(section["content"])
        for c in chunks:
            all_chunks.append(c)

    print(f"Total chunks available: {len(all_chunks)}")
    
    pairs_per_chunk = max(3, int(NUM_PAIRS_TARGET / len(all_chunks)) + 1)
    
    if args.limit > 0:
        all_chunks = all_chunks[:args.limit]
        print(f"Limiting to first {args.limit} chunks for testing.")

    client = Anthropic(api_key=API_KEY)
    results = []

    # Sequential for debugging if limit is set, else threaded
    if args.limit > 0:
        for chunk in all_chunks:
            data = generate_batch(client, chunk, pairs_per_chunk)
            if data:
                results.extend(data)
    else:
        # We also add a small delay between submissions to smooth out spikes if workers > 1
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_chunk = {executor.submit(generate_batch, client, chunk, pairs_per_chunk): chunk for chunk in all_chunks}
            
            for future in tqdm(as_completed(future_to_chunk), total=len(all_chunks)):
                data = future.result()
                if data:
                    results.extend(data)
                # Small sleep to be nice to the API if running single threaded loop effectively
                time.sleep(1.0)

    print(f"Generated {len(results)} pairs.")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
