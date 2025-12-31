import json
import os

INPUT_FILE = "data_extraction/alpaca_physics_5k.jsonl"
TEMP_FILE = "data_extraction/alpaca_physics_5k_normalized.jsonl"

print(f"Normalizing {INPUT_FILE}...")
count = 0
fixed_count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(TEMP_FILE, "w", encoding="utf-8") as fout:
    for line in fin:
        try:
            item = json.loads(line)
            count += 1
            
            # Check if 'output' is not a string
            if not isinstance(item["output"], str):
                # If it's a list or dict, dump it to string
                if isinstance(item["output"], (list, dict)):
                    # For clarity, if it's a dict with one key like "explanation" or "definition", maybe extract it?
                    # But simpler is to just stringify the whole thing so we don't lose data.
                    # Actually, looking at the examples: 
                    # Line 1: "output": {"examples": [...], "explanation": "..."}
                    # We should probably formatting it as a string: "Examples: ...\nExplanation: ..."
                    # But json.dumps is safer for now.
                    item["output"] = json.dumps(item["output"], ensure_ascii=False)
                    fixed_count += 1
                else:
                    item["output"] = str(item["output"])
                    fixed_count += 1
            
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        except json.JSONDecodeError:
            print(f"Skipping bad JSON at line {count}")

print(f"Processed {count} lines.")
print(f"Fixed {fixed_count} lines where output was not a string.")

# Replace original
# os.replace(TEMP_FILE, INPUT_FILE)
print(f"Normalized data saved to {TEMP_FILE}. Please verify and then replace.")
