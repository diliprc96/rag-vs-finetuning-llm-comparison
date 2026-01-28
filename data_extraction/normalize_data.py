import json
import os

INPUT_FILE = "data_extraction/alpaca_physics_5k_cleaned.jsonl"
TEMP_FILE = "data_extraction/alpaca_physics_5k_cleaned_normalized.jsonl"

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
                # Dump it to string so it's consistent
                item["output"] = json.dumps(item["output"], ensure_ascii=False)
                fixed_count += 1
            
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        except json.JSONDecodeError:
            print(f"Skipping bad JSON at line {count}")

print(f"Processed {count} lines.")
print(f"Fixed {fixed_count} lines where output was not a string.")

# Overwrite original
os.replace(TEMP_FILE, INPUT_FILE)
print(f"Normalized data saved to {INPUT_FILE}.")
