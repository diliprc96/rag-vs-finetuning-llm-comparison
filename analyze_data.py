import json
import re

DATA_PATH = "data_extraction/alpaca_physics_5k_cleaned.jsonl"

def analyze():
    total = 0
    json_count = 0
    text_count = 0
    errors = 0
    
    with open(DATA_PATH, 'r') as f:
        for line in f:
            total += 1
            try:
                item = json.loads(line)
                output = item.get('output', '').strip()
                
                # Check if output looks like JSON
                is_json = False
                if output.startswith('{') and output.endswith('}'):
                    try:
                        json.loads(output)
                        is_json = True
                    except:
                        pass
                
                if is_json:
                    json_count += 1
                else:
                    text_count += 1
            except:
                errors += 1
                
    print(f"Total Lines: {total}")
    print(f"JSON Output Format: {json_count} ({json_count/total*100:.1f}%)")
    print(f"Plain Text Format: {text_count} ({text_count/total*100:.1f}%)")
    print(f"Parse Errors: {errors}")

if __name__ == "__main__":
    analyze()
