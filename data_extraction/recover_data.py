import json
import re
import os

EXISTING_FILE = "data_extraction/alpaca_physics_5k.jsonl"
RAW_FILE = "data_extraction/raw_claude_responses.jsonl"
RECOVERED_FILE = "data_extraction/recovered_data.jsonl"

def load_existing_instructions():
    instructions = set()
    if os.path.exists(EXISTING_FILE):
        with open(EXISTING_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "instruction" in data:
                            instructions.add(data["instruction"].strip())
                    except:
                        pass
    return instructions

def extract_json_objects(text):
    """
    Robustly extract JSON objects from a string, handling broken lists or extra text.
    Uses a brace-counting method to find valid JSON objects.
    """
    objects = []
    text = text.strip()
    
    # Attempt 1: Try parsing the whole thing as a valid JSON list or object
    try:
        data = json.loads(text, strict=False)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except:
        pass

    # Attempt 2: Clean markdown and try again
    clean_text = re.sub(r'```json\s*|\s*```', '', text)
    try:
        data = json.loads(clean_text, strict=False)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except:
        pass

    # Attempt 3: Brace counting extraction
    # This is useful if the list structure is broken e.g. `[ {...}, {...] `
    idx = 0
    n = len(text)
    while idx < n:
        # Find start of an object
        start = text.find('{', idx)
        if start == -1:
            break
        
        # Scan forward to find balancing brace
        balance = 0
        for i in range(start, n):
            char = text[i]
            if char == '{':
                balance += 1
            elif char == '}':
                balance -= 1
            
            if balance == 0:
                # Potential object text from start to i+1
                candidate = text[start:i+1]
                try:
                    obj = json.loads(candidate, strict=False)
                    if isinstance(obj, dict) and "instruction" in obj and "output" in obj:
                        objects.append(obj)
                        idx = i + 1 # Move past this object
                        break
                except:
                    # Not a valid JSON object, keep scanning from same start? 
                    # No, usually if it fails it's because of internal syntax. 
                    # But maybe we captured too much or too little. 
                    # For simplicity, we just continue scanning for the next brace
                    pass
        else:
            # Reached end without balance
            break
            
        # If we didn't break out of the for loop successfully (i.e. didn't find match),
        # we need to increment idx to avoid infinite loop
        if idx <= start:
            idx = start + 1

    return objects

def main():
    print(f"Loading existing instructions from {EXISTING_FILE}...")
    existing_set = load_existing_instructions()
    print(f"Found {len(existing_set)} existing instructions.")

    recovered_count = 0
    parse_failures = 0

    print(f"Scanning raw logs from {RAW_FILE}...")
    
    # Open valid file to append
    with open(RECOVERED_FILE, "a", encoding="utf-8") as f_out:
        if os.path.exists(RAW_FILE):
            with open(RAW_FILE, "r", encoding="utf-8") as f_in:
                for line_num, line in enumerate(f_in):
                    if not line.strip():
                        continue
                    
                    try:
                        log_entry = json.loads(line)
                        raw_response = log_entry.get("raw_response", "")
                        
                        if not raw_response:
                            continue

                        extracted_items = extract_json_objects(raw_response)
                        
                        if not extracted_items:
                            # If raw_response was not empty but we got nothing, log it
                            parse_failures += 1
                            # print(f"Failed to extract any JSON from line {line_num}")
                        
                        for item in extracted_items:
                            instr = item.get("instruction", "").strip()
                            if instr and instr not in existing_set:
                                # Found a new one!
                                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                                existing_set.add(instr)
                                recovered_count += 1
                                
                    except Exception as e:
                        print(f"Error processing log line {line_num}: {e}")

    print("-" * 40)
    print(f"Recovery Complete.")
    print(f"Recovered {recovered_count} new pairs.")
    print(f"Saved to {RECOVERED_FILE}")
    print(f"Parse failures (chunks yielding 0 items): {parse_failures}")

if __name__ == "__main__":
    main()
