import json
import os
import re

RAW_FILE = "data_extraction/raw_claude_responses.jsonl"
FAILURES_DIR = "data_extraction/failures"

def extract_json_objects_robust(text):
    # Same logic as investigate_data.py to see if it parses
    objects = []
    text = text.strip()
    try:
        data = json.loads(text, strict=False)
        if isinstance(data, list): return data
        if isinstance(data, dict): return [data]
    except: pass
    
    clean_text = re.sub(r'```json\s*|\s*```', '', text)
    try:
        data = json.loads(clean_text, strict=False)
        if isinstance(data, list): return data
        if isinstance(data, dict): return [data]
    except: pass
    
    # Check brace counting
    idx = 0; n = len(text)
    while idx < n:
        start = text.find('{', idx)
        if start == -1: break
        balance = 0
        for i in range(start, n):
            char = text[i]
            if char == '{': balance += 1
            elif char == '}': balance -= 1
            if balance == 0:
                candidate = text[start:i+1]
                try:
                    obj = json.loads(candidate, strict=False)
                    if isinstance(obj, dict) and "instruction" in obj and "output" in obj:
                        objects.append(obj)
                        idx = i+1; break
                except: pass
        else: break
        if idx <= start: idx = start + 1
    return objects

def main():
    if not os.path.exists(FAILURES_DIR):
        os.makedirs(FAILURES_DIR)
        print(f"Created directory: {FAILURES_DIR}")
    
    chunks_exported = 0
    
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if not line.strip(): continue
            try:
                log_entry = json.loads(line)
                raw_response = log_entry.get("raw_response", "")
                
                # Check if it fails parsing completely (yields 0 results)
                items = extract_json_objects_robust(raw_response)
                
                if len(items) == 0 and len(raw_response) > 5: # Skip empty/tiny responses
                    filename = os.path.join(FAILURES_DIR, f"chunk_{line_num+1}.json")
                    
                    # Try to neaten it up if it's almost valid JSON but failed, 
                    # otherwise just write the raw string
                    # Actually, for manual fixing, raw string is best, 
                    # but if we can pretty print it, it's easier.
                    
                    # Let's write the RAW string content. 
                    # Use .txt extension if it's not valid JSON yet, but user wants to make it JSON?
                    # Let's use .json so syntax highlighting helps them.
                    
                    with open(filename, "w", encoding="utf-8") as out:
                        out.write(raw_response)
                        
                    chunks_exported += 1
            except Exception as e:
                print(f"Error on line {line_num}: {e}")

    print(f"Exported {chunks_exported} failed chunks to '{FAILURES_DIR}'.")
    print("Please edit each file to be a valid JSON list of objects: [{\"instruction\": \"...\", ...}]")

if __name__ == "__main__":
    main()
