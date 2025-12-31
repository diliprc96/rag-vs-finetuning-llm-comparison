import json
import re
import os

RAW_FILE = "data_extraction/raw_claude_responses.jsonl"

def extract_json_objects_robust(text):
    """
    Robustly extract JSON objects from a string, identical to recover_data.py
    """
    objects = []
    text = text.strip()
    
    # Attempt 1: Try parsing the whole thing
    try:
        data = json.loads(text, strict=False)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except:
        pass

    # Attempt 2: Clean markdown
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
    idx = 0
    n = len(text)
    while idx < n:
        start = text.find('{', idx)
        if start == -1:
            break
        
        balance = 0
        for i in range(start, n):
            char = text[i]
            if char == '{':
                balance += 1
            elif char == '}':
                balance -= 1
            
            if balance == 0:
                candidate = text[start:i+1]
                try:
                    obj = json.loads(candidate, strict=False)
                    if isinstance(obj, dict) and "instruction" in obj and "output" in obj:
                        objects.append(obj)
                        idx = i + 1 
                        break
                except:
                    pass
        else:
            break
            
        if idx <= start:
            idx = start + 1

    return objects

def main():
    if not os.path.exists(RAW_FILE):
        print(f"File not found: {RAW_FILE}")
        return

    total_chunks = 0
    total_pairs_raw = 0
    chunks_with_0 = 0
    chunks_with_less_than_10 = 0
    zero_yield_reasons = []

    print(f"Analyzing {RAW_FILE}...")
    
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            total_chunks += 1
            
            try:
                log_entry = json.loads(line)
                raw_response = log_entry.get("raw_response", "")
                chunk_snippet = log_entry.get("chunk_snippet", "N/A")
                
                start_time = log_entry.get("timestamp", 0)

                items = extract_json_objects_robust(raw_response)
                count = len(items)
                total_pairs_raw += count
                
                if count == 0:
                    chunks_with_0 += 1
                    # Store a snippet of the raw response to see why it failed
                    zero_yield_reasons.append({
                        "line": line_num + 1,
                        "snippet": chunk_snippet,
                        "response_preview": raw_response[:200].replace("\n", " ")
                    })
                elif count < 10:
                    chunks_with_less_than_10 += 1

            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")

    with open("investigation_report.txt", "w", encoding="utf-8") as rep:
        rep.write("="*40 + "\n")
        rep.write(f"STATISTICS REPORT\n")
        rep.write("="*40 + "\n")
        rep.write(f"Total Chunks in Log: {total_chunks}\n")
        rep.write(f"Total Pairs Found (Raw Yield): {total_pairs_raw}\n")
        rep.write(f"Average Pairs per Chunk: {total_pairs_raw / total_chunks if total_chunks else 0:.2f}\n")
        rep.write(f"Chunks with 0 pairs: {chunks_with_0}\n")
        rep.write(f"Chunks with < 10 pairs: {chunks_with_less_than_10}\n")
        rep.write(f"Theoretical Max (if 10/chunk): {total_chunks * 10}\n")
        rep.write(f"Actual Yield Rate: {(total_pairs_raw / (total_chunks * 10) * 100) if total_chunks else 0:.1f}%\n")
        rep.write("-" * 40 + "\n")
        
        if zero_yield_reasons:
            rep.write("\nSAMPLES OF 0-YIELD CHUNKS:\n")
            for fail in zero_yield_reasons[:5]:
                rep.write(f"Line {fail['line']}:\n")
                rep.write(f"  Input Snippet: {fail['snippet']}\n")
                rep.write(f"  Response Preview: {fail['response_preview']}\n")
                rep.write("-" * 20 + "\n")

    print("Investigation complete. Report saved to investigation_report.txt")

if __name__ == "__main__":
    main()
