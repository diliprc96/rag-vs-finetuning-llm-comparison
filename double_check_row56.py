
import json
import os

files = {
    "Original": "data_extraction/alpaca_physics_5k.jsonl",
    "Cleaned": "data_extraction/alpaca_physics_5k_cleaned.jsonl"
}

print(f"{'Metric':<25} | {'Original':<15} | {'Cleaned':<15}")
print("-" * 60)

for name, path in files.items():
    if not os.path.exists(path):
        print(f"{name:<25} | {'MISSING':<15}")
        continue
    
    with open(path, 'r') as f:
        line56 = ""
        for i, line in enumerate(f):
            if i == 55: # 0-indexed, so row 56 is index 55
                line56 = line
                break
        
    try:
        data = json.loads(line56)
        out_type = type(data.get('output')).__name__
        print(f"{name + ' Row 56 Type':<25} | {out_type:<15}")
        print(f"{name + ' Row 56 Content':<25} | {str(data.get('output'))[:50]}...")
    except Exception as e:
        print(f"{name:<25} | Error: {e}")
