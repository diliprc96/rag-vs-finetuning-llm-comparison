
import json
import os
from collections import Counter

files = {
    "Original": "data_extraction/alpaca_physics_5k.jsonl",
    "Cleaned": "data_extraction/alpaca_physics_5k_cleaned.jsonl"
}

print(f"{'Metric':<25} | {'Original':<15} | {'Cleaned':<15}")
print("-" * 60)

stats = {}

for name, path in files.items():
    if not os.path.exists(path):
        stats[name] = {"error": "File not found"}
        continue

    rows = []
    with open(path, 'r') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except:
                pass
    
    total = len(rows)
    output_types = Counter([type(r.get('output')).__name__ for r in rows])
    input_types = Counter([type(r.get('input')).__name__ for r in rows])
    
    # Check for JSON-like strings in output
    json_string_outputs = 0
    for r in rows:
        out = r.get('output', '')
        if isinstance(out, str) and out.strip().startswith('{') and out.strip().endswith('}'):
             try:
                 json.loads(out)
                 json_string_outputs += 1
             except:
                 pass

    unique_instructions = len(set([r.get('instruction', '') for r in rows]))
    avg_len = sum([len(str(r.get('output', ''))) for r in rows]) / total if total > 0 else 0
    
    stats[name] = {
        "Total Rows": total,
        "Output Types": dict(output_types),
        "Input Types": dict(input_types),
        "JSON-like Strings": json_string_outputs,
        "Unique Instructions": unique_instructions,
        "Avg Output Len": f"{avg_len:.1f}"
    }

metrics = ["Total Rows", "JSON-like Strings", "Unique Instructions", "Avg Output Len"]
for m in metrics:
    v1 = stats["Original"].get(m, "N/A")
    v2 = stats["Cleaned"].get(m, "N/A")
    print(f"{m:<25} | {str(v1):<15} | {str(v2):<15}")

print(f"{'Output Types':<25} | {str(stats['Original']['Output Types']):<15} | {str(stats['Cleaned']['Output Types']):<15}")
