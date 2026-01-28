import json
from collections import Counter

file_path = '/workspace/rag-vs-finetuning-llm-comparison/data_extraction/alpaca_physics_5k.jsonl'

types = Counter()
keys = Counter()
sample_jsons = []

with open(file_path, 'r') as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            output = data.get('output')
            
            if isinstance(output, dict):
                types['dict'] += 1
                keys.update(output.keys())
                if len(sample_jsons) < 3:
                     sample_jsons.append(f"Line {i}: Dict keys: {list(output.keys())}")
            elif isinstance(output, str):
                # Check if it looks like JSON
                output = output.strip()
                if output.startswith('{') and output.endswith('}'):
                    try:
                        parsed = json.loads(output)
                        if isinstance(parsed, dict):
                            types['stringified_dict'] += 1
                            keys.update(parsed.keys())
                            if len(sample_jsons) < 6 and len(sample_jsons) >= 3:
                                sample_jsons.append(f"Line {i}: Stringified Dict keys: {list(parsed.keys())}")
                        else:
                            types['string_other'] += 1
                    except:
                        types['string_plain'] += 1
                else:
                    types['string_plain'] += 1
            else:
                types[type(output).__name__] += 1
                
        except json.JSONDecodeError:
            print(f"Line {i}: JSON decode error")

print("Output Types Distribution:")
print(types)
print("\nKeys found in JSON/Dict outputs:")
print(keys.most_common(20))
print("\nSamples:")
for s in sample_jsons:
    print(s)
