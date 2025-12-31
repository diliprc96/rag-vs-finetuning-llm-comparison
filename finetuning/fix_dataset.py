import json
import os

input_path = '/workspace/rag-vs-finetuning-llm-comparison/data_extraction/alpaca_physics_5k.jsonl'
output_path = '/workspace/rag-vs-finetuning-llm-comparison/data_extraction/alpaca_physics_5k_fixed.jsonl'

processed_count = 0
fixed_count = 0

with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
    for i, line in enumerate(infile):
        try:
            data = json.loads(line)
            modified = False
            if 'input' in data and not isinstance(data['input'], str):
                # Convert non-string input to string
                data['input'] = json.dumps(data['input'])
                modified = True
            
            # Also check output, just in case
            if 'output' in data and not isinstance(data['output'], str):
                data['output'] = json.dumps(data['output'])
                modified = True
            
            if modified:
                fixed_count += 1
                
            outfile.write(json.dumps(data) + '\n')
            processed_count += 1
        except json.JSONDecodeError:
            print(f"Skipping invalid json at line {i+1}")

print(f"Finished fixing dataset. Processed {processed_count} lines. Fixed {fixed_count} lines.")
