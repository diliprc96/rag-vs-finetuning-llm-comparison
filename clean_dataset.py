import json
import os

input_file = '/workspace/rag-vs-finetuning-llm-comparison/data_extraction/alpaca_physics_5k.jsonl'
output_file = '/workspace/rag-vs-finetuning-llm-comparison/data_extraction/alpaca_physics_5k_cleaned.jsonl'

# Keys that we want to "unwrap" if they are the only key in the JSON object
KEYS_TO_UNWRAP = {
    'explanation', 'definition', 'answer', 'solution', 'result', 
    'formula', 'steps', 'equation', 'unit', 'relationship', 
    'direction', 'diagram', 'units', 'significance', 'velocity',
    'a', 'mass', 'description', 'acceleration'
}

def clean_output(output_val):
    """
    Normalizes the output value to a plain string.
    - If it's a dict (or stringified dict) with a single key in KEYS_TO_UNWRAP, return the value.
    - If it's a dict/list, json.dump it to a string.
    - Always returns a string.
    """
    data = None
    
    # helper to check if we should unwrap a dict
    def try_unwrap(d):
        if isinstance(d, dict):
            # Case 1: Single key - unwrap it regardless of the key name
            if len(d) == 1:
                k = list(d.keys())[0]
                # Previously we checked KEYS_TO_UNWRAP, now we unwrap everything 
                # to ensure we get plain strings for all single-key JSONs.
                val = d[k]
                if isinstance(val, (dict, list)):
                    return json.dumps(val) 
                return str(val)
            
            # Case 2: Check for "explanation" key specifically, often combined with "answer" or others
            # Some entries might be {"answer": "...", "explanation": "..."}. 
            # For now, per plan, let's stick to single-key unwrapping to be safe, 
            # or if it's mixed, we leave it as JSON to preserve structure.
        return None

    # parsing
    if isinstance(output_val, dict):
        data = output_val
    elif isinstance(output_val, str):
        output_val = output_val.strip()
        if output_val.startswith('{') and output_val.endswith('}'):
            try:
                data = json.loads(output_val)
            except json.JSONDecodeError:
                pass # it's just a string that looks like json but isn't valid
    
    # processing
    if data is not None:
        unwrapped = try_unwrap(data)
        if unwrapped is not None:
            return unwrapped
        # If we couldn't unwrap, but it was a valid dict/list, ensure it is a string
        if not isinstance(output_val, str):
             return json.dumps(output_val)
        return output_val

    # Fallback: ensure it's a string
    if not isinstance(output_val, str):
        return str(output_val)
    
    return output_val

def main():
    print(f"Cleaning {input_file}...")
    cleaned_count = 0
    total_count = 0
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            total_count += 1
            try:
                row = json.loads(line)
                original_output = row.get('output', "")
                cleaned_output = clean_output(original_output)
                row['output'] = cleaned_output
                
                # Also clean input and instruction just in case
                original_input = row.get('input', "")
                row['input'] = clean_output(original_input)
                
                original_instruction = row.get('instruction', "")
                row['instruction'] = clean_output(original_instruction)

                fout.write(json.dumps(row) + '\n')
                cleaned_count += 1
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line {total_count}")
    
    print(f"Finished. Processed {total_count} lines. Output saved to {output_file}")

if __name__ == '__main__':
    main()
