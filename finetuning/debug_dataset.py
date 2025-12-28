
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_FILE = "data_extraction/alpaca_physics_5k.jsonl"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DATASET_FILE)
    args = parser.parse_args()

    print(f"Loading tokenizer {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Loading dataset from {args.dataset}...")
    try:
        dataset = load_dataset('json', data_files=args.dataset, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have generated the synthetic data first!")
        return

    print(f"Dataset size: {len(dataset)}")

    # Formatting function (Same as train.py)
    def format_prompt(example):
        instruction = example['instruction']
        input_text = example.get('input', "")
        output = example['output']
        
        if input_text:
            prompt = f"[INST] {instruction}\n\n{input_text} [/INST] {output}"
        else:
            prompt = f"[INST] {instruction} [/INST] {output}"
        return prompt

    print("\n--- Checking Samples ---")
    lengths = []
    
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        formatted = format_prompt(sample)
        tokens = tokenizer.encode(formatted)
        lengths.append(len(tokens))
        
        print(f"\n[Sample {i}]")
        print(f"Instruction: {sample['instruction']}")
        print(f"Input: {sample.get('input', '')}")
        print(f"Output (Preview): {sample['output'][:100]}...")
        print("Formatted:\n" + "-"*20)
        print(formatted)
        print("-" * 20)
        print(f"Token Count: {len(tokens)}")

    # Check whole dataset stats
    print("\n--- Analyzing Token Lengths (First 100 or all) ---")
    all_lengths = []
    check_limit = min(500, len(dataset))
    for i in range(check_limit):
        txt = format_prompt(dataset[i])
        all_lengths.append(len(tokenizer.encode(txt)))
    
    print(f"Max Length: {max(all_lengths)}")
    print(f"Mean Length: {np.mean(all_lengths):.2f}")
    if max(all_lengths) > 2048:
        print("WARNING: Some samples exceed 2048 tokens and will be truncated during training.")
    else:
        print("SUCCESS: All samples fit within 2048 context window.")

if __name__ == "__main__":
    main()
