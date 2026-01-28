from datasets import load_dataset
import os

cleaned_file = '/workspace/rag-vs-finetuning-llm-comparison/data_extraction/alpaca_physics_5k_cleaned.jsonl'

def verify():
    if not os.path.exists(cleaned_file):
        print(f"Error: {cleaned_file} does not exist.")
        return

    print(f"Attempting to load dataset from {cleaned_file}...")
    try:
        # Replicates the loading call from train.py
        dataset = load_dataset('json', data_files=cleaned_file, split='train')
        print(f"SUCCESS: Dataset loaded. Count: {len(dataset)} rows.")
        
        # Check a few rows to ensure types
        print("Sampling 3 rows:")
        for i in range(3):
            print(f"Row {i} output type: {type(dataset[i]['output'])}")
            print(f"Row {i} output content: {dataset[i]['output'][:100]}...")

    except Exception as e:
        print(f"FAILED: Loading dataset failed with error: {e}")

if __name__ == "__main__":
    verify()
