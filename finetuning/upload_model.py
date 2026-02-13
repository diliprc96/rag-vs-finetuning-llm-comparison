import os
import argparse
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained adapter (local directory)")
    parser.add_argument("--repo_name", type=str, required=True, help="Hugging Face repo name (e.g., username/repo)")
    parser.add_argument("--commit_message", type=str, default="Upload trained adapter", help="Commit message")
    args = parser.parse_args()

    print(f"Loading adapter from {args.model_path}...")
    # We don't need to load the full model to push the adapter, 
    # but loading it as a PeftModel ensures it's valid.
    # Actually, the easiest way to push just the adapter files is to use the API or load PeftModel and push.
    
    # Check if files exist
    if not os.path.exists(os.path.join(args.model_path, "adapter_model.safetensors")) and \
       not os.path.exists(os.path.join(args.model_path, "adapter_model.bin")):
        print("Error: No adapter model file found in directory.")
        return

    print(f"Pushing to {args.repo_name}...")
    
    # Method 1: Use PeftModel.from_pretrained -> push_to_hub 
    # This requires the base model to be downloaded which is slow/expensive if not cached.
    # 
    # Method 2: Use huggingface_hub API to upload directory directly. Faster.
    from huggingface_hub import HfApi
    api = HfApi()
    
    api.upload_folder(
        folder_path=args.model_path,
        repo_id=args.repo_name,
        commit_message=args.commit_message,
        ignore_patterns=["checkpoint-*"] # Don't upload checkpoints
    )
    
    print("Upload complete!")

if __name__ == "__main__":
    main()
