
import os
import torch
import argparse
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from trl import SFTTrainer

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
NEW_MODEL_NAME = "mistral-7b-physics-finetune"
DATASET_FILE = "data_extraction/alpaca_physics_5k.jsonl" # Path on Runpod
OUTPUT_DIR = "./results"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--dataset_path", type=str, default=DATASET_FILE)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint or 'True' to resume from latest")
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_path}...")
    # ... (rest of loading) ...

    # [SKIPPED CODE]

    # Training Arguments
    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=50,
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        save_total_limit=2, # Limits to saving only the last 2 checkpoints to save space
    )

    print("Starting SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=format_prompts,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    # Handle boolean vs string for resume_from_checkpoint
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint == "True":
        resume_checkpoint = True
    elif resume_checkpoint == "False":
        resume_checkpoint = False

    print(f"Training (Resume: {resume_checkpoint})...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    print(f"Saving model to {OUTPUT_DIR}/{NEW_MODEL_NAME}...")
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/{NEW_MODEL_NAME}")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/{NEW_MODEL_NAME}")
    print("Training Complete.")

if __name__ == "__main__":
    main()
