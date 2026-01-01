from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import os

BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = "results/mistral-7b-physics-finetune"

print(f"Checking adapter path: {ADAPTER_PATH}")
if not os.path.exists(ADAPTER_PATH):
    print("Adapter path does not exist!")
    exit(1)
print(f"Files in adapter path: {os.listdir(ADAPTER_PATH)}")

print("Loading Base Model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
try:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    print("Base Model Loaded.")

    print(f"Loading Adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("Adapter Loaded Successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
