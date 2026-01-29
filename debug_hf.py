from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()

repo_id = "diliprc96/mistral-7b-physics-finetune"
token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

print(f"Checking repo: {repo_id}")
print(f"Token found: {'Yes' if token else 'No'}")

api = HfApi(token=token)

try:
    models = api.list_models(author="diliprc96")
    print("Models for diliprc96:")
    for m in models:
        print(f" - {m.id}")
except Exception as e:
    print(f"Error listing models: {e}")
