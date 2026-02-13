import os
import json
import logging
from typing import List, Tuple
from evaluation.rag import RAGIndex

logger = logging.getLogger(__name__)

INDEX_PATH = "evaluation/rag_index.faiss"
META_PATH = "evaluation/rag_meta.txt"
DATA_PATH = "data_extraction/alpaca_physics_5k_cleaned.jsonl"

def build_index_from_dataset():
    logger.info("Building RAG index from dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    texts = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                # Combine instruction and output for context
                text = f"Q: {item.get('instruction', '')}\nA: {item.get('output', '')}\n"
                texts.append(text)
            except:
                continue
    
    rag = RAGIndex()
    rag.build(texts)
    rag.save(INDEX_PATH, META_PATH)
    logger.info(f"Index built with {len(texts)} documents.")
    return rag

def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        return build_index_from_dataset()
    
    rag = RAGIndex()
    rag.load(INDEX_PATH, META_PATH)
    return rag

def retrieve(rag_index, query, k=3):
    return rag_index.retrieve(query, k=k)

def format_docs(docs: List[Tuple[int, float, str]]) -> str:
    # docs is list of (id, score, text)
    context = ""
    for _, _, text in docs:
        context += text + "\n---\n"
    return context.strip()
