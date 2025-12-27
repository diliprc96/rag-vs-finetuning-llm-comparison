from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple


class RAGIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.id_to_text = {}

    def build(self, texts: List[str]):
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        d = embs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embs)
        self.id_to_text = {i: t for i, t in enumerate(texts)}

    def save(self, index_path: str, meta_path: str):
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            for i in range(len(self.id_to_text)):
                f.write(self.id_to_text[i].replace('\n', ' ') + "\n")

    def load(self, index_path: str, meta_path: str):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]
        self.id_to_text = {i: lines[i] for i in range(len(lines))}

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[int, float, str]]:
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q_emb, k)
        results = []
        for idx, score in zip(I[0], D[0]):
            results.append((int(idx), float(score), self.id_to_text[int(idx)]))
        return results


if __name__ == '__main__':
    # quick local demo
    texts = [
        "Near Earth's surface, g = 9.8 m/s^2 due to gravitational force...",
        "Free fall: a = g regardless of mass...",
        "Air resistance negligible for dense objects...",
    ]
    r = RAGIndex()
    r.build(texts)
    print(r.retrieve("Why is free fall 9.8 m/s^2?", k=3))
