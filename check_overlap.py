import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

EVAL_FILE = "evaluation/physics_questions_50.json"
TRAIN_FILE = "data_extraction/alpaca_physics_5k_cleaned.jsonl"

def check_overlap():
    print("Loading datasets...")
    
    # Load Eval
    with open(EVAL_FILE, 'r') as f:
        eval_data = json.load(f)
    eval_questions = [q['question'] for q in eval_data]
    print(f"Loaded {len(eval_questions)} evaluation questions.")
    
    # Load Train
    train_questions = []
    with open(TRAIN_FILE, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                # Check instruction and input
                text = item.get('instruction', '') + " " + item.get('input', '')
                train_questions.append(text.strip())
            except:
                pass
    print(f"Loaded {len(train_questions)} training samples.")

    # 1. Exact Match Check
    print("\n--- Exact/Substring Match Check ---")
    exact_matches = 0
    for eq in eval_questions:
        for tq in train_questions:
            if eq in tq or tq in eq:
                exact_matches += 1
                break
    print(f"Exact/Substring Matches: {exact_matches}")

    # 2. Semantic Similarity (TF-IDF Cosine)
    print("\n--- Semantic Similarity Check (Top 3) ---")
    # Fit on all data
    vectorizer = TfidfVectorizer(stop_words='english').fit(train_questions + eval_questions)
    train_vecs = vectorizer.transform(train_questions)
    eval_vecs = vectorizer.transform(eval_questions)
    
    # Calculate similarity
    sim_matrix = cosine_similarity(eval_vecs, train_vecs)
    
    max_sims = np.max(sim_matrix, axis=1)
    avg_max_sim = np.mean(max_sims)
    
    print(f"Average Max Similarity: {avg_max_sim:.4f}")
    
    # Show closest matches
    top_indices = np.argsort(max_sims)[-3:][::-1]
    for idx in top_indices:
        print(f"\nEval Q: {eval_questions[idx]}")
        best_train_idx = np.argmax(sim_matrix[idx])
        print(f"Best Train Match: {train_questions[best_train_idx]}")
        print(f"Similarity Score: {max_sims[idx]:.4f}")

if __name__ == "__main__":
    check_overlap()
