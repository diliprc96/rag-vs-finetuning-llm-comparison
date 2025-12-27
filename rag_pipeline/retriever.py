import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_PATH = "rag_pipeline/faiss_index"

def load_index():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index not found at {INDEX_PATH}. Run indexer.py first.")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Allow dangerous deserialization because we created the index ourselves
    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def retrieve(db, query, k=5):
    docs = db.similarity_search(query, k=k)
    return docs

def format_docs(docs):
    return "\n\n".join([f"[Source: {d.metadata.get('title', 'Unknown')}]\n{d.page_content}" for d in docs])

if __name__ == "__main__":
    # Test
    try:
        db = load_index()
        query = "What is the scope of physics?"
        print(f"Query: {query}")
        results = retrieve(db, query)
        print(f"Found {len(results)} results:")
        print(format_docs(results))
    except Exception as e:
        print(f"Error: {e}")
