import json
import os
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

INPUT_FILE = "data_extraction/openstax_physics_vol1_ch1_6.json"
INDEX_PATH = "rag_pipeline/faiss_index"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print("Loading raw data...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    print("Chunking documents...")
    for section in data:
        # Create a document for each chunk
        chunks = text_splitter.split_text(section["content"])
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": section["url"],
                    "title": section["title"],
                    "chapter": section["chapter"]
                }
            )
            documents.append(doc)
    
    print(f"Created {len(documents)} chunks.")

    print("Initializing Embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Building FAISS index...")
    db = FAISS.from_documents(documents, embeddings)

    print(f"Saving index to {INDEX_PATH}...")
    db.save_local(INDEX_PATH)
    print("Done.")

if __name__ == "__main__":
    main()
