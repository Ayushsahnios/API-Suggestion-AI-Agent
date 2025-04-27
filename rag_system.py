# rag_system.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

def load_api_data(file_path="Sikka_APIs - Sikka_APIs.xlsx"):
    df = pd.read_excel(file_path)

    print(f"Columns found: {df.columns.tolist()}")

    if not all(col in df.columns for col in ['API Name', 'API Endpoints', 'Description']):
        raise Exception("Missing one of the required columns: API Name, API Endpoints, Description")

    api_records = []
    texts_for_embedding = []

    for idx, row in df.iterrows():
        api_name = str(row.get('API Name', '')).strip()
        endpoint = str(row.get('API Endpoints', '')).strip()
        description = str(row.get('Description', '')).strip()

        if api_name and endpoint:
            record = {
                "api_name": api_name,
                "endpoint": endpoint,
                "description": description
            }
            api_records.append(record)
            combined_text = f"{api_name} {endpoint} {description}"
            texts_for_embedding.append(combined_text)

    return api_records, texts_for_embedding

def embed_texts(texts):
    #Creates embeddings using a transformer model.
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings, model

def build_faiss_index(embeddings):
    """Creates a FAISS index for fast similarity search."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_faiss(index, embeddings, texts, save_path="api_faiss_index"):
    """Saves FAISS index and related files."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    faiss.write_index(index, os.path.join(save_path, "index.faiss"))
    np.save(os.path.join(save_path, "embeddings.npy"), embeddings)

    with open(os.path.join(save_path, "texts.txt"), "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")

    print("FAISS index and text data saved successfully!")

def prepare_rag(file_path="Sikka_APIs - Sikka_APIs.xlsx"):
    """Builds the entire RAG system."""
    api_records, texts_for_embedding = load_api_data(file_path)
    embeddings, _ = embed_texts(texts_for_embedding)
    index = build_faiss_index(np.array(embeddings))
    save_faiss(index, np.array(embeddings), texts_for_embedding)

    pd.DataFrame(api_records).to_json("api_faiss_index/api_records.json", orient='records', indent=2)
    print("API records saved separately!")

def search_apis(query, top_k=3, index_path="api_faiss_index"):
    """Searches APIs and returns structured API records."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index(os.path.join(index_path, "index.faiss"))

    # Load texts (for vector matching)
    with open(os.path.join(index_path, "texts.txt"), "r", encoding="utf-8") as f:
        texts = f.readlines()

    # Load structured API records
    import json
    with open(os.path.join(index_path, "api_records.json"), "r", encoding="utf-8") as f:
        api_records = json.load(f)

    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)

    results = []
    for idx in I[0]:
        results.append(api_records[idx])

    return results


if __name__ == "__main__":
    prepare_rag()
