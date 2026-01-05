import json
import pandas as pd
import numpy as np
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
META_PATH = "dataset/metadata.csv"
QUERIES_PATH = "queries/queries_val.jsonl"
OUT_PATH = "query_results_flat.jsonl"
TOPK = 5

def main():
    print("1. Loading Data (No Transformers)...")
    df = pd.read_csv(META_PATH)
    
    # We will match keywords in the Query against the Abstract/Title in metadata
    # If 'abstract' or 'summary' isn't there, we use 'file_name' or any text column available.
    # Adjust column names based on your actual CSV. 
    # Usually: 'title', 'abstract', 'summary', or just 'file_name'
    
    # Fill NaN with empty string to prevent errors
    df_text = df["file_name"].fillna("") + " " + df.get("summary", df.get("title", "")).fillna("")
    doc_ids = df["id"].astype(str).values

    print("2. Loading Queries...")
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f]
    
    q_texts = [q["query_text"] for q in queries]
    q_ids = [q["query_id"] for q in queries]

    print("3. Vectorizing Text (TF-IDF)...")
    # Limit max_features to keep it fast
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    
    # Learn vocabulary from docs and queries together to ensure overlap
    all_texts = df_text.tolist() + q_texts
    vectorizer.fit(all_texts)
    
    # Transform docs and queries to sparse vectors
    doc_matrix = vectorizer.transform(df_text)
    q_matrix = vectorizer.transform(q_texts)

    print("4. Computing Similarity...")
    # Compute Cosine Similarity (Query vs All Docs)
    # Result is a matrix of shape (n_queries, n_docs)
    scores = cosine_similarity(q_matrix, doc_matrix)

    print(f"5. Saving Top {TOPK} results...")
    with open(OUT_PATH, "w") as f:
        for i, row_scores in enumerate(scores):
            # Sort scores descending and take top K
            best_indices = np.argsort(row_scores)[-TOPK:][::-1]
            top_docs = doc_ids[best_indices].tolist()
            
            record = {
                "query_id": q_ids[i],
                "results": top_docs
            }
            f.write(json.dumps(record) + "\n")

    print(f"Done! Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()