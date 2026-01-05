"""
Flat Retrieval Baseline - Inter-IIT Tech Meet 14.0
Simple cosine similarity-based retrieval (no tree structure).
This serves as a baseline for participants to compare against.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict
import time

# Paths
METADATA_CSV = "dataset/metadata.csv"
QUERY_FILE = "queries/queries_train.jsonl"  # Can be train or val
OUTPUT_FILE = "query_results_flat.jsonl"

def load_metadata(csv_path: str) -> pd.DataFrame:
    """Load metadata with embeddings"""
    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} documents")
    return df

def parse_embedding(emb_str: str) -> np.ndarray:
    """Parse embedding from JSON string"""
    emb_list = json.loads(emb_str)
    return np.array(emb_list, dtype=np.float32)

def load_queries(query_path: str) -> List[Dict]:
    """Load queries from JSONL file"""
    print(f"Loading queries from {query_path}...")
    queries = []
    with open(query_path, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))
    print(f"Loaded {len(queries)} queries")
    return queries

def embed_query(query_text: str, model) -> np.ndarray:
    """Generate embedding for query text"""
    emb = model.encode([query_text], convert_to_numpy=True)[0]
    return emb.astype(np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

def retrieve_top_k(query_emb: np.ndarray, doc_embeddings: np.ndarray, k: int = 5) -> List[int]:
    """Retrieve top-k documents by cosine similarity"""
    # Calculate similarities
    similarities = []
    for idx, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_emb, doc_emb)
        similarities.append((idx, sim))
    
    # Sort by similarity (desc) and return top-k indices
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = [idx for idx, _ in similarities[:k]]
    return top_k_indices

def main():
    print("=" * 60)
    print("FLAT RETRIEVAL BASELINE")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load metadata
    df = load_metadata(METADATA_CSV)
    
    # Parse embeddings
    print("Parsing document embeddings...")
    doc_embeddings = np.array([parse_embedding(emb) for emb in df['embedding']])
    doc_ids = df['id'].tolist()
    
    # Load embedding model
    print("Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load queries
    queries = load_queries(QUERY_FILE)
    
    # Process queries
    print("\nProcessing queries...")
    results = []
    for query in queries:
        query_id = query['query_id']
        query_text = query['query_text']
        
        # Embed query
        query_emb = embed_query(query_text, model)
        
        # Retrieve top-5
        top_k_indices = retrieve_top_k(query_emb, doc_embeddings, k=5)
        top_k_ids = [doc_ids[idx] for idx in top_k_indices]
        
        # Save result
        results.append({
            "query_id": query_id,
            "results": top_k_ids
        })
    
    # Save results
    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    elapsed_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"COMPLETE")
    print(f"{'=' * 60}")
    print(f"Processed {len(queries)} queries")
    print(f"Results saved to {OUTPUT_FILE}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Avg time per query: {elapsed_time/len(queries):.3f} seconds")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
