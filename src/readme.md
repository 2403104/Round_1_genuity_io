# Semantic Tree-Based Document Retrieval  
### Inter-IIT Tech Meet 14.0 – Team Submission

This repository implements the full pipeline required for the Inter-IIT Tech Meet 14.0 Semantic Retrieval problem:

- Semantic tree construction  
- Tree-based hierarchical retrieval  
- Flat baseline retrieval  
- Reproducible run.sh pipeline

All requirements from the problem statement are satisfied.

## 1. Overview

The goal is to organize 203 research papers into a hierarchical semantic tree, and then perform efficient retrieval by routing each query through the tree.

This repository contains:

- build_tree.py → Builds semantic tree  
- tree_retrieval.py → Performs tree-based retrieval  
- flat_retrieval.py → Baseline TF-IDF retrieval  
- tree.json → Output tree  
- query_results_tree.jsonl → Tree retrieval predictions  
- query_results_flat.jsonl → Flat baseline predictions  

## 2. Dataset Description

We use the provided metadata.csv, queries_train.jsonl, and queries_val.jsonl.

metadata.csv contains:

- Document ID  
- File name  
- 384-d embedding from MiniLM (string JSON)  
- Optional title / summary fields  

Embeddings are converted into numpy.float arrays.

## 3. Tree Construction (build_tree.py)

Tree building follows a top-down recursive clustering approach.

### 3.1 PCA Projection (Speed Optimization)

384-dim embeddings → 64-dim PCA projection.  
Used only for clustering.  
The original 384-d embeddings remain untouched.

### 3.2 Clustering Strategy

At each level:

- Branch factor: 5  
- Maximum depth: 4  
- Minimum cluster size: 10  

If fewer than 2 samples → create leaf node.

### 3.3 Node Types

Internal node:

{
  "name": "...",
  "centroid": [384-dim],
  "size": number_of_docs,
  "children": {...},
  "is_leaf_container": false
}

Leaf container:

{
  "name": "...",
  "centroid": [...],
  "size": N,
  "children": {
      "doc_id": { "id": "...", "filename": "...", "embedding": [384 dims] }
  },
  "is_leaf_container": true
}

### 3.4 Output File

A valid tree.json under 100MB is produced.

## 4. Tree-Based Retrieval (tree_retrieval.py)

### 4.1 Query Encoding

Each query is encoded using all-MiniLM-L6-v2 and normalized.

### 4.2 Tree Index Flattening

Internal nodes → child centroids  
Leaf nodes → document embeddings  

### 4.3 Beam Search Routing Algorithm

For each query:

1. Start from root  
2. Compare query with centroids  
3. Select top 5 branches  
4. Descend  
5. At leaves, compute cosine similarity with documents  
6. Output top 5 results  

### 4.4 Output

query_results_tree.jsonl:

{"query_id": "q101", "results": ["id12","id45","id90","id3","id77"]}

## 5. Flat Retrieval Baseline (flat_retrieval.py)

Due to Windows Torch issues, TF-IDF retrieval is implemented:

1. Combine filename + summary  
2. Train TF-IDF  
3. Compute cosine similarity  
4. Select top-5  

Output: query_results_flat.jsonl

## 6. Running Pipeline (run.sh)

bash run.sh  
or  
wsl bash run.sh  

## 7. Repository Structure

.
├── tree.json  
├── query_results_tree.jsonl  
├── query_results_flat.jsonl  
├── run.sh  
├── src/  
│   ├── build_tree.py  
│   ├── tree_retrieval.py  
│   └── flat_retrieval.py  
├── dataset/  
│   └── metadata.csv  
└── queries/  
    ├── queries_train.jsonl  
    └── queries_val.jsonl  

## 8. Requirements Satisfaction

✔ Tree depth ≤ 5  
✔ Embeddings preserved  
✔ No vector DB  
✔ No LLM generation  
✔ PCA + beam search optimized  
✔ JSON schema valid  
