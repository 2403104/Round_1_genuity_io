# Inter-IIT Tech Meet 14.0 - Semantic Tree Retrieval Challenge

##  Overview

Build a **semantic tree-based document retrieval system** that organizes 203 research papers into a hierarchical structure and retrieves relevant documents for natural language queries.

---

##  Competition Goal

Given a dataset of research papers and queries:
1. **Build a semantic tree** (`tree.json`) that clusters documents hierarchically
2. **Implement tree-based retrieval** to answer queries
3. **Submit predictions** on validation queries
4. **Beat the flat baseline** using structural advantages

---

##  Package Contents

```
competition_package/
├── dataset/
│   ├── metadata.csv           # 203 documents with embeddings (384-dim)
│   └── documents/             # PDF files by domain (6 scientific fields)
│
├── queries/
│   ├── queries_train.jsonl    # 100 queries WITH ground truth (for practice)
│   └── queries_val.jsonl      # 30 queries WITHOUT ground truth (for submission)
│
├── helpers/
│   ├── flat_retrieval.py      # Baseline flat retrieval script
│   └── evaluate_local.py      # Local evaluation tool (train queries only)
│
├── schemas/
│   ├── tree_schema.json       # Validation schema for tree.json
│   ├── query_schema.json      # Validation schema for queries
│   └── submission_schema.json # Validation schema for results
│
├── Problem_Statement.pdf      # Official problem statement and rules
└── README.md                  # This file
```

---

##  Quick Start

### 1. Install Dependencies
Ensure you are using **Python 3.9+**.

```bash
pip install -r requirements.txt
```

### 2. Run Baseline (Flat Retrieval)

```bash
python helpers/flat_retrieval.py
```

This creates `query_results_flat.jsonl` with predictions for train queries.

### 3. Evaluate Locally

```bash
python helpers/evaluate_local.py query_results_flat.jsonl
```

**Expected Baseline Scores:**
- Precision@3: ~0.01
- Precision@5: ~0.02
- MRR: ~0.045

---

##  Dataset Overview

### Documents (203 papers)
- **6 Domains**: Computer Science, Physics, Mathematics, Biology, Economics, Statistics
- **Format**: Academic papers (PDFs) from arXiv
- **Embeddings**: Pre-computed 384-dimensional vectors (`all-MiniLM-L6-v2`)
- **Metadata**: `id`, `file_name`, `file_location`, `summary`, `embedding`, `Level_0`, `Level_1`, `Level_2`

### Queries
- **Train**: 100 queries with ground truth (for development)
- **Val**: 30 queries without ground truth (for final submission)

---

##  Your Task

### Step 1: Build Semantic Tree

**Your Goal:** Design and implement your own logic to cluster documents into a hierarchical tree structure. You decide the clustering algorithm (K-Means, Agglomerative, etc.) and tree topology.

**Requirements:**
- Store as `tree.json` (max 100MB)
- Each internal node must have:
  - `name`, `centroid` (384-dim), `size`, `children`
- Each leaf node must have:
  - `id`, `file_name`, `embedding` (384-dim)
- Depth: Variable (recommended 3-5 levels)
- Branching: Variable (recommended 3-10 children per node)

**Example Structure:**
```json
{
  "name": "root",
  "centroid": [0.1, 0.2, ..., 0.5],
  "size": 203,
  "children": {
    "Computer_Science": {
      "name": "Computer_Science",
      "centroid": [...],
      "size": 80,
      "children": {...}
    },
    "Physics": {...}
  }
}
```

### Step 2: Implement Tree-Based Retrieval

**Your Goal:** Implement a custom traversal algorithm that navigates your `tree.json` to find relevant documents for a given query. You define how to route queries through the tree (e.g., top-k, beam search).

**Routing Strategy:**
- Embed the query using `sentence-transformers`
- At each level, compute similarity between query embedding and child centroids
- Select top-k most similar branches (k=3 to 5 recommended)
- Continue until reaching leaves
- Return top-5 documents

### Step 3: Generate Predictions

Run your system on **validation queries** (`queries_val.jsonl`):

```python
# Pseudocode
for query in queries_val:
    results = tree_retrieval(query)  # Your implementation
    save_result(query_id, results[:5])
```

**Output Format** (`query_results_tree.jsonl`):
You must generate a JSONL file where each line corresponds to a validation query:
```json
{"query_id": "q101", "results": ["doc_id_A", "doc_id_B", "doc_id_C", "doc_id_D", "doc_id_E"]}
{"query_id": "q102", "results": [...]}
```
*Note: Exactly 5 results per query are required.*

---

##  Evaluation Metrics

### Retrieval Quality (80%)
1. **Precision@3 (30%)**: Fraction of relevant docs in top 3
2. **Precision@5 (20%)**: Fraction of relevant docs in top 5  
3. **MRR (30%)**: Mean Reciprocal Rank of first relevant doc
4. **Routing Accuracy (20%)**: % of queries where ground truth was reachable in visited subtrees

### Tree Quality (10%)
- Semantic coherence (intra-cluster vs inter-cluster similarity)
- Structural balance (node size distribution)

### Efficiency (5%)
- Query latency (CPU only)
- Memory usage

### Reproducibility (5%)
- Code runs successfully via `run.sh`

---

##  Constraints

### Hardware
- **CPU Only** (No GPU allowed)
- **RAM Limit**: 16GB
- **Time Limit**: 30 minutes for full retrieval pipeline

### Allowed Libraries
- `numpy`, `pandas`, `scikit-learn`, `scipy`
- `sentence-transformers` (Must use `all-MiniLM-L6-v2` model to match dataset)
- Standard Python libraries

### Prohibited
- Pre-built vector databases (Pinecone, FAISS, Weaviate, etc.)
- RAG frameworks (LangChain, LlamaIndex)
- LLM generation APIs (OpenAI, Anthropic, etc.)

---

##  Submission Requirements

### Files to Submit
You must submit a zip file containing the following structure:

```
submission/
├── tree.json                  # Your custom semantic tree structure
├── query_results_tree.jsonl   # Your predictions on queries_val.jsonl
├── query_results_flat.jsonl   # Baseline predictions
├── run.sh                     # Script to reproduce your results
└── code/                      # All your source code files
```

### Validation Checklist
- [ ] `tree.json` is valid JSON and under 100MB
- [ ] `query_results_*.jsonl` has exactly 30 predictions (one per val query)
- [ ] Each prediction has exactly 5 results (file IDs from `metadata.csv`)
- [ ] `run.sh` executes successfully in clean environment
- [ ] **Metadata is unmodified**: You MUST NOT modify `dataset/metadata.csv`.
- [ ] Completes within 30 minutes on standard laptop

---

##  Development Tips

### 1. Use the Training Set
Practice on `queries_train.jsonl` which includes ground truth:

```bash
# Generate predictions
python your_tree_retrieval.py

# Evaluate locally
python helpers/evaluate_local.py query_results_tree.jsonl
```

### 2. Experiment with Tree Depth
- Shallow trees (2-3 levels): Faster, less precise
- Deep trees (5-7 levels): Slower, more precise
- Find the sweet spot for your clustering algorithm

### 3. Optimize Routing
- **Top-1 routing**: Fast but may miss relevant branches
- **Top-k routing (k=3-5)**: Better recall, slightly slower
- **Beam search**: Best recall, highest latency

### 4. Improve Clustering
- Try different algorithms: K-Means, Agglomerative, DBSCAN
- **Semantic Focus**: You may inspect metadata (like `Level_0`) for analysis, but your tree structure **must be based on semantic embeddings**. Do not hardcode directory paths as clusters.
- Balance cluster sizes for efficient search

### 5. Profile Performance
```python
import time
start = time.time()
results = tree_retrieval(query)
print(f"Latency: {time.time() - start:.3f}s")
```

---

##  Resources

### Understanding the Dataset
```python
import pandas as pd
df = pd.read_csv('dataset/metadata.csv')
print(df.head())
print(df['Level_0'].value_counts())  # Domain distribution
```

### Loading Embeddings
```python
import json
import numpy as np

embedding_str = df.iloc[0]['embedding']
embedding = np.array(json.loads(embedding_str))
print(embedding.shape)  # (384,)
```

### Building Clusters
```python
from sklearn.cluster import KMeans

embeddings = np.array([json.loads(e) for e in df['embedding']])
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(embeddings)
```

---

## 🎓 Evaluation Example

Train a model → Generate predictions → Evaluate:

```bash
# 1. Build your tree
python build_tree.py  # Creates tree.json

# 2. Run retrieval on train queries
python tree_retrieval.py queries/queries_train.jsonl

# 3. Evaluate
python helpers/evaluate_local.py query_results_tree.jsonl
```

**Sample Output:**
```
============================================================
RESULTS
============================================================
Precision@3:  0.2500
Precision@5:  0.3200
MRR:          0.4100
Queries:      100
============================================================
Weighted Retrieval Score: 0.3070
============================================================
```

---

## ❓ FAQs

### Q: Can I use the directory structure for retrieval?
**A:** No! Your system must work with a flat directory. Don't rely on `Level_0`, `Level_1`, etc. for tree construction (though you can use them as hints during training).

### Q: How do I handle queries that don't match any documents well?
**A:** Return the top-5 most similar documents anyway. Partial credit is given based on ranking.

### Q: Can I use pre-trained models from HuggingFace?
**A:** Yes, BUT you must use `sentence-transformers` with the `all-MiniLM-L6-v2` model for generating query embeddings. Using a different model will result in mismatched vectors and poor performance.

### Q: What if my tree.json is too large?
**A:** Reduce tree depth, prune small clusters, or quantize centroids to float16.

---

## 📞 Support

For questions or issues:
- Check `Problem_Statement.pdf` for official rules
- Review `schemas/` for validation requirements
- Test with `helpers/evaluate_local.py` before submission

---

**Good luck!** 🚀
