# Inter-IIT Tech Meet 14.0 - Semantic Tree Retrieval Challenge

<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
</head>
<body>

<h1>Semantic Tree-Based Document Retrieval</h1>

<p>
This repository contains an implementation for the Inter-IIT Tech Meet 14.0 problem
"Semantic Tree-Based Document Retrieval".
The goal is to organize research documents into a hierarchical tree and perform efficient
semantic retrieval using tree routing.
</p>

<hr>

<h2>Problem Objective</h2>

<p>
Given 203 research documents with embeddings, the task is to:
</p>

<ul>
  <li>Build a hierarchical semantic tree using document embeddings.</li>
  <li>Use the tree to route and retrieve relevant documents for each query.</li>
  <li>Return exactly 5 document IDs per query.</li>
  <li>Compare against a flat retrieval baseline.</li>
</ul>

<hr>

<h2>Input Files</h2>

<ul>
  <li>dataset/metadata.csv — document IDs, filenames, and 384-dim embeddings</li>
  <li>queries/queries_val.jsonl — queries</li>
  <li>tree_schema.json — schema for tree validation</li>
</ul>

<hr>

<h2>Output Files</h2>

<ul>
  <li>tree.json — hierarchical semantic tree</li>
  <li>query_results_tree.jsonl — tree-based retrieval results</li>
  <li>query_results_flat.jsonl — flat retrieval baseline</li>
</ul>

<hr>

<h2>Tree Construction</h2>

<p>
The tree is built recursively using KMeans clustering on projected embeddings.
</p>

<ul>
  <li>Embeddings are loaded from metadata.csv.</li>
  <li>PCA is applied to reduce dimensionality for clustering.</li>
  <li>KMeans clusters documents into child nodes.</li>
  <li>Recursion continues until maximum depth or minimum cluster size is reached.</li>
  <li>Leaf nodes store document IDs, filenames, and embeddings.</li>
  <li>Internal nodes store centroids, size, and children.</li>
</ul>

<hr>

<h2>Tree Building Script</h2>

<p>File: build_tree.py</p>

<pre>
Loads metadata → projects embeddings → clusters recursively → saves tree.json
</pre>

<hr>

<h2>Flat Retrieval</h2>

<p>
Flat retrieval computes similarity between queries and documents using TF-IDF text features.
This serves only as a baseline and is not graded.
</p>

<ul>
  <li>TF-IDF vectorization of filenames and summaries.</li>
  <li>Cosine similarity between query vectors and document vectors.</li>
  <li>Top-5 documents returned per query.</li>
</ul>

<hr>

<h2>Tree-Based Retrieval</h2>

<p>
Tree retrieval routes each query through the semantic tree using embedding similarity.
</p>

<ul>
  <li>Queries are encoded using all-MiniLM-L6-v2.</li>
  <li>At each internal node, similarity is computed against child centroids.</li>
  <li>Top children are selected and traversed recursively.</li>
  <li>Leaf document embeddings are compared and ranked.</li>
  <li>Top-5 unique document IDs are returned.</li>
</ul>

<hr>

<h2>Retrieval Pipeline</h2>

<ol>
  <li>Load tree.json.</li>
  <li>Flatten tree into an index.</li>
  <li>Encode queries.</li>
  <li>Route queries through the tree.</li>
  <li>Collect and rank candidate documents.</li>
  <li>Save results to query_results_tree.jsonl.</li>
</ol>

<hr>

<h2>How to Run</h2>

<pre>
python build_tree.py
python tree_retrieval.py
python flat_retrieval.py
</pre>

<hr>

<h2>Constraints Followed</h2>

<ul>
  <li>Only numpy, pandas, sklearn, sentence-transformers are used.</li>
  <li>No vector databases or external retrieval systems are used.</li>
  <li>No ground truth is used during tree construction.</li>
</ul>

<hr>

<h2>Author</h2>

<p>Ankit Kumar</p>

</body>
</html>
