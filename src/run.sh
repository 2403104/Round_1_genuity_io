#!/bin/bash
set -e

echo "=== Building Semantic Tree ==="
python build_tree.py

echo "=== Running Tree-Based Retrieval ==="
python tree_retrieval.py queries/queries_val.jsonl query_results_tree.jsonl

echo "=== Running Flat Retrieval Baseline ==="
python flat_retrieval.py queries/queries_val.jsonl query_results_tree.jsonl

echo "Generated: tree.json, query_results_tree.jsonl, query_results_flat.jsonl"
