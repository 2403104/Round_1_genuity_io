"""
Local Evaluation Script - Inter-IIT Tech Meet 14.0
Evaluates retrieval results against ground truth (train queries only).

Metrics:
- Precision@3
- Precision@5
- MRR (Mean Reciprocal Rank)
"""

import json
from pathlib import Path
from typing import List, Dict, Set

def load_queries_with_gt(query_path: str) -> Dict[str, List[str]]:
    """Load queries and extract ground truth"""
    ground_truth = {}
    with open(query_path, 'r', encoding='utf-8') as f:
        for line in f:
            query = json.loads(line)
            query_id = query['query_id']
            gt = query.get('ground_truth', [])
            if gt:  # Only add if ground truth exists
                ground_truth[query_id] = gt
    return ground_truth

def load_results(results_path: str) -> Dict[str, List[str]]:
    """Load prediction results"""
    results = {}
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            query_id = result['query_id']
            pred = result['results']
            results[query_id] = pred
    return results

def precision_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
    """Calculate Precision@K"""
    if not predictions or not ground_truth:
        return 0.0
    
    # Take top-k predictions
    top_k = predictions[:k]
    
    # Count how many are in ground truth
    relevant = sum(1 for pred in top_k if pred in ground_truth)
    
    return relevant / k

def mean_reciprocal_rank(predictions: List[str], ground_truth: List[str]) -> float:
    """Calculate MRR (Mean Reciprocal Rank)"""
    if not predictions or not ground_truth:
        return 0.0
    
    # Find rank of first relevant document
    for rank, pred in enumerate(predictions, start=1):
        if pred in ground_truth:
            return 1.0 / rank
    
    return 0.0  # No relevant document found

def evaluate(results: Dict[str, List[str]], ground_truth: Dict[str, List[str]]) -> Dict[str, float]:
    """Evaluate all metrics"""
    p3_scores = []
    p5_scores = []
    mrr_scores = []
    
    for query_id, gt in ground_truth.items():
        if query_id not in results:
            print(f"Warning: No results for query {query_id}")
            continue
        
        preds = results[query_id]
        
        # Calculate metrics
        p3 = precision_at_k(preds, gt, k=3)
        p5 = precision_at_k(preds, gt, k=5)
        mrr = mean_reciprocal_rank(preds, gt)
        
        p3_scores.append(p3)
        p5_scores.append(p5)
        mrr_scores.append(mrr)
    
    # Aggregate
    metrics = {
        "Precision@3": sum(p3_scores) / len(p3_scores) if p3_scores else 0.0,
        "Precision@5": sum(p5_scores) / len(p5_scores) if p5_scores else 0.0,
        "MRR": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
        "Num_Queries": len(p3_scores)
    }
    
    return metrics

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_local.py <results_file>")
        print("Example: python evaluate_local.py query_results_flat.jsonl")
        sys.exit(1)
    
    results_path = sys.argv[1]
    query_path = "queries/queries_train.jsonl"  # Evaluation only works on train set
    
    print("=" * 60)
    print("LOCAL EVALUATION (Train Set Only)")
    print("=" * 60)
    
    # Load data
    print(f"Loading ground truth from {query_path}...")
    ground_truth = load_queries_with_gt(query_path)
    print(f"Loaded ground truth for {len(ground_truth)} queries")
    
    print(f"\nLoading results from {results_path}...")
    results = load_results(results_path)
    print(f"Loaded results for {len(results)} queries")
    
    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate(results, ground_truth)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Precision@3:  {metrics['Precision@3']:.4f}")
    print(f"Precision@5:  {metrics['Precision@5']:.4f}")
    print(f"MRR:          {metrics['MRR']:.4f}")
    print(f"Queries:      {metrics['Num_Queries']}")
    print("=" * 60)
    
    # Calculate weighted score (matching competition weights)
    # P@3 (30%), P@5 (20%), MRR (30%) = 80% total (rest is routing accuracy)
    weighted_score = (
        metrics['Precision@3'] * 0.30 +
        metrics['Precision@5'] * 0.20 +
        metrics['MRR'] * 0.30
    )
    print(f"\nWeighted Retrieval Score: {weighted_score:.4f}")
    print("(Note: This excludes Routing Accuracy which requires tree structure)")
    print("=" * 60)

if __name__ == "__main__":
    main()
