import json, sys, os
import numpy as np
from sentence_transformers import SentenceTransformer

TREE_PATH = "tree.json"
QUERY_FILE = "queries/queries_val.jsonl" if len(sys.argv) < 2 else sys.argv[1]
OUTPUT_FILE = "query_results_tree.jsonl" if len(sys.argv) < 3 else sys.argv[2]

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_DOCS = 5
BEAM_WIDTH = 5

def load_model():
    return SentenceTransformer(MODEL_NAME)

def cosine_similarity(q_vec, candidates):
    return np.dot(candidates, q_vec)

def flatten_tree(tree):
    index = {}
    def walk(node):
        name = node["name"]
        is_leaf_container = node.get("is_leaf_container", False)
        children = node["children"]
        if is_leaf_container:
            doc_ids = []
            doc_embs = []
            for doc_id, doc_data in children.items():
                doc_ids.append(doc_id)
                doc_embs.append(doc_data["embedding"])
            index[name] = {
                "type": "leaf",
                "ids": doc_ids,
                "matrix": np.array(doc_embs)
            }
        else:
            child_names = []
            child_centroids = []
            for child_name, child_node in children.items():
                child_names.append(child_name)
                child_centroids.append(child_node["centroid"])
                walk(child_node)
            index[name] = {
                "type": "internal",
                "ids": child_names,
                "matrix": np.array(child_centroids)
            }
    walk(tree)
    return index

def run_retrieval(model, index, queries):
    results_out = []
    q_texts = [q["query_text"] for q in queries]
    q_vecs = model.encode(q_texts, convert_to_numpy=True, normalize_embeddings=True)
    for i, q_vec in enumerate(q_vecs):
        qid = queries[i]["query_id"]
        beam = ["root"]
        candidate_docs = []
        for _ in range(10):
            if not beam:
                break
            next_beam = []
            for node_name in beam:
                node_data = index.get(node_name)
                if not node_data:
                    continue
                scores = cosine_similarity(q_vec, node_data["matrix"])
                if node_data["type"] == "leaf":
                    for idx, score in enumerate(scores):
                        doc_id = node_data["ids"][idx]
                        candidate_docs.append((doc_id, score))
                else:
                    num_children = len(scores)
                    k_safe = min(BEAM_WIDTH, num_children)
                    if k_safe > 0:
                        top_k_indices = np.argsort(scores)[-k_safe:]
                        for idx in top_k_indices:
                            child_name = node_data["ids"][idx]
                            next_beam.append(child_name)
            beam = next_beam
        candidate_docs.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        final_ids = []
        for doc_id, score in candidate_docs:
            if doc_id not in seen:
                final_ids.append(doc_id)
                seen.add(doc_id)
            if len(final_ids) == TOP_K_DOCS:
                break
        results_out.append({
            "query_id": qid,
            "results": final_ids
        })
    return results_out

def main():
    with open(TREE_PATH, "r") as f:
        tree = json.load(f)
    index = flatten_tree(tree)
    with open(QUERY_FILE, "r") as f:
        queries = [json.loads(line) for line in f]
    model = load_model()
    results = run_retrieval(model, index, queries)
    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
