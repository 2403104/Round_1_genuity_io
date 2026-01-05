import json, os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DATA_META = os.path.join("dataset", "metadata.csv")
OUT_TREE = "tree.json"

RANDOM_STATE = 42
PROJ_DIM = 64
BRANCH_FACTOR = 5
MAX_DEPTH = 4
MIN_CLUSTER_SIZE = 10

def load_data():
    df = pd.read_csv(DATA_META)
    df["embedding"] = df["embedding"].apply(lambda x: np.array(json.loads(x), dtype=float))
    return df

def get_centroid(embeddings):
    return np.mean(embeddings, axis=0).tolist()

def make_leaf_node(df, name):
    children = {}
    for _, row in df.iterrows():
        doc_id = str(row["id"])
        children[doc_id] = {
            "id": doc_id,
            "filename": row.get("file_name", ""),
            "embedding": row["embedding"].tolist()
        }
    return {
        "name": name,
        "centroid": get_centroid(df["embedding"].to_list()),
        "size": len(df),
        "children": children,
        "is_leaf_container": True
    }

def recursive_build(df, proj_embs, depth, name_prefix):
    n = len(df)
    if depth >= MAX_DEPTH or n <= MIN_CLUSTER_SIZE:
        return make_leaf_node(df, name_prefix)
    k = min(BRANCH_FACTOR, n)
    if k < 2:
        return make_leaf_node(df, name_prefix)
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(proj_embs)
    children = {}
    for i in range(k):
        mask = (labels == i)
        sub_df = df[mask]
        if len(sub_df) == 0:
            continue
        sub_proj = proj_embs[mask]
        child_name = f"{name_prefix}_L{depth}_C{i}"
        child_node = recursive_build(sub_df, sub_proj, depth + 1, child_name)
        children[child_name] = child_node
    if not children:
        return make_leaf_node(df, name_prefix)
    return {
        "name": name_prefix,
        "centroid": get_centroid(df["embedding"].to_list()),
        "size": n,
        "children": children,
        "is_leaf_container": False
    }

def main():
    df = load_data()
    matrix = np.vstack(df["embedding"].values)
    pca = PCA(n_components=min(PROJ_DIM, len(df)), random_state=RANDOM_STATE)
    proj_matrix = pca.fit_transform(matrix)
    norms = np.linalg.norm(proj_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    proj_matrix = proj_matrix / norms
    root_node = recursive_build(df, proj_matrix, 0, "root")
    tree_structure = {
        "name": "root",
        "centroid": root_node["centroid"],
        "size": root_node["size"],
        "children": root_node["children"],
        "is_leaf_container": root_node.get("is_leaf_container", False)
    }
    with open(OUT_TREE, "w") as f:
        json.dump(tree_structure, f)

if __name__ == "__main__":
    main()
