import scvelo as scv
import numpy as np
import torch
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

def visualize_graph_structure(adata, adj_sparse, data):
    """Generate diagnostic plots to verify graph correctness."""
    print("   -> Generating inspection plots...")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    scv.pl.velocity_embedding_stream(
        adata, basis='umap', show=False, ax=ax[0], title="Velocity Stream (UMAP)"
    )

    subset = 200
    ax[1].spy(adj_sparse[:subset, :subset], markersize=1)
    ax[1].set_title(f"Velocity Graph (First {subset} cells)")
    ax[1].set_xlabel("Target Cell")
    ax[1].set_ylabel("Source Cell")

    classes, counts = np.unique(data.y.numpy(), return_counts=True)
    ax[2].bar(classes, counts)
    ax[2].set_title("Class Distribution")
    ax[2].set_xlabel("Class ID")
    ax[2].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("graph_inspection.png", dpi=200)
    print("   -> Saved graph inspection figure: graph_inspection.png")

def load_velocity_graph():

    print("\n" + "="*50)
    print("--- 1. Loading & Processing Pancreas Data ---")
    print("="*50)

    adata = scv.datasets.pancreas()
    print(f"   -> Raw Data Shape: {adata.shape}")

    # --- Filtering + Normalization ---
    print("   -> Filtering / Normalizing...")
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=500)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    # --- Dynamics ---
    print("   -> Recovering Dynamics...")
    scv.tl.recover_dynamics(adata, n_jobs=1)

    print("   -> Computing Velocity...")
    scv.tl.velocity(adata, mode="dynamical")

    print("   -> Building Velocity Graph...")
    scv.tl.velocity_graph(adata, n_jobs=1)

    print("\n" + "="*50)
    print("--- 2. Constructing PyTorch Geometric Graph ---")
    print("="*50)

    #============= A. Node Features (PCA) =============
    sc.tl.pca(adata, n_comps=30)
    x = torch.tensor(adata.obsm["X_pca"], dtype=torch.float)
    print(f"   -> Node Features: {x.shape}")

    #============= B. Edges =============
    adj = adata.uns["velocity_graph"]
    rows, cols = adj.nonzero()
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.tensor(adj.data, dtype=torch.float)

    print(f"   -> Velocity Graph Edges: {edge_index.shape[1]}")

    #============= C. Labels =============
    # robust extraction
    for key in ["cell_type", "clusters", "clusters_coarse", "louvain"]:
        if key in adata.obs:
            label_key = key
            break

    print(f"   -> Using label column: {label_key}")

    le = LabelEncoder()
    y = torch.tensor(le.fit_transform(adata.obs[label_key]), dtype=torch.long)
    classes = le.classes_
    print(f"   -> Classes ({len(classes)}): {classes}")

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )

    n = data.num_nodes
    perm = torch.randperm(n)
    split = int(0.8 * n)

    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.test_mask = torch.zeros(n, dtype=torch.bool)
    data.train_mask[perm[:split]] = True
    data.test_mask[perm[split:]] = True

    print("\n" + "="*50)
    print("FINAL PyG DATA OBJECT")
    print(data)
    print("="*50)

    visualize_graph_structure(adata, adj, data)

    return data, classes

if __name__ == "__main__":
    data, classes = load_velocity_graph()
