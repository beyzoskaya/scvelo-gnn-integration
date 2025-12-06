import scvelo as scv
import numpy as np
import torch
import scanpy as sc
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import scipy.sparse 

# start with an scRNA-seq matrix (cells × genes)/ spliced and unspliced counts to compute RNA Velocity (the time derivative of gene expression)
# Because 2,000 genes are too sparse and noisy, project the data into a Latent PCA Space (30 dimensions)
# generate a "Ground Truth" velocity vector for every cell using the standard scVelo dynamical model --> vector v represents where the cell should move in the PCA space
def get_velocity_data(dataset_name='dentate'):
    print(f"--- 1. Initializing Data: {dataset_name.upper()} ---")
    
    # 1. Load the specific dataset
    if dataset_name == 'pancreas':
        adata = scv.datasets.pancreas()
    elif dataset_name == 'dentate':
        # Correct function name for Dentate Gyrus
        adata = scv.datasets.dentategyrus()
    else:
        raise ValueError("Dataset not supported. Choose 'pancreas' or 'dentate'.")

    # 2. Preprocessing
    # We keep top 2000 genes - these will be the targets for the generative model
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    
    # 3. Dynamics
    print("--- 2. Computing Dynamics ---")
    scv.tl.recover_dynamics(adata, n_jobs=1)
    scv.tl.velocity(adata, mode='dynamical')
    
    # 4. Graph & Projection
    print("--- 3. Constructing Graph ---")
    scv.tl.velocity_graph(adata, n_jobs=1)
    
    print("--- 4. Projecting to PCA ---")
    sc.tl.pca(adata, n_comps=30)
    scv.tl.velocity_embedding(adata, basis='pca')
    
    # 5. Build Tensor Data
    # A. Input Features: Latent State (PCA)
    X = torch.tensor(adata.obsm['X_pca'], dtype=torch.float)
    
    # B. Input/Target: Velocity Vector in PCA space
    V_pca = torch.tensor(adata.obsm['velocity_pca'], dtype=torch.float)
    
    if torch.isnan(V_pca).any():
        V_pca = torch.nan_to_num(V_pca)

    # C. NEW TARGET: Raw Gene Expression (For Generative Model)
    print("--- 5. Extracting Gene Expression Matrix ---")
    # adata.X is usually a sparse matrix. We need a dense tensor for the decoder.
    if scipy.sparse.issparse(adata.X):
        y_genes = torch.tensor(adata.X.toarray(), dtype=torch.float)
    else:
        y_genes = torch.tensor(adata.X, dtype=torch.float)
        
    print(f"   -> Extracted Gene Matrix Shape: {y_genes.shape}")

    # D. Graph Edges
    adj = adata.uns['velocity_graph']
    rows, cols = adj.nonzero()
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.tensor(adj.data, dtype=torch.float)
    
    # 6. Construct Final Data Object
    data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, v=V_pca, y_genes=y_genes)
    
    # Masks (80/20 split)
    # Graph Edges
    # Every node is a single Cell
    # Node Features (X): The cell's current transcriptional state (its 30 PCA coordinates)
    # The Edges: Connections between cells
    # An edge exists between Cell A and Cell B if they are transcriptomically similar AND there is a high probability that Cell A is differentiating into Cell B
    n_nodes = data.num_nodes
    perm = torch.randperm(n_nodes)
    val_size = int(n_nodes * 0.2)
    data.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.train_mask[perm[val_size:]] = True
    data.val_mask[perm[:val_size]] = True

    adata.uns['dataset_name'] = dataset_name
    
    print(f"Data Loaded: {data.num_nodes} cells. Topology: {dataset_name}")
    return data, adata

if __name__ == "__main__":
    data, adata = get_velocity_data('dentate')