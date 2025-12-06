import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scanpy as sc
import scvelo as scv
from matplotlib.collections import LineCollection

def plot_graph_connectivity(data, adata, subset_size=500):
    print("   -> Generating Graph Connectivity Plot...")
    X = data.x.numpy()
    edge_index = data.edge_index.numpy()
    
    # Subsample nodes
    if data.num_nodes > subset_size:
        indices = np.random.choice(data.num_nodes, subset_size, replace=False)
    else:
        indices = np.arange(data.num_nodes)

    mask = np.isin(edge_index[0], indices) & np.isin(edge_index[1], indices)
    subset_edges = edge_index[:, mask]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:,0], X[:,1], c='lightgrey', s=5, alpha=0.3, label='Cells')
    
    start_pts = X[subset_edges[0]]
    end_pts = X[subset_edges[1]]
    lines = list(zip(start_pts[:, :2], end_pts[:, :2]))
    
    lc = LineCollection(lines, colors='teal', linewidths=0.1, alpha=0.3)
    plt.gca().add_collection(lc)
    plt.scatter(X[indices,0], X[indices,1], c='black', s=10, zorder=5)
    
    plt.title(f"Graph Topology ({adata.uns.get('dataset_name', 'dataset')})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig("visual_1_graph_topology.png", dpi=200)
    plt.close()

def plot_per_cluster_performance(adata, cosine_scores):
    print("   -> Generating Per-Cluster Performance Plot...")
    
    # Robust key finding
    key = None
    for k in ['clusters', 'louvain', 'cell_type']:
        if k in adata.obs:
            key = k
            break
    
    if key is None:
        print("   ! Warning: No cluster labels found for boxplot.")
        return

    clusters = adata.obs[key].values
    
    import pandas as pd
    df = pd.DataFrame({
        'Cell Type': clusters,
        'Cosine Similarity': cosine_scores
    })
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Cell Type', y='Cosine Similarity', palette='viridis')
    plt.axhline(0, color='r', linestyle='--', alpha=0.5, label='Random Guess')
    plt.title(f"GNN Predictive Performance ({adata.uns.get('dataset_name', 'dataset')})")
    plt.xticks(rotation=45)
    plt.ylim(-0.5, 1.05) 
    plt.tight_layout()
    plt.savefig("visual_2_cluster_performance.png", dpi=150)
    plt.close()

def plot_streamlines(adata, V_pred_pca):
    print("   -> Generating Streamline Comparison...")
    adata_pred = adata.copy()
    adata_pred.obsm['velocity_pca'] = V_pred_pca
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    scv.pl.velocity_embedding_stream(adata, basis='pca', title='Ground Truth', ax=ax[0], show=False)
    scv.pl.velocity_embedding_stream(adata_pred, basis='pca', title='GNN Predicted (Latent)', ax=ax[1], show=False)
    
    plt.savefig("visual_3_streamlines.png", dpi=150)
    plt.close()