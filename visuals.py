import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scanpy as sc
import scvelo as scv
from matplotlib.collections import LineCollection
import pandas as pd

def plot_graph_connectivity(data, adata, subset_size=500):
    # (Same as before - keeping it short for this snippet)
    pass 

def plot_comparison_boxplots(adata, scores_gcn, scores_gat):
    """
    Side-by-side boxplots to statistically compare GCN vs GAT.
    """
    print("   -> Generating Model Comparison Boxplot...")
    
    key = None
    for k in ['clusters', 'louvain', 'cell_type']:
        if k in adata.obs:
            key = k
            break
    if not key: return

    clusters = adata.obs[key].values
    
    # Create combined DataFrame
    df_gcn = pd.DataFrame({'Cell Type': clusters, 'Cosine Sim': scores_gcn, 'Model': 'GCN'})
    df_gat = pd.DataFrame({'Cell Type': clusters, 'Cosine Sim': scores_gat, 'Model': 'GAT'})
    df_all = pd.concat([df_gcn, df_gat])
    
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df_all, x='Cell Type', y='Cosine Sim', hue='Model', palette={'GCN': 'steelblue', 'GAT': 'orange'})
    plt.axhline(0, color='r', linestyle='--', alpha=0.5)
    plt.title("Model Showdown: GCN vs GAT by Cell Type")
    plt.xticks(rotation=45)
    plt.ylim(-0.5, 1.05)
    plt.tight_layout()
    plt.savefig("compare_2_cluster_performance.png", dpi=200)
    plt.close()

def plot_comparison_streamlines(adata, v_pred_gcn, v_pred_gat):
    """
    3-Panel Plot: Ground Truth | GCN | GAT
    """
    print("   -> Generating Comparison Streamlines...")
    
    adata_gcn = adata.copy()
    adata_gcn.obsm['velocity_pca'] = v_pred_gcn
    
    adata_gat = adata.copy()
    adata_gat.obsm['velocity_pca'] = v_pred_gat
    
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    
    scv.pl.velocity_embedding_stream(adata, basis='pca', title='Ground Truth', ax=ax[0], show=False)
    scv.pl.velocity_embedding_stream(adata_gcn, basis='pca', title='GCN Prediction (Smoothed)', ax=ax[1], show=False)
    scv.pl.velocity_embedding_stream(adata_gat, basis='pca', title='GAT Prediction (Attention)', ax=ax[2], show=False)
    
    plt.savefig("compare_3_streamlines.png", dpi=200)
    plt.close()

def plot_attention_map(adata, attention_weights, edge_index):
    """
    Visualizes Biological Drivers (Nodes with high incoming attention).
    """
    print("   -> Generating Biological Attention Map...")
    
    num_nodes = adata.shape[0]
    node_score = np.zeros(num_nodes)
    
    sources = edge_index[0].cpu().numpy()
    weights = attention_weights.detach().cpu().numpy().flatten()
    
    # Sum attention weights per node (how much does this node influence others?)
    np.add.at(node_score, sources, weights)
    
    plt.figure(figsize=(10, 8))
    sc_plot = plt.scatter(adata.obsm['X_pca'][:,0], adata.obsm['X_pca'][:,1], 
                     c=node_score, cmap='inferno', s=15, alpha=0.9)
    plt.colorbar(sc_plot, label="Influence Score (Sum of Attention)")
    plt.title("GAT Attention Map:\nHighlighting Biological Driver Cells")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig("compare_5_gat_attention.png", dpi=200)
    plt.close()

def plot_error_distribution(scores_gcn, scores_gat):
    """
    Histogram of errors to see the spread.
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(scores_gcn, label='GCN', fill=True, color='steelblue', alpha=0.4)
    sns.kdeplot(scores_gat, label='GAT', fill=True, color='orange', alpha=0.4)
    plt.title("Distribution of Prediction Accuracy (Cosine Similarity)")
    plt.xlabel("Cosine Similarity (1.0 = Perfect)")
    plt.xlim(-1, 1)
    plt.legend()
    plt.savefig("compare_4_error_dist.png", dpi=150)
    plt.close()