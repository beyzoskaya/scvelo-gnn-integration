import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import numpy as np
import visuals
from data_loader_step1 import get_velocity_data
from model import VelocityGNN, VelocityGAT

DATASET = 'dentate'
EPOCHS = 150
LR = 0.005
DEVICE = torch.device('cpu')

def train_model(model, data, name="Model"):
    print(f"\n--- Training {name} ---")
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    mse_crit = nn.MSELoss()
    
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        # Loss: Direction + Magnitude
        loss_mse = mse_crit(out[data.train_mask], data.v[data.train_mask])
        cos_sim = torch.nn.functional.cosine_similarity(out[data.train_mask], data.v[data.train_mask], dim=1)
        loss_cos = (1.0 - cos_sim).mean()
        loss = loss_mse + (1.5 * loss_cos)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"   Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            
    return model

def get_metrics(model, data):
    model.eval()
    with torch.no_grad():
        # Get raw output (handle tuple return for GAT)
        res = model(data.x, data.edge_index, return_attention=True)
        if isinstance(res, tuple):
            pred, attention_data = res
        else:
            pred, attention_data = res, None
            
        # Calculate Cosine Sim per cell
        v_true = data.v.cpu().numpy()
        pred_np = pred.cpu().numpy()
        
        norms_p = np.linalg.norm(pred_np, axis=1)
        norms_t = np.linalg.norm(v_true, axis=1)
        norms_p[norms_p==0] = 1e-8
        norms_t[norms_t==0] = 1e-8
        
        cosine_scores = np.sum(pred_np * v_true, axis=1) / (norms_p * norms_t)
        
    return pred_np, cosine_scores, attention_data

def main():
    print("=== SCVELO GNN vs GAT COMPARISON ===")
    
    # 1. Load Data
    data, adata = get_velocity_data(DATASET)
    data = data.to(DEVICE)
    
    # 2. Train GCN
    gcn = VelocityGNN(input_dim=30, hidden_dim=128, output_dim=30)
    gcn = train_model(gcn, data, name="VelocityGCN")
    pred_gcn, scores_gcn, _ = get_metrics(gcn, data)
    print(f"   -> GCN Mean Accuracy: {np.mean(scores_gcn):.4f}")
    
    # 3. Train GAT
    gat = VelocityGAT(input_dim=30, hidden_dim=64, output_dim=30, heads=4)
    gat = train_model(gat, data, name="VelocityGAT")
    pred_gat, scores_gat, att_data = get_metrics(gat, data)
    print(f"   -> GAT Mean Accuracy: {np.mean(scores_gat):.4f}")
    
    # 4. Generate Comparison Plots
    print("\n--- Generating Comparison Reports ---")
    
    # A. Boxplots (The Head-to-Head)
    visuals.plot_comparison_boxplots(adata, scores_gcn, scores_gat)
    
    # B. Streamlines (The Visual Flow)
    visuals.plot_comparison_streamlines(adata, pred_gcn, pred_gat)
    
    # C. Error Distribution (Who is more robust?)
    visuals.plot_error_distribution(scores_gcn, scores_gat)
    
    # D. Attention Map (The Biological Insight - GAT only)
    if att_data:
        att_edge_index, att_weights = att_data
        visuals.plot_attention_map(adata, att_weights, att_edge_index)
        
    print("\n=== COMPARISON COMPLETE ===")
    print("Files Generated:")
    print(" 1. compare_2_cluster_performance.png (Boxplot)")
    print(" 2. compare_3_streamlines.png         (Flow Visuals)")
    print(" 3. compare_4_error_dist.png          (Error Histogram)")
    print(" 4. compare_5_gat_attention.png       (Biological Drivers)")

if __name__ == "__main__":
    main()