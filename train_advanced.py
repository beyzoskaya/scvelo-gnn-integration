import os
# --- MAC OS STABILITY FIXES ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np

from data_loader_step1 import get_velocity_data
from model import VelocityGNN
import visuals
import simulate 

def direction_loss(pred, target, mse_crit, alpha=1.5):
    loss_mse = mse_crit(pred, target)
    cos_sim = torch.nn.functional.cosine_similarity(pred, target, dim=1)
    loss_cos = (1.0 - cos_sim).mean()
    return loss_mse + (alpha * loss_cos)

def main():
    device = torch.device('cpu') 
    print("--- 1. Initializing Environment ---")

    dataset_choice = 'dentate'
    data, adata = get_velocity_data(dataset_choice)
    data = data.to(device)

    visuals.plot_graph_connectivity(data, adata)

    print("\n--- 2. Building Model ---")
    model = VelocityGNN(input_dim=30, hidden_dim=128, output_dim=30).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    mse_crit = nn.MSELoss()

    print("\n--- 3. Starting Training ---")
    epochs = 120 

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        v_pred = model(data.x, data.edge_index)
        # The Output: A "Denoised Vector Field." For any cell (or any point in the latent space), the model can output a vector indicating the direction of differentiation
        loss = direction_loss(v_pred[data.train_mask], data.v[data.train_mask], mse_crit)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    print("\n--- 4. Evaluation & Visualization ---")
    model.eval()
    with torch.no_grad():
        final_pred = model(data.x, data.edge_index).cpu().numpy()

        v_true = data.v.cpu().numpy()
        norms_p = np.linalg.norm(final_pred, axis=1)
        norms_t = np.linalg.norm(v_true, axis=1)
        norms_p[norms_p==0] = 1e-8
        norms_t[norms_t==0] = 1e-8
        cosine_scores = np.sum(final_pred * v_true, axis=1) / (norms_p * norms_t)

    visuals.plot_per_cluster_performance(adata, cosine_scores)
    visuals.plot_streamlines(adata, final_pred)

    simulate.run_simulation(model, data, adata)

    print("\n=== PROJECT COMPLETE ===")
    print("Dataset Used:", dataset_choice)
    print("Check your folder for:")
    print(" - visual_1_graph_topology.png")
    print(" - visual_2_cluster_performance.png")
    print(" - visual_3_streamlines.png")
    print(" - visual_4_simulation.png")

if __name__ == "__main__":
    main()