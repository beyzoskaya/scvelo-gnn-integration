import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import numpy as np
from data_loader_step1 import get_velocity_data
from model import GenerativeVelocityGNN
import visuals

def main():
    print("=== GENERATIVE GENE RECONSTRUCTION START ===")
    device = torch.device('cpu')
    
    # 1. Load Data
    # Change to 'pancreas' or 'dentate'
    data, adata = get_velocity_data('dentate') 
    data = data.to(device)
    
    print("\n" + "="*40)
    print("   DATA INSPECTION (Example Rows)")
    print("="*40)
    
    print(f"1. Input Feature Shape (PCA): {data.x.shape}")
    print(f"   -> Row 0 (First 5 vals): {data.x[0, :5].numpy()}")
    
    print(f"2. Velocity Input Shape:    {data.v.shape}")
    print(f"   -> Row 0 (First 5 vals): {data.v[0, :5].numpy()}")
    
    print(f"3. Target Genes Shape:      {data.y_genes.shape}")
    print(f"   -> Row 0 (First 5 vals): {data.y_genes[0, :5].numpy()}")
    print("-" * 40 + "\n")

    # 2. Initialize Generative Model
    # Input dim = 30 (PCA) + 30 (Velocity) = 60
    # Output dim = number of genes in adata
    num_genes = data.y_genes.shape[1]
    
    model = GenerativeVelocityGNN(input_dim=60, hidden_dim=128, output_genes=num_genes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    criterion = nn.MSELoss() # Reconstruction Loss

    # 3. Training Loop
    print(f"--- Starting Generative Training (Predicting {num_genes} Genes) ---")
    epochs = 100
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward Pass: Give it PCA + Velocity -> Predict Genes
        pred_genes = model(data.x, data.v, data.edge_index)
        
        # Loss: Compare Predicted Genes vs True Genes
        loss = criterion(pred_genes[data.train_mask], data.y_genes[data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"   Epoch {epoch:03d} | Reconstruction Loss: {loss.item():.4f}")

    # 4. Evaluation
    print("\n--- Evaluation ---")
    model.eval()
    with torch.no_grad():
        final_pred = model(data.x, data.v, data.edge_index)
        
        test_flat_true = data.y_genes[data.val_mask].flatten()
        test_flat_pred = final_pred[data.val_mask].flatten()
        
        print("   Sample True Genes (Test Set):", test_flat_true[:5].numpy())
        print("   Sample Predicted Genes (Test Set):", test_flat_pred[:5].numpy())

        corr = torch.corrcoef(torch.stack([test_flat_true, test_flat_pred]))[0, 1]
        print(f"   Global Pearson Correlation (Test Set): {corr.item():.4f}")

    # 5. Visualization
    gene_names = adata.var_names.tolist()
    #visuals.plot_gene_reconstruction(data.y_genes, final_pred, gene_names)
    visuals.plot_gene_spatial_comparison(adata, data.y_genes, final_pred, gene_names)
    
    print("\n=== GENERATIVE TASK COMPLETE ===")
    print("Check 'visual_6_gene_reconstruction.png'")

if __name__ == "__main__":
    main()