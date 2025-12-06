import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from data_loader_step1 import get_velocity_data
from model import VelocityGNN

device = torch.device("cpu")
print(f"Using device: {device}")

def direction_loss(pred, target):

    loss_mse = nn.MSELoss()(pred, target)
    cos_sim = torch.nn.functional.cosine_similarity(pred, target, dim=1)
    loss_cos = (1 - cos_sim).mean()

    return loss_mse + (1.5 * loss_cos)

print("--- Loading Data ---")
data, adata = get_velocity_data()
data = data.to(device)

input_dim = data.x.shape[1] # PCA dimensions
output_dim = data.v.shape[1] # Velocity vector dimensions

model = VelocityGNN(input_dim=input_dim, hidden_dim=128, output_dim=output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

print("\n--- Starting Training ---")
losses = []

model.train()
epochs = 200

for epoch in range(epochs):
    optimizer.zero_grad()
    v_pred = model(data.x, data.edge_index)
    loss = direction_loss(v_pred[data.train_mask], data.v[data.train_mask])
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("\n--- Generating Comparison Plot ---")
model.eval()
with torch.no_grad():
    v_pred = model(data.x, data.edge_index).cpu().numpy()

X = data.x.cpu().numpy()
V_true = data.v.cpu().numpy()

fig, ax = plt.subplots(1, 2, figsize=(18, 8))
idx = np.random.choice(data.num_nodes, 800, replace=False)

# PLOT 1: Ground Truth (What scVelo calculated)
ax[0].scatter(X[:,0], X[:,1], s=10, color='lightgrey', alpha=0.5)
ax[0].quiver(X[idx, 0], X[idx, 1], 
             V_true[idx, 0], V_true[idx, 1],
             color='black', scale=3, scale_units='xy', alpha=0.7)
ax[0].set_title("Ground Truth (scVelo)\nRaw noisy vectors")
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")

# PLOT 2: GNN Prediction (What the Network Learned)
ax[1].scatter(X[:,0], X[:,1], s=10, color='lightgrey', alpha=0.5)
ax[1].quiver(X[idx, 0], X[idx, 1], 
             v_pred_all[idx, 0], v_pred_all[idx, 1],
             color='teal', scale=3, scale_units='xy', alpha=0.9) # Teal for prediction
ax[1].set_title("GNN Predicted Dynamics\nLearned Vector Field")
ax[1].set_xlabel("PC1")
ax[1].set_ylabel("PC2")

plt.tight_layout()
plt.savefig("gnn_velocity_comparison.png", dpi=200)
print("-> Saved 'gnn_velocity_comparison.png'")

# Plot Loss
plt.figure(figsize=(6,4))
plt.plot(losses)
plt.title("Training Loss (MSE + Cosine)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("training_loss.png")