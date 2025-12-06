import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

class VelocityGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VelocityGNN, self).__init__()
        
        # --- ENCODER ---
        # Layer 1: Aggregate info from immediate neighbors
        # We use GCNConv which is great for smoothing features over the graph
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Layer 2: Deeper aggregation (neighbors of neighbors)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # --- DECODER (Predictor) ---
        # A simple linear layer to map the hidden latent state 
        # back to the Velocity Vector dimension
        self.predictor = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr=None):
        # 1. Message Passing Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x) # ELU is often better for dynamics than ReLU
        x = F.dropout(x, p=0.3, training=self.training)
        
        # 2. Message Passing Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # 3. Predict Velocity Vector
        out = self.predictor(x)
        
        return out