import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

class VelocityGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VelocityGNN, self).__init__()
        
        # --- ENCODER (GCN) ---
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Layer 2
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # --- DECODER ---
        self.predictor = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, return_attention=False):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        out = self.predictor(x)
        
        # GCN has no attention weights, return None to keep API consistent
        if return_attention:
            return out, None
        return out

class VelocityGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super(VelocityGAT, self).__init__()
        
        # --- ENCODER (GAT) ---
        # GATv2Conv calculates "Attention Weights"
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        
        # Layer 2
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # --- DECODER ---
        self.predictor = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, return_attention=False):
        # Layer 1
        x, _ = self.conv1(x, edge_index, return_attention_weights=False)
        x = self.bn1(x)
        x = F.elu(x) 
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Layer 2
        # We capture attention weights here
        x, (att_edge_index, att_weights) = self.conv2(x, edge_index, return_attention_weights=True)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        out = self.predictor(x)
        
        if return_attention:
            return out, (att_edge_index, att_weights)
        return out