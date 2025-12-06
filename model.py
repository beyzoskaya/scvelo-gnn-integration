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
        
        # GCN has no attention weights
        if return_attention:
            return out, None
        return out

class VelocityGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super(VelocityGAT, self).__init__()
        
        # --- ENCODER (GAT) ---
        # Heads=4 means we learn 4 different perspectives
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        
        # Layer 2
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # --- DECODER ---
        self.predictor = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, return_attention=False):
        # Layer 1
        # Fix: GATv2Conv returns just 'x' if return_attention_weights is False (default)
        x = self.conv1(x, edge_index) 
        x = self.bn1(x)
        x = F.elu(x) 
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Layer 2
        if return_attention:
            # Here we request weights explicitly
            x, (att_edge_index, att_weights) = self.conv2(x, edge_index, return_attention_weights=True)
        else:
            x = self.conv2(x, edge_index)
            att_edge_index, att_weights = None, None

        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        out = self.predictor(x)
        
        if return_attention:
            return out, (att_edge_index, att_weights)
        return out

class GenerativeVelocityGNN(nn.Module):
    def __init__(self, input_dim=60, hidden_dim=128, output_genes=2000, heads=4):
        super(GenerativeVelocityGNN, self).__init__()
        
        # --- ENCODER (GAT) ---
        # Input is 60 dims (30 PCA + 30 Velocity)
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        
        # Layer 2
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # --- DECODER (Generative) ---
        # Expands latent state back to 2000 Genes
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, output_genes) # Output: Reconstructed Genes
        )

    def forward(self, x_pca, v_pca, edge_index):
        # 1. Combine State + Velocity
        x_in = torch.cat([x_pca, v_pca], dim=1) 
        
        # 2. Graph Encoding
        # Fix: Do NOT unpack (_, _) because we are not asking for weights
        h = self.conv1(x_in, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)
        
        # 3. Generative Decoding
        gene_expression_pred = self.decoder(h)
        
        return gene_expression_pred