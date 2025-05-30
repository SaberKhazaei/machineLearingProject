import torch
from torch_geometric.nn import GCNConv, global_mean_pool

class MolecularGNN(torch.nn.Module):
    """Graph Neural Network for molecular representation learning"""
    
    def __init__(self, num_features=78, hidden_dim=128, embedding_dim=64, num_layers=3):
        super(MolecularGNN, self).__init__()
        
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(num_features, hidden_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, embedding_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(embedding_dim))
        
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch):
        # Apply GCN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(self.convs) - 1:  # No activation on last layer
                x = torch.relu(x)
                x = self.dropout(x)
        
        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        
        # L2 normalize embeddings
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        
        return x
