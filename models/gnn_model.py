import torch
from torch_geometric.nn import GCNConv, global_mean_pool

class MolecularGNN(torch.nn.Module):

    def __init__(self, num_features: int = 38, hidden_dim: int = 128, embedding_dim: int = 64, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(p=dropout)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(num_features, hidden_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))

        # Output projection to embedding_dim
        self.convs.append(GCNConv(hidden_dim, embedding_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(embedding_dim))

    def forward(self, x, edge_index, batch):
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(self.convs) - 1:  # no activation on final layer
                x = torch.relu(x)
                x = self.dropout(x)  # why: جلوگیری از overfit حتی در inference تستی
        x = global_mean_pool(x, batch)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x