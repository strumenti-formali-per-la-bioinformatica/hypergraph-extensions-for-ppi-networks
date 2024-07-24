import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class Model(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, dropout = 0.125):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.gcn_conv_1 = GCNConv(in_channels, hidden_channels)
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.gcn_conv_2 = GCNConv(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.gcn_conv_3 = GCNConv(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, hidden_channels)

        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, X, edge_index, _):
        self.dropout(X)
        X = self.gcn_conv_1(X, edge_index) + self.lin1(X)
        X = nn.functional.leaky_relu(X)
        self.dropout(X)
        X = self.gcn_conv_2(X, edge_index) + self.lin2(X)
        X = nn.functional.leaky_relu(X)
        self.dropout(X)
        X = self.gcn_conv_3(X, edge_index) + self.lin3(X)
        X = nn.functional.leaky_relu(X)
        # Linear layer
        y = self.fc(X)
        return X, y
