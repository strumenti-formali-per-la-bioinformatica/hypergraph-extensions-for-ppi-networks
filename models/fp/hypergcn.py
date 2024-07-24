import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv, HypergraphConv

class Model(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, dropout = 0.125):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.gcn_lin1 = torch.nn.Linear(in_channels, hidden_channels)

        self.hypergraph_conv_1 = HypergraphConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.hypergraph_conv_2 = HypergraphConv(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, X, edge_index, hyperedge_index):

        self.dropout(X)
        X = self.conv1(X, edge_index) + self.gcn_lin1(X)
        X = nn.functional.leaky_relu(X)

        self.dropout(X)
        X = self.hypergraph_conv_1(X, hyperedge_index) + self.lin1(X)
        X = nn.functional.leaky_relu(X)

        self.dropout(X)
        X = self.hypergraph_conv_2(X, hyperedge_index) + self.lin2(X)
        X = nn.functional.leaky_relu(X)

        # Linear layer
        y = self.fc(X)
        return X, y
