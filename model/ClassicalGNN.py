"""Three-layer GNN alternatives used by the PrivIM appendix experiments."""

import torch
import torch.nn as nn
from dgl.nn import GATConv, GraphConv, SAGEConv, GINConv


class _ProbabilisticModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        # Keep the historical ``grat`` attribute because evaluation uses the
        # pre-sigmoid scores for top-k selection.
        self.grat = backbone

    def forward(self, graph, features):
        return torch.sigmoid(self.logits(graph, features))

    def logits(self, graph, features):
        return self.grat(graph, features)


class _GCN3(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphConv(input_dim, hidden1, allow_zero_in_degree=True),
            GraphConv(hidden1, hidden2, allow_zero_in_degree=True),
            GraphConv(hidden2, 1, allow_zero_in_degree=True),
        ])

    def forward(self, graph, features):
        value = torch.relu(self.layers[0](graph, features))
        value = torch.relu(self.layers[1](graph, value))
        return self.layers[2](graph, value)


class _GAT3(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2):
        super().__init__()
        self.layers = nn.ModuleList([
            GATConv(input_dim, hidden1, 1, allow_zero_in_degree=True),
            GATConv(hidden1, hidden2, 1, allow_zero_in_degree=True),
            GATConv(hidden2, 1, 1, allow_zero_in_degree=True),
        ])

    def forward(self, graph, features):
        value = torch.relu(self.layers[0](graph, features).squeeze(1))
        value = torch.relu(self.layers[1](graph, value).squeeze(1))
        return self.layers[2](graph, value).squeeze(1)


class _GIN3(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2):
        super().__init__()
        self.layers = nn.ModuleList([
            GINConv(nn.Linear(input_dim, hidden1), "sum"),
            GINConv(nn.Linear(hidden1, hidden2), "sum"),
            GINConv(nn.Linear(hidden2, 1), "sum"),
        ])

    def forward(self, graph, features):
        value = torch.relu(self.layers[0](graph, features))
        value = torch.relu(self.layers[1](graph, value))
        return self.layers[2](graph, value)


class _GraphSAGE3(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2):
        super().__init__()
        self.layers = nn.ModuleList([
            SAGEConv(input_dim, hidden1, "mean"),
            SAGEConv(hidden1, hidden2, "mean"),
            SAGEConv(hidden2, 1, "mean"),
        ])

    def forward(self, graph, features):
        value = torch.relu(self.layers[0](graph, features))
        value = torch.relu(self.layers[1](graph, value))
        return self.layers[2](graph, value)


def GCN3(input_dim, hidden1, hidden2, *args):
    return _ProbabilisticModel(_GCN3(input_dim, hidden1, hidden2))


def GAT3(input_dim, hidden1, hidden2, *args):
    return _ProbabilisticModel(_GAT3(input_dim, hidden1, hidden2))


def GIN3(input_dim, hidden1, hidden2, *args):
    return _ProbabilisticModel(_GIN3(input_dim, hidden1, hidden2))


def GraphSAGE3(input_dim, hidden1, hidden2, *args):
    return _ProbabilisticModel(_GraphSAGE3(input_dim, hidden1, hidden2))
