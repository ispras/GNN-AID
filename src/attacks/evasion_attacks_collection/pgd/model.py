import torch
import torch.nn as nn
import torch.nn.functional as F


class MatGCNConv(nn.Module):
    """This is an implementation of GCNConv from pyg based on matrix operations"""
    def __init__(self, in_channels, out_channels):
        super(MatGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, adj):
        # x: node features, size (num_nodes, in_channels)
        # adj: adjacency matrix, size (num_nodes, num_nodes)

        # Normalization of the adjacency matrix
        adj_normalized = self.normalize_adjacency(adj)

        # Performing graph convolution
        support = torch.matmul(adj_normalized, x)
        out = self.linear(support)

        return out

    @staticmethod
    def normalize_adjacency(adj):
        # Converting Adjacency Matrix to Binary to Calculate Vertex Degree
        binary_adj = (adj > 0).float()

        # Calculating the degree of a vertex
        deg = torch.sum(binary_adj, dim=1)

        # Inverse square power
        deg_sqrt_inv = torch.pow(deg, -0.5)
        deg_sqrt_inv[torch.isinf(deg_sqrt_inv)] = 0.0

        #  TODO так как здесь не добавляется единичная матрица I к матрице A, то не учитываются self_loops.
        #   Поэтому эта свертка это аналог GCNConv из torch_geometric с параметром add_self_loops=False.
        #   Во всем остальном она работает абсолютно также. Можно добавить функционал добавления матрицы I

        # Normalization of the adjacency matrix: D^(-1/2) * A * D^(-1/2)
        normalized_adj = adj * deg_sqrt_inv.view(-1, 1) * deg_sqrt_inv.view(1, -1)
        return normalized_adj


class MatGNN(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes):
        super(MatGNN, self).__init__()

        # Initialize the layers
        self.conv0 = MatGCNConv(num_features, hidden)
        self.conv1 = MatGCNConv(hidden, num_classes)

    def forward(self, x=None, adj=None, **kwargs):
        x = self.conv0(x, adj)
        x = x.relu()
        x = self.conv1(x, adj)
        x = F.log_softmax(x, dim=1)
        return x
