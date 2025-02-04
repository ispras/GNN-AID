import torch.nn as nn

from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act='prelu', dropout=0.0, bias=True):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_ft, 2 * out_ft, bias=bias)
        self.conv2 = GCNConv(2 * out_ft, out_ft, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.PReLU() if act == 'prelu' else nn.ReLU() if act == 'relu' else nn.Identity()

    def forward(self, x, edge_index):
        """
        Forward pass of the GCN.

        Args:
            x (Tensor): Input feature matrix of shape [num_nodes, num_features].
            edge_index (Tensor): Edge indices of shape [2, num_edges].

        Returns:
            Tensor: Node embeddings of shape [num_nodes, out_ft].
        """
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)

        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = self.act(x)
        return x
