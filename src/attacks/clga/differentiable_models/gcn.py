import torch.nn as nn
import torch

from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(
            self,
            in_ft: int,
            out_ft: int,
            act: str = 'prelu',
            dropout: float = 0.0,
            bias: bool = True
    ):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_ft, 2 * out_ft, bias=bias)
        self.conv2 = GCNConv(2 * out_ft, out_ft, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.PReLU() if act == 'prelu' else nn.ReLU() if act == 'relu' else nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the GCN.

        :param x: Input feature matrix of shape [num_nodes, num_features].
        :param edge_index: Edge indices of shape [2, num_edges].
        :returns: Node embeddings of shape [num_nodes, out_ft].

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
