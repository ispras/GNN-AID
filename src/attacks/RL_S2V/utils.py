import torch
from torch_geometric.utils import add_self_loops, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Optional

def normalize_sparse_tensor(
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        fill_value: int = 1
        ):
    """Normalize sparse tensor. Need to import torch_scatter
    """
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index)

    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    assert isinstance(edge_weight, torch.Tensor)
    edge_index, edge_weight = add_self_loops(edge_index, -edge_weight, fill_value=1., num_nodes=num_nodes)
    return edge_index, edge_weight


def degree_normalize_sparse_tensor(
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        fill_value: int = 1
        ):
    """degree_normalize_sparse_tensor.
    """
    # TODO check if not bug
    if edge_index is None:
        return None
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index)

    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight

    # L = I - A_norm.
    assert isinstance(edge_weight, torch.Tensor)
    edge_index, edge_weight = add_self_loops(edge_index, -edge_weight, fill_value=1., num_nodes=num_nodes)

    return edge_index, edge_weight


import torch
from torch_scatter import scatter_add


def sum_coo_tensors(edge_index1, edge_weight1, edge_index2, edge_weight2, num_nodes):
    """
    Sum two COO sparse tensors given as (edge_index, edge_weight) pairs,
    handling cases where one of the edge weights is None.

    Args:
        edge_index1 (torch.Tensor): Shape (2, E1), indices of the first sparse tensor.
        edge_weight1 (torch.Tensor or None): Shape (E1,), values of the first sparse tensor.
        edge_index2 (torch.Tensor): Shape (2, E2), indices of the second sparse tensor.
        edge_weight2 (torch.Tensor or None): Shape (E2,), values of the second sparse tensor.
        num_nodes (int): Number of nodes in the graph.

    Returns:
        edge_index (torch.Tensor): Merged edge indices.
        edge_weight (torch.Tensor): Summed edge weights.
    """
    # Assign zero-filled weights if None
    if edge_weight1 is None:
        edge_weight1 = torch.zeros(edge_index1.shape[1], device=edge_index1.device)
    if edge_weight2 is None:
        edge_weight2 = torch.zeros(edge_index2.shape[1], device=edge_index2.device)

    # Step 1: Concatenate edges and weights
    edge_index = torch.cat([edge_index1, edge_index2], dim=1)  # Shape (2, E1+E2)
    edge_weight = torch.cat([edge_weight1, edge_weight2], dim=0)  # Shape (E1+E2,)

    # Step 2: Convert edge indices to a unique 1D index for aggregation
    unique_index = edge_index[0] * num_nodes + edge_index[1]  # Flattened index for uniqueness

    # Step 3: Aggregate (sum) duplicate edges using scatter_add
    unique_index, perm = unique_index.sort()  # Sort to group duplicates
    edge_index = edge_index[:, perm]  # Sort edge_index accordingly
    edge_weight = edge_weight[perm]  # Sort edge_weight accordingly

    summed_weight = scatter_add(edge_weight, unique_index, dim=0, dim_size=num_nodes * num_nodes)

    # Step 4: Extract only nonzero elements (valid edges)
    mask = summed_weight.nonzero(as_tuple=True)[0]
    edge_weight = summed_weight[mask]
    edge_index = torch.stack([mask // num_nodes, mask % num_nodes], dim=0)

    return edge_index, edge_weight

def norm_adj(
        edge_index,
        edge_weight: Optional[torch.Tensor] = None,
        gm: str = 'gcn',
        device: torch.device = torch.device('cpu')
):
    if gm == 'gcn':
        normed_edge_index, normed_edge_weight = normalize_sparse_tensor(edge_index, edge_weight)
    else:
        normed_edge_index, normed_edge_weight = degree_normalize_sparse_tensor(edge_index, edge_weight)

    return normed_edge_index, normed_edge_weight

# def norm_extra(
#         edge_index,
#         edge_weight: Optional[torch.Tensor] = None,
#         gm: str = 'gcn',
#         device: torch.device = torch.device('cpu')
# ):
#     pass

def edge_index_to_dict_of_lists(edge_index):
    edge_dict = {}
    for src, tgt in edge_index.t().tolist():
        if src not in edge_dict:
            edge_dict[src] = []
        edge_dict[src].append(tgt)

    return edge_dict