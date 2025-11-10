import torch
import numpy as np
from torch_geometric.utils.num_nodes import maybe_num_nodes


def k_hop_subgraph_with_default_whole_graph(node_idx, num_hops, edge_index,
                                            remap_edges=False, num_nodes=None,
                                            flow='source_to_target'):

    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`. If the node_idx is -1, then return the original graph.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`. when num_hops == -1,
            the whole graph will be returned.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index  # edge_index 0 to 1, col: source, row: target

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    inv = None

    if int(node_idx) == -1:
        subsets = torch.tensor([0])
        cur_subsets = subsets
        while 1:
            node_mask.fill_(False)
            node_mask[subsets] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets = torch.cat([subsets, col[edge_mask]]).unique()
            if not cur_subsets.equal(subsets):
                cur_subsets = subsets
            else:
                subset = subsets
                break
    else:
        if isinstance(node_idx, (int, list, tuple)):
            node_idx = torch.tensor([node_idx], device=row.device, dtype=torch.int64).flatten()
        elif isinstance(node_idx, torch.Tensor) and len(node_idx.shape) == 0:
            node_idx = torch.tensor([node_idx])
        else:
            node_idx = node_idx.to(row.device)

        subsets = [node_idx]
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    edge_index_original_index = edge_index
    if remap_edges:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, edge_index_original_index, inv, edge_mask


def get_topk_edges_subgraph(edge_index, edge_mask, top_k, un_directed=False):
    if un_directed:
        top_k = 2 * top_k
    edge_mask = edge_mask.reshape(-1)
    thres_index = max(edge_mask.shape[0] - top_k, 0)
    threshold = float(edge_mask.reshape(-1).sort().values[thres_index])
    hard_edge_mask = (edge_mask >= threshold)
    selected_edge_idx = np.where(hard_edge_mask == 1)[0].tolist()
    edgelist = []
    important_edges = {}
    for edge_idx in selected_edge_idx:
        edge = edge_index[:, edge_idx].tolist()
        edgelist.append((edge[0], edge[1]))
        imp = edge_mask[edge_idx]
        # important_edges[f"{edge[0]},{edge[1]}"] = format(imp, '.4f')
        important_edges[f"{edge[0]},{edge[1]}"] = 1
    return edgelist, important_edges
