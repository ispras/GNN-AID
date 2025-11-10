import copy

import torch

def get_adj_list(gen_dataset):
    """
    Get adjacency list from gen_dataset of GNN-AID format
    """
    if hasattr(gen_dataset, 'dataset'):
       gen_dataset = gen_dataset.dataset
    adj_list = {}
    for u, v in zip(gen_dataset.edge_index[0].tolist(), gen_dataset.edge_index[1].tolist()):
        if u in adj_list.keys():
            adj_list[u].append(v)
        else:
            adj_list[u] = [v]

    return adj_list

def from_adj_list(adj_list):
    """
    Get edge_index in COO-format from adjacency list
    """
    in_nodes = []
    out_nodes = []
    for n, edges in adj_list.items():
        for e in edges:
            in_nodes.append(n)
            out_nodes.append(e)
    return torch.tensor([in_nodes, out_nodes], dtype=torch.int)

def adj_list_oriented_to_non_oriented(adj_list):
    non_oriented_adj_list = {}
    for node, neighs in adj_list.items():
        if node not in non_oriented_adj_list.keys():
            non_oriented_adj_list[node] = copy.deepcopy(adj_list[node])
        else:
            non_oriented_adj_list[node] += copy.deepcopy(adj_list[node])
        for in_node in adj_list[node]:
            if in_node not in non_oriented_adj_list.keys():
                non_oriented_adj_list[in_node] = [node]
            else:
                non_oriented_adj_list[in_node].append(node)
    for k in non_oriented_adj_list.keys():
        non_oriented_adj_list[k] = list(set(non_oriented_adj_list[k]))
    return non_oriented_adj_list