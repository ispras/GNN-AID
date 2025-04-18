import torch
from collections import defaultdict
from enum import Enum
from rdkit import Chem

""" The molecule code comes from https://github.com/Wuyxin/ReFine """


def e_map_mutag(bond_type, reverse=False):
    from rdkit import Chem
    if not reverse:
        if bond_type == Chem.BondType.SINGLE:
            return 0
        elif bond_type == Chem.BondType.DOUBLE:
            return 1
        elif bond_type == Chem.BondType.AROMATIC:
            return 2
        elif bond_type == Chem.BondType.TRIPLE:
            return 3
        else:
            raise Exception("No bond type found")

    if bond_type == 0:
        return Chem.BondType.SINGLE
    elif bond_type == 1:
        return Chem.BondType.DOUBLE
    elif bond_type == 2:
        return Chem.BondType.AROMATIC
    elif bond_type == 3:
        return Chem.BondType.TRIPLE
    else:
        raise Exception("No bond type found")


class x_map_mutag(Enum):
    C = 0
    N = 1
    O = 2
    F = 3
    I = 4
    Cl = 5
    Br = 6


def graph_to_mol(X, edge_index, edge_attr):
    mol = Chem.RWMol()
    X = [
        Chem.Atom(x_map_mutag(x.index(1)).name)
        for x in X
    ]

    E = edge_index
    for x in X:
        print(x)
        mol.AddAtom(x)

    for atom in mol.GetAtoms():
        atom.SetProp('atomLabel', atom.GetSymbol())

    if edge_attr is None:
        edge_attr = [[1, 0, 0, 0] for _ in range(len(edge_index))]

    for (u, v), attr in zip(E, edge_attr):
        attr = e_map_mutag(attr.index(1), reverse=True)

        if mol.GetBondBetweenAtoms(u, v):
            continue
        mol.AddBond(u, v, attr)
    return mol


def edge_index_to_adj_list(edge_index, hops=1, exclude_self=False):
    adj = defaultdict(list)
    if edge_index.shape[1] == 0:
        return adj
    if hops == 1:
        for i in range(edge_index.shape[1]):
            adj[edge_index[0][i].item()].append(edge_index[1][i].item())
    else:
        A = edge_index_to_adj_matrix(edge_index)
        A_ = A
        B = A
        for _ in range(1, hops):
            A = A @ A_
            B = B + A  # do not use +=
        A = B
        for i in range(edge_index.shape[1]):
            u = edge_index[0][i].item()
            adj[u] = list(A[u].nonzero().squeeze(1).cpu().detach().numpy())
    if exclude_self and hops > 1:
        for v, k in adj.items():
            k.remove(v)
    return adj


def edge_index_to_tuples(edge_index):
    return [(pair[0].item(), pair[1].item()) for pair in edge_index.T]


def edge_index_to_adj_matrix(edge_index, x=None):
    num_nodes = x.shape[0] if x is not None else torch.unique(edge_index).max().item() + 1

    A = torch.zeros((num_nodes, num_nodes))
    for i in range(edge_index.shape[1]):
        A[edge_index[0][i].item(), edge_index[1][i].item()] = 1
    return A


def k_hop_neighbourhood(edge_index, node, k):
    adj = edge_index_to_adj_list(edge_index)
    visited = set()
    ret = []

    queue = [(node, 0)]
    while queue != []:
        u, d = queue.pop(-1)
        if d == k:
            continue
        for v in adj[u]:
            if v not in visited:
                queue.insert(0, (v, d + 1))
                ret.append(v)
                visited.add(v)

    return ret


def add_edge(g, u, v):
    if (u, v) in edge_index_to_tuples(g.edge_index):
        return
    g.edge_index = torch.column_stack([g.edge_index, torch.tensor([[u, v], [v, u]])])


def remove_edge(g, u, v):
    num_edges = g.edge_index.shape[1]
    inds1 = torch.logical_and(g.edge_index[0, :] == torch.LongTensor([u] * num_edges),
                              g.edge_index[1, :] == torch.LongTensor([v] * num_edges))
    inds2 = torch.logical_and(g.edge_index[0, :] == torch.LongTensor([v] * num_edges),
                              g.edge_index[1, :] == torch.LongTensor([u] * num_edges))
    inds = torch.logical_not(torch.logical_or(inds1, inds2)).nonzero().squeeze(1)
    g.edge_index = g.edge_index.index_select(1, inds)
