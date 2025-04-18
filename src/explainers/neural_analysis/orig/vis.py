import io
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric
from PIL import Image
from pyvis.network import Network
from rdkit.Chem.Draw import rdMolDraw2D
from pathlib import Path

from explainers.neural_analysis.orig.graph_utils import graph_to_mol


def create_dir(file_path):
    Path(file_path[:file_path.rindex('/')]).mkdir(parents=True, exist_ok=True)


def visualise_graph(graph, special=None, labs=None, mask=None, layout='spring'):
    plt.figure(figsize=(10, 10), dpi=50)

    G = torch_geometric.utils.to_networkx(graph)

    color_map = None

    if graph.x.shape[1] == 7:
        node_colors = torch.argmax(graph.x, axis=1).detach().numpy()
        color_map = []
        labs = []
        for i, c in enumerate(node_colors):
            color_map.append(['red', 'orange', 'green', 'yellow', 'blue', 'purple', 'pink'][c])
            labs.append(['C', 'N', 'O', 'F', 'I', 'Cl', 'Br'][c])
            labs[-1] += str(i)
            if special is not None and i == special:
                color_map[-1] = 'pink'
    elif graph.x.shape[1] == 14:
        node_colors = torch.argmax(graph.x, axis=1).detach().numpy()
        color_map = []
        labs = []
        for i, c in enumerate(node_colors):
            color_map.append(
                ['red', 'orange', 'green', 'yellow', 'blue', 'purple', 'pink', 'violet', 'cyan', 'tan', 'silver',
                 'salmon', 'orchid', 'crimson'][c])
            labs.append(['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca'][c])
            labs[-1] += ' ' + str(i)
            if special is not None and i == special:
                color_map[-1] = 'pink'
    elif graph.x.shape[1] == 4:
        node_colors = torch.argmax(graph.x, axis=1).detach().numpy()
        color_map = []
        for i, c in enumerate(node_colors):
            color_map.append(['grey', 'violet', 'cyan', 'gold'][c])
            if special is not None and i == special:
                color_map[-1] = 'pink'
    elif graph.x.shape[1] == 3:
        node_colors = torch.argmax(graph.x, axis=1).detach().numpy()
        color_map = []
        for i, c in enumerate(node_colors):
            color_map.append(['darkturquoise', 'orchid', 'darkorange'][c])
            if special is not None and i == special:
                color_map[-1] = 'pink'

    if labs is None:
        labs = np.arange(graph.x.shape[0])
        labs = [str(lab) for lab in labs]
        for i in range(graph.x.shape[0]):
            labs[i] += ' ' + str(i)

    if color_map is None:
        color_map = ['red'] * graph.x.shape[0]

    if mask is not None:
        for i, b in enumerate(mask):
            if not b:
                color_map[i] = 'grey'

    net = Network(notebook=True)
    net.from_nx(G)

    for i, n in enumerate(net.nodes):
        n['color'] = color_map[n['id']]
    net.show_buttons(filter_=["physics"])
    net.show('../graphs/example.html')

    dic = {}

    if labs:
        assert (graph.x.shape[0] == len(labs))
        dic = {}
        for i, s in enumerate(labs):
            dic[i] = s

    if layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == 'spring':
        pos = nx.spring_layout(G, seed=111)

    nx.draw(G, pos=pos, alpha=.8, arrows=False, labels=dic, node_color=color_map)


def show_graph(graph, edge_values, special=None, labs=None, node_values=None, show_labels=True, anchor=None,
               custom_name=None, as_molecule=False):
    # print(edge_values)
    # edge_values[np.where(edge_values > 0.001)] = 100.0

    if graph.x.shape[1] == 7:
        node_colors = torch.argmax(graph.x, axis=1).detach().numpy()
        color_map = []
        labs = []
        for i, c in enumerate(node_colors):
            color_map.append(['red', 'orange', 'green', 'yellow', 'blue', 'purple', 'pink'][c])
            if special is not None and i == special:
                color_map[-1] = 'pink'
            labs.append(['C', 'N', 'O', 'F', 'I', 'Cl', 'Br'][c])
            labs[-1] += str(i)
    elif graph.x.shape[1] == 14:
        node_colors = torch.argmax(graph.x, axis=1).detach().numpy()
        color_map = []
        labs = []
        for i, c in enumerate(node_colors):
            color_map.append(
                ['red', 'orange', 'green', 'yellow', 'blue', 'purple', 'pink', 'violet', 'cyan', 'tan', 'silver',
                 'salmon', 'orchid', 'crimson'][c])
            labs.append(['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca'][c])
            labs[-1] += ' ' + str(i)
            if special is not None and i == special:
                color_map[-1] = 'pink'
    elif graph.x.shape[1] == 4:
        node_colors = torch.argmax(graph.x, axis=1).detach().numpy()
        color_map = []
        for i, c in enumerate(node_colors):
            color_map.append(['black', 'red', 'blue', 'pink'][c])
            if special is not None and i == special:
                color_map[-1] = 'pink'
    elif graph.x.shape[1] == 3:
        node_colors = torch.argmax(graph.x, axis=1).detach().numpy()
        color_map = []
        for i, c in enumerate(node_colors):
            color_map.append(['darkturquoise', 'orchid', 'darkorange'][c])
            if special is not None and i == special:
                color_map[-1] = 'pink'
    else:
        color_map = ['cyan'] * graph.x.shape[0]
        if special is not None:
            color_map[special] = 'pink'

    if labs is None:
        labs = np.arange(graph.x.shape[0])
        labs = [str(lab) for lab in labs]
        for i in range(graph.x.shape[0]):
            labs[i] += ' ' + str(i)

    if as_molecule:
        g = graph

        idx = np.arange(g.edge_index.shape[1])
        edge_vals = edge_values
        node_vals = node_values

        G = nx.DiGraph()
        G.add_nodes_from(range(g.num_nodes))
        G.add_edges_from(list(g.edge_index.cpu().numpy().T))

        x = g.x.detach().cpu().tolist()
        edge_index = g.edge_index.T.detach().cpu().tolist()
        edge_attr = g.edge_attr.detach().cpu().tolist() if g.edge_attr is not None else None
        mol = graph_to_mol(x, edge_index, edge_attr)
        d = rdMolDraw2D.MolDraw2DCairo(500, 500)
        hit_at = np.unique(g.edge_index[:, idx].detach().cpu().numpy()).tolist()
        hit_bonds = []

        if edge_values is not None:
            norm = matplotlib.colors.Normalize(vmin=0, vmax=max(edge_values) if anchor is None else anchor, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('plasma'))
            bond_colors = {}
            atom_colors = {}

            for edge_idx in idx:
                u, v = g.edge_index.T[edge_idx]
                edge_val = edge_vals[edge_idx]

                bond_idx = mol.GetBondBetweenAtoms(int(u), int(v)).GetIdx()
                hit_bonds.append(bond_idx)
                rgba = mapper.to_rgba(edge_val)
                bond_colors[bond_idx] = (rgba[0], rgba[1], rgba[2], 0.7)

            for idx, atom in enumerate(hit_at):
                if node_vals is not None:
                    node_val = node_vals[atom]
                    rgba = mapper.to_rgba(node_val)
                    atom_colors[idx] = (rgba[0], rgba[1], rgba[2], 0.7)
                else:
                    atom_colors[idx] = (1, 1, 1, 0)

            rdMolDraw2D.PrepareAndDrawMolecule(
                d, mol, highlightAtoms=hit_at, highlightBonds=hit_bonds,
                highlightAtomColors=atom_colors,
                highlightBondColors=bond_colors)
        else:
            rdMolDraw2D.PrepareAndDrawMolecule(
                d, mol)

        d.FinishDrawing()

        bindata = d.GetDrawingText()
        iobuf = io.BytesIO(bindata)
        image = Image.open(iobuf)
        file_path = f'../figs/{custom_name}' if custom_name else '../figs/mols/mol.png'
        create_dir(file_path)
        image.save(file_path)
        return

    G = nx.DiGraph()
    G.add_nodes_from(range(graph.num_nodes))
    edges_ = graph.edge_index.cpu().numpy().T.tolist()
    G.add_edges_from(edges_)
    pos = nx.spring_layout(G, seed=111)

    plt.figure(figsize=(8, 6), dpi=80)

    dic = {}

    if labs:
        assert (graph.x.shape[0] == len(labs))
        dic = {}
        for i, s in enumerate(labs):
            dic[i] = s

    G_edges = list(G.edges)
    temp = [tuple(pair) for pair in graph.edge_index.cpu().numpy().T]
    perm = [temp.index(pair) for pair in G_edges]
    edge_values = [edge_values[perm[i]] for i in range(len(edge_values))]

    from collections import defaultdict
    map = defaultdict(lambda: float('-inf'))
    for _ in range(2):
        for i, pair in enumerate(G_edges):
            key = tuple(sorted(list(pair)))
            map[key] = max(map[key], edge_values[i])
            edge_values[i] = map[key]

    edge_values_dic = {}
    for i, e in enumerate(G.edges):
        edge_values_dic[e] = edge_values[i]

    edge_val_min = min(edge_values) if len(edge_values) > 0 else 0
    edge_val_max = max(edge_values) if len(edge_values) > 0 else 0

    norm = matplotlib.colors.Normalize(vmin=0, vmax=edge_val_max if anchor is None else anchor, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('plasma'))

    net = Network(notebook=True)
    net.from_nx(G)
    for i, e in enumerate(net.edges):
        edge_val = edge_values_dic[(e['from'], e['to'])]
        f = (edge_val - edge_val_min) / (edge_val_max - edge_val_min)
        rgba = mapper.to_rgba(edge_val)
        e['color'] = '#%02x%02x%02x' % (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
        e['width'] = 2 + f * 8
        e['label'] = str(f)
    if node_values is not None:
        for n in net.nodes:
            rgba = mapper.to_rgba(node_values[n['id']])
            n['color'] = '#%02x%02x%02x' % (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
    else:
        for i, n in enumerate(net.nodes):
            color = color_map[i]
            n['color'] = color

    file_path = '../graphs/example.html' if not custom_name else f'../graphs/{custom_name}.html'
    create_dir(file_path)

    net.show_buttons(filter_=["physics"])
    net.show(file_path)

    nx.draw(G, pos=pos, alpha=1, width=0, node_size=0)
    if node_values is not None:
        nx.draw_networkx_nodes(G, pos=pos, node_color=node_values, cmap=cm.get_cmap('plasma'), alpha=.6, node_size=260,
                               vmin=0.0, vmax=max(node_values) if anchor is None else anchor)
    else:
        nx.draw_networkx_nodes(G, pos=pos, node_color=color_map, alpha=.6, node_size=260)
    nx.draw_networkx_edges(G, pos=pos, edge_color=edge_values,
                           edge_cmap=cm.get_cmap('plasma'), width=6,
                           arrows=False, alpha=1, edge_vmin=0.0, edge_vmax=edge_val_max if anchor is None else anchor)
    nx.draw_networkx_edges(G, pos=pos, width=3, arrows=False, alpha=.1)

    file_path = f'../figs/{custom_name}' if custom_name else '../figs/output.png'
    create_dir(file_path)
    plt.savefig(file_path)
