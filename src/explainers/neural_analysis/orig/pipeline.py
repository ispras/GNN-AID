import networkx as nx
import numpy as np
import os.path as osp
import random
import torch
from collections import defaultdict
from os.path import exists
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

from explainers.neural_analysis.orig.graph_utils import k_hop_neighbourhood
from explainers.neural_analysis.orig.node_explainer import ClearGNN


class SyntheticDataset(TUDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, name, transform, pre_transform, pre_filter)

    def gen_graph(self):
        if self.name == 'test':
            while True:
                num_nodes = random.randint(10, 40)
                num_edges = random.randint(20, 40)
                g = nx.gnm_random_graph(num_nodes, num_edges)  # nx.barabasi_albert_graph(n=num_nodes, m=3)
                node_labels = torch.multinomial(torch.tensor([0.5, 0.25, 0.25]), num_nodes,
                                                replacement=True)  # torch.randint(0, 2, [num_nodes])

                special_idx = torch.randint(0, num_nodes, [1]).item()
                node_labels[special_idx] = 3

                g = from_networkx(g)

                if special_idx not in g.edge_index.unique().numpy():
                    u = special_idx
                    v = u
                    while v == u:
                        v = torch.randint(0, num_nodes, [1]).item()
                    new_edge = torch.LongTensor([[u, v], [v, u]])
                    g.edge_index = torch.column_stack([g.edge_index, new_edge])

                k_hop = k_hop_neighbourhood(g.edge_index, special_idx, 3)

                ok = any([node_labels[n] == 1 or node_labels[n] == 2 for n in k_hop])
                if not ok:
                    v = special_idx
                    while v == special_idx:
                        v = k_hop[torch.randint(0, len(k_hop), [1]).item()]
                    node_labels[v] = torch.randint(1, 3, [1]).item()

                s1 = sum([1 for j in k_hop if node_labels[j] == 1])
                s2 = sum([1 for j in k_hop if node_labels[j] == 2])

                if s1 == s2:
                    continue

                y = 1 if s1 >= s2 else 0
                break
        elif self.name == 'frontier3':
            G = nx.Graph()

            frontier = [0]
            node_counter = 1

            for _ in range(5):
                frontier_ = []
                while frontier != []:
                    u = frontier.pop()
                    for _ in range(torch.randint(1, 4, [1]).item()):
                        v = node_counter
                        frontier_.append(v)
                        G.add_edge(u, v)
                        node_counter += 1
                frontier = frontier_

            node_labels = torch.multinomial(torch.tensor([0.5, 0.25, 0.25]), node_counter,
                                            replacement=True)  # torch.randint(0, 2, [num_nodes])
            special_idx = torch.randint(0, node_counter, [1]).item()
            node_labels[special_idx] = 3

            g = from_networkx(G)

            k0_hop = k_hop_neighbourhood(g.edge_index, special_idx, 3)
            # k1_hop = set(k_hop_neighbourhood(g.edge_index, special_idx, 2))

            y = 0
            for j in k0_hop:
                if node_labels[j] == 1:
                    y = 1
                    break

        return node_labels, g.edge_index, y

    def download(self):
        A = osp.join(self.root, self.name, 'raw', f'{self.name}_A.txt')
        graph_indicator = osp.join(self.root, self.name, 'raw', f'{self.name}_graph_indicator.txt')
        graph_labels = osp.join(self.root, self.name, 'raw', f'{self.name}_graph_labels.txt')
        node_labels = osp.join(self.root, self.name, 'raw', f'{self.name}_node_labels.txt')

        if osp.exists(A) and osp.exists(graph_indicator) and osp.exists(graph_labels) and osp.exists(node_labels):
            return

        f_A = open(A, 'w')
        f_graph_indicator = open(graph_indicator, 'w')
        f_graph_labels = open(graph_labels, 'w')
        f_node_labels = open(node_labels, 'w')

        node_counter = 1  # edge indices are one-indexed
        class_counter = defaultdict(int)

        for graph_idx in tqdm(range(2000)):
            while True:
                node_labels, edge_index, y = self.gen_graph()
                if class_counter[y] == 1000:
                    continue
                class_counter[y] += 1
                break

            for pair in (edge_index + node_counter).T:
                line = f'{pair[0].item()}, {pair[1].item()}\n'
                f_A.write(line)

            for node in range(len(node_labels)):
                f_graph_indicator.write(f'{graph_idx + 1}\n')
                f_node_labels.write(f'{node_labels[node]}\n')

            f_graph_labels.write(f'{y}\n')

            node_counter += len(node_labels)

        f_A.close()
        f_graph_labels.close()
        f_graph_indicator.close()
        f_node_labels.close()

    def process(self):
        print('test')
        super().process()


# load a dataset, with a specified fold for train-validation-test split
def load_dataset(name, fold=0):
    seeds1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    seeds2 = [123, 321, 420, 1234, 5121, 554, 3, 5, 6, 7, 11]

    np.random.seed(seeds1[fold])
    torch.random.manual_seed(seeds2[fold])

    if name == 'MUTAG':
        dataset = TUDataset(root='../data/TUDataset', name='MUTAG', use_edge_attr=True)
        dataset = dataset.shuffle()
        train_dataset = dataset[:120]
        test_dataset = dataset[120:]
        val_dataset = dataset[:120]
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    elif name == 'PROTEINS':
        dataset = TUDataset(root='../data/TUDataset', name='PROTEINS')
        dataset = dataset.shuffle()
        train_dataset = dataset[:600]
        val_dataset = dataset[600:800]
        test_dataset = dataset[800:]
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    elif name == 'MUTAGENICITY':
        dataset = TUDataset(root='../data/TUDataset', name='Mutagenicity')
        dataset = dataset.shuffle()
        # train_dataset = dataset[:1500]
        train_dataset = dataset[:100]
        val_dataset = dataset[1500:1700]
        test_dataset = dataset[1700:2200]
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    elif name == 'REDDIT':
        dataset = TUDataset(root='../data/TUDataset', name='REDDIT-BINARY')
        dataset = dataset.shuffle()

        dataset_ = []
        for graph in dataset:
            graph.x = torch.ones((graph.num_nodes, 10))
            dataset_.append(graph)
        dataset = dataset_

        train_dataset = dataset[:1000]
        val_dataset = dataset[1000:1200]
        test_dataset = dataset[1200:]
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    elif name == 'IMDB':
        dataset = TUDataset(root='../data/TUDataset', name='IMDB-BINARY')
        dataset = dataset.shuffle()

        dataset_ = []
        for graph in dataset:
            graph.x = torch.ones((graph.num_nodes, 10))
            dataset_.append(graph)
        dataset = dataset_

        train_dataset = dataset[:500]
        val_dataset = dataset[500:600]
        test_dataset = dataset[600:]

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    elif name == 'FRONTIER3':
        dataset = SyntheticDataset(root='../data/CUDataset', name='frontier3')
        dataset = dataset.shuffle()

        train_dataset = dataset[:700]
        val_dataset = dataset[700:1000]
        test_dataset = dataset[1000:]
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    else:
        raise Exception()

    return train_loader, test_loader, val_loader, dataset, train_dataset, test_dataset, val_dataset


def train_standard_model(name, layer_type='', aggr='', mode='default', custom_name='', custom_lr=None,
                         custom_epochs=None, custom_es=0, fold=0, overwrite=False, custom_layers=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_dir = '../models/'
    train_loader, test_loader, val_loader, dataset, _, _, _ = load_dataset(name, fold=fold)
    lr = 0.001

    early_stop = 10000
    aggr_type_ = 'sum'

    if name == 'BBBP':
        is_graph_task = True
        n_classes = 2
        n_layers = 3
        n_features = 9
        n_units = 64
        conv_type = 'GCN'
        n_epochs = 850
        criterion = torch.nn.CrossEntropyLoss()
    elif name == 'MUTAG':
        is_graph_task = True
        n_classes = 2
        n_layers = 3
        n_features = 7
        n_units = 64
        conv_type = 'GCN'

        #n_epochs = 850
        n_epochs = 20
        criterion = torch.nn.CrossEntropyLoss()
    elif name == 'PROTEINS':
        is_graph_task = True
        n_classes = 2
        n_layers = 3
        n_features = 3
        n_units = 64
        conv_type = 'GCN'

        early_stop = 60
        n_epochs = 700
        criterion = torch.nn.CrossEntropyLoss()
    elif name == 'ENZYMES':
        is_graph_task = True
        n_classes = 6
        n_layers = 3
        n_features = 3
        n_units = 64
        conv_type = 'GCN'

        n_epochs = 2000
        criterion = torch.nn.CrossEntropyLoss()
    elif name == 'NCI':
        is_graph_task = True
        n_classes = 2
        n_layers = 3
        n_features = 37
        n_units = 64
        conv_type = 'GCN'

        n_epochs = 700
        criterion = torch.nn.CrossEntropyLoss()
    elif name == 'BA':
        is_graph_task = False
        n_classes = 4
        n_layers = 3
        n_features = 10
        n_units = 64
        conv_type = 'GCN'

        n_epochs = 2000
        criterion = torch.nn.CrossEntropyLoss()
    elif name == 'BA2MOTIFS':
        is_graph_task = True
        n_classes = 2
        n_layers = 3
        n_features = 10
        n_units = 64
        conv_type = 'GCN'

        n_epochs = 100
        criterion = torch.nn.CrossEntropyLoss()
    elif name == 'SST':
        is_graph_task = True
        n_classes = 2
        n_layers = 3
        n_features = dataset.get(0).x.shape[1]
        n_units = 64
        conv_type = 'GCN'

        n_epochs = 80
        criterion = torch.nn.CrossEntropyLoss()
    elif name == 'MUTAGENICITY':
        is_graph_task = True
        n_classes = 2
        n_layers = 3
        n_features = 14
        n_units = 64
        conv_type = 'GCN'

        #n_epochs = 1600
        n_epochs = 100
        criterion = torch.nn.CrossEntropyLoss()
    elif name == 'REDDIT':
        is_graph_task = True
        n_classes = 2
        n_layers = 3
        n_features = 10
        n_units = 64
        conv_type = 'GCN'
        early_stop = 20000

        n_epochs = 1000
        criterion = torch.nn.CrossEntropyLoss()
    elif name == 'IMDB':
        is_graph_task = True
        n_classes = 2
        n_layers = 3
        n_features = 10
        n_units = 64
        conv_type = 'GCN'

        n_epochs = 1000
        early_stop = 400

        lr = 0.004
        criterion = torch.nn.CrossEntropyLoss()
    elif name == 'FRONTIER3':
        is_graph_task = True
        n_classes = 2
        n_layers = 3
        n_features = 4
        n_units = 64
        conv_type = 'GIN'

        n_epochs = 100
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise Exception('No such model!')

    if custom_layers is not None:
        n_layers = custom_layers

    if custom_es > 0:
        early_stop = custom_es

    if layer_type != '':
        conv_type = layer_type
        name += '-' + layer_type

    if aggr != '':
        aggr_type_ = aggr
        name += '-' + aggr

    name = name + '-fold' + str(fold)

    if mode == 'default':
        model = ClearGNN(n_features=n_features, n_units=n_units, n_layers=n_layers, n_classes=n_classes, device=device,
                         graph_level=is_graph_task, conv_type=conv_type, aggr_type=aggr_type_)
    else:
        raise Exception()

    if custom_name != '':
        name = custom_name
    if custom_lr is not None:
        lr = custom_lr
    if custom_epochs is not None:
        n_epochs = custom_epochs

    if exists(model_dir + name + '.pth') and not overwrite:
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            model.load_state_dict(torch.load('./models/' + name + '.pth'))
        else:
            model.load_state_dict(torch.load('./models/' + name + '.pth', map_location=torch.device('cpu')))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
        model.train_loop(train_loader, criterion, optimizer, num_epochs=n_epochs, data_v=val_loader,
                         path='./models/' + name + '.pth', early_stop=early_stop)

    return model
