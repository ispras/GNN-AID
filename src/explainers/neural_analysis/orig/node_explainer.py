import copy
import numpy as np
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from collections import defaultdict
from pyvis.network import Network
from torch import Tensor
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import SGConv, GATConv, HANConv, GINConv, GCNConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn.dense.linear import Linear
from tqdm import tqdm
from typing import List

from explainers.neural_analysis.orig.vis import show_graph

from explainers.neural_analysis.orig.concepts import ConceptSet
from explainers.neural_analysis.orig.graph_utils import edge_index_to_tuples, add_edge

#from graph_utils import graph_to_mol

THRESH = 0.01


def visualize_network(G):
    net = Network(notebook=True)
    net.from_nx(G)
    net.show_buttons(filter_=["physics"])
    net.show("example.html")


def all_subgraphs(graph, node_idx, k_hop):
    edges = graph.edge_index.T.cpu().numpy().tolist()
    edges = set([tuple(l) for l in edges])

    edges_filtered = set()

    stack = [node_idx]

    while stack:
        u = stack.pop()
        for v in range(graph.x.shape[0]):
            if (u, v) in edges and (u, v) not in edges_filtered:
                edges_filtered.add((u, v))
                if len(stack) < k_hop - 1:
                    stack.append(v)
            if (v, u) in edges and (v, u) not in edges_filtered:
                edges_filtered.add((v, u))
                if len(stack) < k_hop - 1:
                    stack.append(v)

    nodes = set()
    edge_index_new = [[], []]
    for (u, v) in edges_filtered:
        nodes.add(u)
        nodes.add(v)
        edge_index_new[0].append(u)
        edge_index_new[1].append(v)
    nodes = list(nodes)

    x_new = graph.x.index_select(0, torch.tensor(nodes))
    edge_index_new = torch.tensor(edge_index_new)

    return Data(x_new, edge_index_new)


class ModelBase(torch.nn.Module):
    def __init__(self, graph_level, device=None):
        super().__init__()
        self.graph_level = graph_level
        self.device = device

    def forward(self, x, edge_index, batch):
        raise NotImplementedError()

    def test_acc(self, loader):
        self.eval()
        correct = 0
        for data in loader:
            out = self.forward(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
        return correct / len(loader.dataset)

    def train_loop(self, data_t, criterion, optimizer, num_epochs, data_v=None, path=None, early_stop=60):
        pbar = tqdm(range(num_epochs))

        best_train_acc = 0
        counter = 0
        best_val_acc = 0

        for epoch in pbar:
            self.train_epoch(data_t, criterion, optimizer)

            train_acc = self.evaluate(data_t)
            valid_acc = self.evaluate(data_v) if data_v else 0

            if path and valid_acc > best_train_acc:
                best_train_acc = valid_acc
                torch.save(self.state_dict(), path)
                counter = 0

            elif counter == early_stop:
                break

            counter += 1

            pbar.set_description(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}')

    def train_epoch(self, data, criterion, optimizer):
        self.train()

        if isinstance(data, torch_geometric.loader.dataloader.DataLoader):
            if not self.graph_level:
                raise Exception('Node-level model cannot use loaders.')

            for d in data:
                d = d.to(self.device)

                out = self.forward(d.x, d.edge_index, d.batch)
                d_y = d.y * 1.0 if isinstance(criterion, torch.nn.BCELoss) else d.y.long()

                loss = criterion(out, d_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                del d
        elif isinstance(data, torch_geometric.data.Data):
            if self.graph_level:
                raise Exception('Graph-level model must use loaders.')

            data.to(self.device)
            out = self.forward(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            raise Exception()

    def evaluate(self, data, mask=None):
        self.eval()

        if isinstance(data, torch_geometric.loader.dataloader.DataLoader):
            if not self.graph_level:
                raise Exception('Node-level model cannot use loaders.')

            correct = 0

            for d in data:
                d.to(self.device)

                out = self.forward(d.x, d.edge_index, d.batch)
                pred = out.argmax(dim=1) if len(out.shape) > 1 else torch.round(out)

                try:
                    correct += int((pred == d.y).sum())
                except Exception:
                    print(d.batch)

                del d

            return correct / len(data.dataset)

        elif isinstance(data, torch_geometric.data.Data):
            if self.graph_level:
                raise Exception('Graph-level model must use loaders.')

            d = data.to(self.device)
            mask = d.val_mask
            out = self.forward(d.x, d.edge_index)
            t = torch.argmax(out, axis=1)
            return torch.sum(t[mask] == d.y[mask]) / t[mask].shape[0]


class ClearGNN(ModelBase):
    '''
    An explainable graph neural network.
    '''

    def __init__(self, n_features, n_units, n_layers, n_classes, graph_level=True, device=None, conv_type='GCN',
                 aggr_type='sum'):
        super().__init__(graph_level, device=device)

        self.n_features = n_features
        self.n_units = n_units
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.conv_type = conv_type

        self.convs = torch.nn.ModuleList()
        self.convs.append(self.conv_layer(n_features, n_units))
        self.bottleneck_concepts = []
        self.aggr_type = aggr_type

        for i in range(n_layers - 1):
            self.convs.append(self.conv_layer(n_units, n_units))

        self.lin1 = Linear(n_units, n_classes)

        self.erasures = defaultdict(list)

        self.node_only = False
        self.include_top = True
        self.softmax = False

        self.dropout = torch.nn.Dropout(0.5)

        if self.device:
            self.to(self.device)

    def concept_search(self, task, dataset, depth=1, neuron_idxs=[], top=64, augment=True, omega=[10, 20, 20], level=1):
        assert depth >= 1

        if augment:
            print('Graph augmentation')
            dataset_aug = []
            for graph in tqdm(dataset):
                edge_tuples = edge_index_to_tuples(graph.edge_index)
                dataset_aug.append(copy.deepcopy(graph))
                for _ in range(4):
                    graph_new = copy.deepcopy(graph)
                    for _ in range(8):
                        node_i = np.random.randint(0, graph_new.x.shape[0])
                        node_j = np.random.randint(0, graph_new.x.shape[0])
                        add_edge(graph_new, node_i, node_j)
                    dataset_aug.append(graph_new)
            dataset = dataset_aug

        n_graphs = len(dataset)
        concept_set = ConceptSet(dataset, task, omega=omega)

        print('Performing inference')
        neuron_activations = []
        graph_sizes = []
        graph_inds = []

        for i in tqdm(range(n_graphs)):
            graph = dataset[i]
            feature_maps = self.partial_forward(graph.x.to(self.device), graph.edge_index.to(self.device),
                                                ret_layer=self.n_layers - level).detach().cpu().T
            neuron_activations.append(feature_maps)

            graph_sizes.extend([graph.x.shape[0]] * graph.x.shape[0])
            graph_inds.extend([i] * graph.x.shape[0])

        print('Keeping only top neurons')
        neuron_activations = torch.cat(neuron_activations, 1)
        nrns_vals = (neuron_activations != 0).sum(axis=1)
        neuron_idxs = nrns_vals.argsort()
        non_zero_neuron_idxs = []
        for idx in neuron_idxs:
            if nrns_vals[idx] == 0:
                continue
            non_zero_neuron_idxs.append(idx)
        non_zero_neuron_idxs = torch.LongTensor(non_zero_neuron_idxs)
        neuron_idxs = non_zero_neuron_idxs
        neuron_activations = neuron_activations.index_select(0, neuron_idxs[-top:])

        print('Performing search')

        for i in range(depth):
            ret = concept_set.match(neuron_activations, torch.tensor(graph_sizes), torch.tensor(graph_inds))

            if i < depth - 1:
                print('Adding concepts')
                concept_set.expand()
                print('Number of concepts: ' + str(concept_set.num_concepts()))

        concept_set.free()

        ret_dic = {}
        for k, v in ret.items():
            ret_dic[neuron_idxs[k].item()] = v

        return ret_dic

    def test_switch(self, mode):
        for conv in self.convs:
            conv.test_mode = mode

    def conv_layer(self, in_dim, out_dim):
        if self.conv_type == 'GCN':
            return GCNConv(in_dim, out_dim)
        if self.conv_type == 'GAT':
            return GATConv(in_dim, out_dim)
        if self.conv_type == 'GIN':
            seq = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
                nn.ReLU()
            )
            return GINConv(nn=seq)
        if self.conv_type == 'SGC':
            return SGConv(in_dim, out_dim)
        if self.conv_type == 'HAN':
            return HANConv(in_dim, out_dim)

    def bottleneck(self, neurons):
        self.bottleneck_concepts = neurons
        self.lin1 = Linear(len(self.bottleneck_concepts), self.n_classes).to(self.device)

        for conv in self.convs:
            conv.requires_grad_(False)

    def concept_layer(self, x, edge_index, batch=None):
        self.include_top = False
        neurons = self.forward(x, edge_index, batch)
        self.include_top = True
        return neurons

    def forward(self, x, edge_index, batch=None):
        x = self.partial_forward(x, edge_index)

        if self.node_only:
            return x

        if self.graph_level:
            if self.aggr_type == 'sum':
                x = global_add_pool(x, torch.tensor([0], device=self.device) if batch is None else batch)
            elif self.aggr_type == 'max':
                x = global_max_pool(x, torch.tensor([0], device=self.device) if batch is None else batch)
            elif self.aggr_type == 'mean':
                x = global_mean_pool(x, torch.tensor([0], device=self.device) if batch is None else batch)
            else:
                raise Exception()

        if not self.include_top:
            return x

        x = self.lin1(x)

        if self.softmax:
            x = nn.Softmax(dim=1)(x)

        return x

    def neuron_importance(self, train_dataset=None, test_dataset=None, method='shap'):
        if method == 'shap':
            X_train = []
            X_test = []

            for data in train_dataset:
                data = data.to(self.device)
                neurons = self.concept_layer(data.x, data.edge_index, data.batch)
                X_train.append(neurons.detach())
            for data in test_dataset:
                data = data.to(self.device)
                neurons = self.forward(data.x, data.edge_index, data.batch)
                X_test.append(neurons.detach())

            X_train = torch.row_stack(X_train)
            X_test = torch.row_stack(X_test)

            print('DEBUG___')
            print('X_train shape: ' + str(X_train.shape))
            print('X_test shape: ' + str(X_test.shape))

            top_level = torch.nn.Sequential(
                self.lin1,
                torch.nn.Softmax()
            )

            explainer = shap.DeepExplainer(top_level, X_train)
            shap_values = explainer.shap_values(X_test)

            return X_train, X_test, shap_values
        elif method == 'weights':
            t = torch.clone(self.lin1.weight).detach().cpu()
            return [t[0].unsqueeze(0).numpy(), t[1].unsqueeze(0).numpy()]
        else:
            raise Exception('No method %s' % method)

    def inspect_neuron(self, x, edge_index, node_idx, neuron_idx, layer=None):
        if layer is None:
            layer = self.n_layers - 1
        activations = self.partial_forward(x, edge_index, ret_layer=layer)
        return activations[node_idx, neuron_idx].item()

    def fidelity(self, x, edge_index, y, edge_importance, prune_nodes=False, debug=True, limit=0.5, gap=1):
        # edge_importance is 1d tensor
        self.softmax = True

        pred_initial = self.forward(x, edge_index)[0, y].item()

        sparsities = []
        fidelities = []

        mask_order = edge_importance.argsort()[::-1]
        edge_mask = torch.ones(edge_index.shape[1], device=self.device).long()

        if debug:
            print(self.forward(x, edge_index))
            print(mask_order)
            print(edge_mask)

        es = []

        for i in range(0, mask_order.shape[0]):
            edge_mask[mask_order[i]] = 0

            if i % gap != 0:
                continue

            sparsity = 1 - edge_mask.sum().item() / len(edge_importance)

            if sparsity > limit:
                break

            edge_index_ = edge_index.T.index_select(0, edge_mask.nonzero().squeeze(1)).T

            node_mask = torch.zeros(x.shape[0], device=self.device)
            for [u, v] in edge_index_.T:
                node_mask[u] = 1
                node_mask[v] = 1
            x_ = x * node_mask[:, None] if prune_nodes else x

            es.append(edge_index_)

            pred_ = self.forward(x_, edge_index_)[0, y].item()

            if debug:
                print(self.forward(x_, edge_index_))

            fid = (pred_initial - pred_) / pred_initial

            sparsities.append(sparsity)
            fidelities.append(fid)

        self.softmax = False

        return np.array(sparsities), np.array(fidelities),  # es

    def neuron_accuracy(self, data):
        self.eval()

        correct = 0
        results = []
        neuron_on = []

        for d in data:
            d.to(self.device)

            acts = self.partial_forward(d.x, d.edge_index, d.batch)
            neuron_on.append(acts.sum(dim=0) != 0)

            out = self.forward(d.x, d.edge_index, d.batch)
            pred = out.argmax(dim=1) if len(out.shape) > 1 else torch.round(out)
            correct += int((pred == d.y).sum())
            print(pred == d.y)
            results.append((pred == d.y))

            del d

        results = torch.cat(results)  # bool [n_graphs]
        neuron_on = torch.row_stack(neuron_on)  # bool mat [n_graphs, n_neurons]
        correct_conditioned = neuron_on & results[:, None]  # col broadcast
        neuron_accs = correct_conditioned.sum(dim=0) / (neuron_on.sum(dim=0) + 0.001)

        return neuron_accs

    def back_flow(self, graph, map, xs, ws, cms, flag='increase'):
        x = graph.x
        edge_index = graph.edge_index

        adj = defaultdict(list)
        for j in range(edge_index.shape[1]):
            adj[edge_index[0][j].item()].append(edge_index[1][j].item())
        for n in adj.keys():
            if n not in adj[n]:
                adj[n].append(n)

        deg = torch.tensor([len(adj[n]) for n in range(x.shape[0])])
        deg = deg ** (-0.5)

        cur_map = map
        masks = []

        seen = set()

        for i in reversed(range(len(cms))):
            x, cm, w = xs[i], cms[i], ws[i]
            edge_weight_map = defaultdict(float)
            edge_weight_norm = defaultdict(float)

            new_map = {}
            seen_ = set()

            for node, neurons in cur_map.items():
                adj_list = torch.LongTensor(adj[node])
                neighbor_deg = deg.index_select(0, adj_list)
                self_deg = deg[node]

                for nrn, c in neurons.items():
                    if c == 0:
                        continue

                    x_norm = x.index_select(0, adj_list) * neighbor_deg[:, None] * self_deg
                    x_norm *= c

                    if flag == 'increase':

                        msk = torch.zeros(x_norm.shape[1])
                        msk.index_fill_(0, w[nrn].argsort()[-1:], 1)
                        x_norm_cmt = torch.maximum(x_norm * w[nrn] * msk,
                                                   torch.zeros(x_norm.shape))  # zero-out negative neurons
                        x_norm_ = x_norm_cmt

                        nrn_conts = (x_norm * w[nrn]).flatten()
                        nrns_sig = nrn_conts.argsort()[-8:]

                    elif flag == 'decrease':

                        nrn_conts = (x_norm * w[nrn]).flatten()
                        nrns_sig = torch.maximum(nrn_conts, torch.zeros(nrn_conts.shape)).argsort()[:32]

                        msk = torch.zeros(x_norm.shape[1])
                        msk.index_fill_(0, w[nrn].argsort()[:1], 1)
                        x_norm_cmt = torch.abs(torch.minimum(x_norm * w[nrn] * msk, torch.zeros(x_norm.shape)))  # up
                        flag = 'increase'
                    else:
                        raise Exception()

                    norm_term = x_norm.sum()

                    for idx in nrns_sig:
                        node_ = idx.item() // x_norm.shape[1]

                        from_node = adj_list[node_].item()

                        edge = (from_node, node)
                        if edge in seen:
                            continue

                        neuron_contrib = torch.abs(nrn_conts[idx.item()])
                        edge_weight_map[edge] += neuron_contrib.item()
                        edge_weight_norm[edge] = 1.0

                        seen_.add(edge)
                        seen_.add((edge[1], edge[0]))

                    for pair in x_norm_cmt.nonzero():
                        node_, nrn_ = pair[0].item(), pair[1].item()
                        from_node = adj_list[node_].item()

                        edge = (from_node, node)
                        if edge in seen:
                            continue

                        if from_node not in new_map:
                            new_map[from_node] = defaultdict(list)

                        neuron_contrib = x_norm_cmt[node_, nrn_]  # / norm_term * c
                        new_map[from_node][nrn_].append(neuron_contrib.item())

                        seen_.add(edge)
                        seen_.add((edge[1], edge[0]))

            edge_weights = torch.zeros(edge_index.shape[1])
            for e in range(edge_index.shape[1]):
                edge = (edge_index[0][e].item(), edge_index[1][e].item())
                edge_weights[e] = edge_weight_map[edge] / (1 if 0 == edge_weight_norm[edge] else edge_weight_norm[edge])
            masks.append(edge_weights)

            for key, dic in new_map.items():
                dic_ = defaultdict(float)
                for nrn, nums in dic.items():
                    dic_[nrn] = sum(nums)  # / len(nums)
                cur_map[key] = dic_

            seen = seen.union(seen_)

        return masks

    def expl_gp_flow(self, graph, y, level, visualise=True, labels=None, debug=False, top=1, toptop=3, nrns=None,
                     force_node=None,
                     flag='increase'):
        x = graph.x
        edge_index = graph.edge_index

        xs = []  # features before conv layer
        cms = []  # features after aggregation, before linear
        ws = []  # linear weights

        x_dev = x.to(self.device)
        edge_index_dev = edge_index.to(self.device)

        for idx, conv in enumerate(self.convs):
            xs.append(torch.clone(x_dev.cpu().detach()))
            x_dev = conv(x_dev, edge_index_dev)
            cms.append(conv.cached_messages.cpu().detach())
            ws.append(torch.clone(conv.lin.weight.cpu().detach()))
            x_dev = F.relu(x_dev)

        if debug:
            return x_dev.detach().cpu()

        mask_final = None

        if nrns is None:
            importance = self.lin1.weight[y]
            nrns = []

            self.include_top = False
            concept_layer = self.forward(x.to(self.device), edge_index.to(self.device)).detach()
            self.include_top = True
            weights = (importance * concept_layer)[0]
            ranking = weights.argsort()

            for nrn in ranking[:top]:
                nns = x_dev[:, nrn].argsort()[-toptop:] if force_node is None else [force_node]
                for node_idx in nns:
                    idx = node_idx * x_dev.shape[1] + nrn
                    nrns.append(idx)

        for idx in nrns:
            map = {}
            node_ = idx.item() // x_dev.shape[1]
            nrn_ = idx.item() % x_dev.shape[1]
            contrib = x_dev[node_][nrn_].item() * abs(self.lin1.weight[y, nrn_].item())  # 1.0

            if contrib == 0:
                continue

            if node_ not in map:
                map[node_] = defaultdict(float)
            map[node_][nrn_] += contrib
            masks = self.back_flow(graph, map, xs, ws, cms, flag=flag)

            if mask_final is None:
                mask_final = torch.zeros(masks[0].shape)

            for m in masks:
                mask_final = mask_final + m

        if debug:
            print((x.flatten()).sort()[0][-32:])
            for n, d in map.items():
                print(f'{n} {len(d)}')

        # mask_final = mask_final ** 0.1

        if visualise:
            show_graph(Data(graph.x, graph.edge_index), mask_final.detach().cpu().numpy(), None, labels)

        return mask_final.cpu().numpy()

    def rand_edge_mask(self, graph):
        final_mask = torch.rand(graph.edge_index.shape[1]).numpy()
        node_values = torch.rand(graph.x.shape[0]).numpy()
        return final_mask, node_values

    def expl_gp_neurons(self, graph, y, concepts=None, gamma=0.5, rank=1, visualise=True, labels=None, debug=False,
                        res=300, mode='greedy',
                        cum=True, explore=False, level=None, pool='mean', aggr='sum', show_labels=True,
                        show_contribs=False, entropic=False, show_node_mask=False, edge_thresh=None, force=False,
                        sigma=1.0, scores=[], anchor=None, custom_name=None, names=[], as_molecule=False):
        x = graph.x
        edge_index = graph.edge_index

        assert pool in ['mean', 'min', 'rand']
        assert mode in ['greedy', 'fair']

        from explainers.neural_analysis.orig import concept_ranker

        if entropic or show_contribs:
            vals_ent = concept_ranker.by_entropy(self, graph.x, graph.edge_index, y, epochs=200, beta=1, sort=False)

        if show_contribs:
            vals_abs = concept_ranker.by_weight_x_val(self, graph.x, graph.edge_index, y, sort=False)

        self.include_top = False
        concept_layer = self.forward(x.to(self.device), edge_index.to(self.device)).detach().cpu().numpy()
        self.include_top = True

        feature_maps = self.partial_forward(x.to(self.device), edge_index.to(self.device),
                                            ret_layer=level).detach().cpu()
        edge_weights = defaultdict(float)
        edge_set = {(pair[0].item(), pair[1].item()) for pair in edge_index.T}
        node_values = np.zeros(feature_maps.shape[0])

        if mode == 'greedy':
            importance = self.neuron_importance(method='weights')[y].squeeze()
            if entropic:
                weights = vals_ent - 0.5
            else:
                weights = (importance * concept_layer)[0]

            ranking = weights.argsort()

            if type(rank) is list:
                units = [ranking[-r] for r in rank]
            elif type(rank) is set:
                units = list(rank)
            else:
                units = ranking[-rank:][::-1] if cum else [ranking[-rank]]
                units_ = []
                for u in units:
                    if weights[u] <= 0:
                        continue
                    units_.append(u)
                units = units_

            mask_set = []
            node_vals_set = []

            def iou(s1, s2):
                if len(s1) == 0 or len(s2) == 0:
                    return 1 if len(s1) == len(s2) else 0
                assert len(s1) > 0 and len(s2) > 0
                s1_s = set(s1)
                s2_s = set(s2)
                assert len(s1_s) == len(s1) and len(s2_s) == len(s2)
                return len(s1_s.intersection(s2_s)) / len(s1_s.union(s2_s))

            for unit in units:
                if concepts is not None:
                    print(concepts[unit])

                if type(sigma) is list:
                    mask = feature_maps[:, unit] > sigma[unit]
                else:
                    order = feature_maps[:, unit].argsort().flip(0)
                    mask = torch.zeros(feature_maps.shape[0])
                    denom = (10 * feature_maps[:, unit]).exp().sum()
                    probs = (10 * feature_maps[:, unit]).exp() / denom
                    tot = 0
                    for k in order:
                        tot += probs[k]
                        mask[k] = 1.0
                        if tot >= sigma:
                            break

                where = mask.nonzero().squeeze(1).numpy()

                edges = []
                vals = []
                node_vals = []

                for i in range(feature_maps.shape[0]):
                    node_vals.append(feature_maps[i, unit])

                for (i, j) in edge_set:
                    if i == j:
                        continue
                    if (explore and (i in where or j in where)) or (not explore and i in where and j in where):
                        v_i = feature_maps[i, unit].item()
                        v_j = feature_maps[j, unit].item()
                        if pool == 'mean':
                            r = 0.5 * v_i + 0.5 * v_j
                        elif pool == 'min':
                            r = min(v_i, v_j)

                        edges.append((i, j))
                        vals.append(r)

                if (len(edges) == 0 or sum(node_vals) == 0) and not force:
                    continue

                too_similar = False

                if len(edges) != 0:
                    vals, edges = zip(*sorted(zip(vals, edges), reverse=True))

                    for unit_, edges_, vals_ in mask_set:
                        if iou(edges_[:gamma], edges[:gamma]) > 0.5:
                            too_similar = True
                            break

                if not too_similar:
                    if names != [] and scores != []:
                        print(f'Unit: {unit}  Concept: {names[unit]}  Score: {scores[unit]}')
                        if show_contribs:
                            print(f'ABS: {vals_abs[unit]} ENT: {vals_ent[unit]}')
                    mask_set.append((unit, edges, vals))
                    node_vals_set.append(node_vals)

            if debug:
                print(f'Mask set contains {len(mask_set)} masks')
                print([u for u, _, _ in mask_set])

            for i, (unit, edges, vals) in enumerate(mask_set):

                factor = 1.0
                for j, ((u, v), val) in enumerate(zip(edges, vals)):
                    if j >= gamma:
                        break
                    if aggr == 'sum':
                        edge_weights[(u, v)] = val * factor + edge_weights[(u, v)]
                    elif aggr == 'max':
                        edge_weights[(u, v)] = max(val * factor, edge_weights[(u, v)])
                node_values += np.array(node_vals_set[i]) * factor

        elif mode == 'fair':
            importance = torch.clone(self.lin1.weight[y].squeeze()).cpu()
            node_weights = (feature_maps * importance).index_select(1, (importance > 0).nonzero().squeeze()).sum(axis=1)
            edge_list = [(pair[0].item(), pair[1].item()) for pair in edge_index.T]

            for (i, j) in edge_list:
                v_i = node_weights[i]
                v_j = node_weights[j]
                if pool == 'mean':
                    r = 0.5 * v_i + 0.5 * v_j
                elif pool == 'min':
                    r = min(v_i, v_j)
                edge_weights[(i, j)] = r

        final_mask = np.zeros(edge_index.shape[1])
        for i in range(edge_index.shape[1]):
            edge = (edge_index[0, i].item(), edge_index[1, i].item())
            if edge in edge_weights:
                final_mask[i] = edge_weights[edge]

        if edge_thresh is not None:
            inds = (final_mask <= edge_thresh).nonzero()[0]
            final_mask[(final_mask > edge_thresh).nonzero()[0]] = 1.0
            final_mask[inds] = 0.0

        return final_mask, node_values

    def partial_forward(self, x, edge_index, ret_layer=None):
        if ret_layer is None:
            ret_layer = self.n_layers - 1

        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            x = F.relu(x)

            if len(self.erasures[idx]) != 0:
                rem = torch.tensor(self.erasures[idx]).to(self.device)
                x = torch.where(F.one_hot(rem, x.shape[1]).sum(axis=0) == 1, torch.tensor(0.).to(self.device), x)

            if idx == ret_layer:
                break

        if self.bottleneck_concepts != []:
            x = x.index_select(1, torch.LongTensor(self.bottleneck_concepts).to(self.device))

        return x

    def most_activated_units(self, x, edge_index, layer=None):
        if not layer:
            layer = self.n_layers - 1

        embeddings = self.partial_forward(x, edge_index, layer)
        return (embeddings > THRESH).sum(axis=0).sort()

    def all_embeddings(self, x, edge_index) -> List[Tensor]:
        xs = []

        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if idx < self.n_layers - 1:
                x = F.relu(x)

            if len(self.erasures[idx]) != 0:
                rem = torch.tensor(self.erasures[idx])
                x = torch.where(F.one_hot(rem, x.shape[1]).sum(axis=0) == 1, torch.tensor(0.), x)

            xs.append(x)

        return xs

    def set_erase(self, layer_i, dim_i):
        self.erasures[layer_i].append(dim_i)

    def reset_erase(self):
        self.erasures = defaultdict(list)


def max_corr(models, m, d, num_neurons):
    num_models = len(models)
    vals = []

    reprs = [repr(m, d)]
    reprs = reprs + [repr(m_, d) for m_ in range(num_models) if m_ != m]
    reprs = np.concatenate(reprs)
    C = np.corrcoef(reprs)
    C = np.nan_to_num(C)

    print(reprs.shape)

    for i in range(num_neurons):
        vals.append(np.max(np.abs(C[i, num_neurons:])))

    return np.array(vals)


def min_corr(models, m, d, num_neurons):
    num_models = len(models)
    vals = []

    reprs = [repr(m, d)]
    reprs = reprs + [repr(m_, d) for m_ in range(num_models) if m_ != m]
    reprs = np.concatenate(reprs)
    C = np.corrcoef(reprs)
    C = np.nan_to_num(C)

    for i in range(num_neurons):
        v = float('inf')
        for j in range(1, num_models):
            k = np.max(np.abs(C[i, num_neurons * j: num_neurons * (j + 1)]))
            v = min(v, k)
        vals.append(v)

    return np.array(vals)


def erasure(model, data_loader):
    model.eval()

    test_accs = []

    for i in range(model.n_units):
        model.reset_erase()
        model.set_erase(model.n_layers - 1, i)

        if isinstance(data_loader, torch_geometric.loader.dataloader.DataLoader):
            label_p = []
            label_t = []

            for data in data_loader:
                data.to(model.device)
                out = model(data.x, data.edge_index, data.batch)
                label_p.extend(torch.argmax(out, axis=1).cpu().detach().numpy())
                label_t.extend(data.y.cpu().detach().numpy())

            label_p = np.array(label_p)
            label_t = np.array(label_t)
        else:
            out = model(data_loader.x, data_loader.edge_index)
            label_p = torch.argmax(out, axis=1).cpu().detach().numpy()
            label_t = data_loader.y.cpu().detach().numpy()

        a = np.sum(label_p == label_t) / label_p.shape[0]
        test_accs.append(a)

    return np.argsort(test_accs), np.sort(test_accs)
