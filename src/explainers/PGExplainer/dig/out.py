from typing import Optional
from math import sqrt

import os
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_networkx

from explainers.PGExplainer.dig.utils import k_hop_subgraph_with_default_whole_graph, get_topk_edges_subgraph
from explainers.explainer import Explainer, finalize_decorator
from data_structures.explanation import AttributionExplanation


from aux.utils import root_dir
from pathlib import Path

EPS = 1e-6


class PGExplainer(nn.Module, Explainer):

    name = 'PGExplainer(dig)'

    @staticmethod
    def check_availability(gen_dataset, model_manager):
        """ Availability check for the given dataset and model manager. """
        return ({'modules', 'flow', 'get_num_hops', 'parameters', 'forward'}.issubset(dir(model_manager.gnn)) and
                any(isinstance(m, MessagePassing) for m in model_manager.gnn.modules()))

    def __init__(self,
                 gen_dataset,
                 model: torch.nn.Module,
                 device,
                 epochs: int = 20,
                 lr: float = 0.003,
                 num_hops: Optional[int] = None):
        super(PGExplainer, self).__init__()
        Explainer.__init__(self, gen_dataset, model)

        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.__num_hops__ = num_hops
        self.device = device

        self.coff_size: float = 0.01  # constrains on mask size
        self.coff_ent: float = 5e-4  # constrains on smooth and continuous mask
        self.init_bias = 0.0
        self.t0: float = 5.0  # temperature denominator
        self.t1: float = 1.0  # temperature numerator

        self.mask_sigmoid = None
        self.elayers = nn.ModuleList()

        self.graph_idx = None

        # TODO how can we take last_conv_emb_size easier?
        self._last_conv_emb_size = self.last_conv_emb_size

        self.elayers.append(nn.Sequential(nn.Linear(self._last_conv_emb_size, 64), nn.ReLU()))
        self.elayers.append(nn.Linear(64, 1))
        self.elayers.to(self.device)

        # TODO think about path / add for different models
        self._ckpt_path = self.ckpt_path  # path to the generator file

    def __set_masks__(self, x, edge_index, edge_mask=None):
        """ Set the weights for message passing """
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1
        init_bias = self.init_bias
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * std)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))

        if edge_mask is None:
            self.edge_mask = torch.randn(E) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask.to(self.device)
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        """ clear the edge weights to None """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None

    @property
    def ckpt_path(self):
        path = Path(root_dir)
        # path /= Path(str(model_manager.model_config()))
        path /= Path("PGE_gen_models")
        path /= Path(self.gen_dataset.name)
        path /= Path("epochs=" + str(self.epochs) + ",learn_rate=" + str(self.lr))
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, "generator")  # path to the generator file
        return path

    @property
    def last_conv_emb_size(self):
        if self.gen_dataset.is_multi():
            # TODO this may not work for non GIN models
            # FOR Test
            # valid architecture: (GIN, GIN, Pool)
            out_features = self.model.structure[-2]['layer']["gin_seq"][0]['layer']['layer_kwargs']['out_features']
            # FOR BA2Motifs
            # out_features = list(self.model.structure[-2]['layer'].values())[0][0]['layer']['layer_kwargs']['out_features']
        else:
            out_features = self.model.structure[-1]['layer']['layer_kwargs']['out_channels']
        emb_size = out_features * 2
        return emb_size

    @property
    def num_hops(self):
        """ return the number of layers of GNN model """
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __loss__(self, prob, ori_pred):
        """
        the pred loss encourages the masked graph with higher probability,
        the size loss encourage small size edge mask,
        the entropy loss encourage the mask to be continuous.
        """
        logit = prob[ori_pred]
        logit = logit + EPS
        pred_loss = -torch.log(logit)
        # size
        edge_mask = torch.sigmoid(self.mask_sigmoid)
        size_loss = self.coff_size * torch.sum(edge_mask)

        # entropy
        edge_mask = edge_mask * 0.99 + 0.005
        mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
        mask_ent_loss = self.coff_ent * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss
        return loss

    @staticmethod
    def concrete_sample(log_alpha, beta=1.0, training=True):
        """ Sample from the instantiation of concrete distribution when training
        epsilon sim  U(0,1), hat{e}_{ij} = sigma (frac{log epsilon-log (1-epsilon)+omega_{i j}}{tau})
        """
        if training:
            random_noise = torch.rand(log_alpha.shape)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def forward(self, inputs, training=None):
        x, embed, edge_index, tmp = inputs
        nodesize = embed.shape[0]
        feature_dim = embed.shape[1]
        f1 = embed.unsqueeze(1).repeat(1, nodesize, 1).reshape(-1, feature_dim)
        f2 = embed.unsqueeze(0).repeat(nodesize, 1, 1).reshape(-1, feature_dim)

        # using the node embedding to calculate the edge weight
        f12self = torch.cat([f1, f2], dim=-1)
        h = f12self.to(self.device)
        for elayer in self.elayers:
            h = elayer(h)
        values = h.reshape(-1)
        values = self.concrete_sample(values, beta=tmp, training=training)
        self.mask_sigmoid = values.reshape(nodesize, nodesize)

        # set the symmetric edge weights
        sym_mask = (self.mask_sigmoid + self.mask_sigmoid.transpose(0, 1)) / 2
        edge_mask = sym_mask[edge_index[0], edge_index[1]]

        self.__clear_masks__()
        self.__set_masks__(x, edge_index, edge_mask)

        outputs = self.get_model_output(x, edge_index)
        return outputs[1].squeeze(), edge_mask

    def get_model_output(self, x, edge_index, edge_mask=None):
        """ return the model outputs with or without (w/wo) edge mask  """
        self.model.eval()
        self.__clear_masks__()
        if edge_mask is not None:
            self.__set_masks__(x, edge_index, edge_mask.to(self.device))

        with torch.no_grad():
            if self.gen_dataset.is_multi():
                all_layer_embeddings = self.model.get_all_layer_embeddings(x, edge_index)
                emb = list(all_layer_embeddings.values())[-2]  # take the last convolutional layer before graph pooling
                prob = self.model.get_predictions(x, edge_index)
            else:
                # emb = self.model(x, edge_index)
                all_layer_embeddings = self.model.get_all_layer_embeddings(x, edge_index)
                # TODO emb = emb_ for node classification:
                #  whether layer activation is taken into account for .get_all_layer_embeddings?
                emb = list(all_layer_embeddings.values())[-1]
                # emb_ = self.model(x, edge_index)
                prob = self.model.get_predictions(x, edge_index)

        self.__clear_masks__()
        return emb, prob

    def get_subgraph(self, node_idx, x, edge_index, y, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        graph = to_networkx(data=Data(x=x, edge_index=edge_index), to_undirected=True)

        subset, edge_index, edge_index_original_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            node_idx, self.num_hops, edge_index, remap_edges=True,
            num_nodes=num_nodes, flow=self.__flow__())

        mapping = {int(v): k for k, v in enumerate(subset)}
        subgraph = graph.subgraph(subset.tolist())
        nx.relabel_nodes(subgraph, mapping)

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item
        y = y[subset]
        return x, edge_index, y, subset, edge_index_original_index

    def get_explanation_network(self, dataset, is_graph_classification=False):
        if os.path.isfile(self._ckpt_path):
            print("fetch network parameters from the saved files")
            state_dict = torch.load(self._ckpt_path)
            self.elayers.load_state_dict(state_dict)
            self.to(self.device)
        elif is_graph_classification:
            self.pbar.reset(total=self.epochs + 1)
            self.train_graph_classification_explanation_network(dataset)
        else:
            self.pbar.reset(total=self.epochs + 1)
            self.train_node_classification_explanation_network(dataset)

    def train_node_classification_explanation_network(self, data):
        optimizer = Adam(self.elayers.parameters(), lr=self.lr)
        dataset_indices = torch.where(self.gen_dataset.train_mask != 0)[0].tolist()

        # collect the embedding of nodes
        x_dict = {}
        edge_index_dict = {}
        node_idx_dict = {}
        emb_dict = {}
        pred_dict = {}
        with torch.no_grad():
            self.model.eval()
            for gid in dataset_indices:
                x, edge_index, y, subset, _ = \
                    self.get_subgraph(node_idx=gid, x=data.x, edge_index=data.edge_index, y=data.y)
                emb, prob = self.get_model_output(x, edge_index)
                # emb - net output without softmax;
                # prob - net output

                x_dict[gid] = x
                edge_index_dict[gid] = edge_index
                node_idx_dict[gid] = int(torch.where(torch.BoolTensor(subset == gid))[0])
                pred_dict[gid] = prob[node_idx_dict[gid]].argmax(-1).cpu()
                emb_dict[gid] = emb.data.cpu()

        # train the explanation network
        for epoch in range(self.epochs):
            loss = 0.0
            optimizer.zero_grad()
            tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
            self.elayers.train()
            for gid in tqdm(dataset_indices):
                pred, edge_mask = self.forward((x_dict[gid], emb_dict[gid], edge_index_dict[gid], tmp), training=True)
                loss_tmp = self.__loss__(pred[node_idx_dict[gid]], pred_dict[gid])
                loss_tmp.backward()
                loss += loss_tmp.item()

            optimizer.step()
            print(f'Epoch: {epoch} | Loss: {loss}')
            torch.save(self.elayers.cpu().state_dict(), self.ckpt_path)
            self.elayers.to(self.device)
            self.pbar.update(1)

    def train_graph_classification_explanation_network(self, dataset):
        optimizer = Adam(self.elayers.parameters(), lr=self.lr)
        dataset_indices = list(range(len(dataset)))

        # collect the embedding of nodes
        emb_dict = {}
        ori_pred_dict = {}
        with torch.no_grad():
            self.model.eval()
            for gid in tqdm(dataset_indices):
                data = dataset[gid]
                emb, prob = self.get_model_output(data.x, data.edge_index)
                emb_dict[gid] = emb.data.cpu()
                ori_pred_dict[gid] = prob.argmax(-1).data.cpu()

        # train the mask generator
        for epoch in range(self.epochs):
            loss = 0.0
            tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
            self.elayers.train()
            optimizer.zero_grad()
            for gid in tqdm(dataset_indices):
                data = dataset[gid]
                prob, _ = self.forward((data.x, emb_dict[gid], data.edge_index, tmp), training=True)
                loss_tmp = self.__loss__(prob, ori_pred_dict[gid])
                loss_tmp.backward()
                loss += loss_tmp.item()

            optimizer.step()
            print(f'Epoch: {epoch} | Loss: {loss}')
            torch.save(self.elayers.cpu().state_dict(), self.ckpt_path)
            self.elayers.to(self.device)
            self.pbar.update(1)

    def explain_edge_mask(self, x, edge_index):
        with torch.no_grad():
            # self.pbar.reset(total=100)
            emb, prob = self.get_model_output(x, edge_index)
            _, edge_mask = self.forward((x, emb, edge_index, 1.0), training=False)

        return edge_mask

    # TODO write run for PGExplainer multi
    @finalize_decorator
    def run(self, mode, kwargs, finalize=True):

        assert mode == "local"

        element_idx = kwargs.pop("element_idx")
        top_k = kwargs.pop("top_k")

        self.pbar.reset(1)

        if self.gen_dataset.is_multi():
            self.graph_idx = element_idx
            dataset = self.gen_dataset.dataset

            self.get_explanation_network(dataset, is_graph_classification=True)

            data = dataset[self.graph_idx]
            edge_mask = self.explain_edge_mask(data.x, data.edge_index)

            self.raw_explanation = {'edge_mask': edge_mask,
                                     'edge_index_original_index': data.edge_index,
                                     'top_k': top_k}

        else:
            node_idx = element_idx
            data = self.gen_dataset.data

            self.get_explanation_network(data, is_graph_classification=False)
            x, edge_index, y, subset, edge_index_original_index = self.get_subgraph(node_idx=node_idx,
                                                                                    x=data.x,
                                                                                    edge_index=data.edge_index,
                                                                                    y=data.y)
            edge_mask = self.explain_edge_mask(x, edge_index)

            self.raw_explanation = {'edge_mask': edge_mask,
                                     'edge_index_original_index': edge_index_original_index,
                                     'top_k': top_k}
        self.pbar.update(1)
        self.pbar.close()

    # TODO write finalize for PGExplainer multi
    def _finalize(self):
        mode = self._run_mode
        assert mode == "local"
        self.explanation = AttributionExplanation(
            local=mode, nodes="binary", edges="binary")

        important_edges = {}
        important_nodes = {}

        # TODO make top_k not in .run kwargs but in real time in front
        edge_mask = self.raw_explanation['edge_mask']
        edge_index_original_index = self.raw_explanation['edge_index_original_index']
        top_k = self.raw_explanation['top_k']

        assert edge_mask.size(0) == edge_index_original_index.size(1)

        _, important_edges = get_topk_edges_subgraph(edge_index_original_index,
                                                     edge_mask,
                                                     top_k=top_k,
                                                     un_directed=True)  # TODO how understand 'un_directed' using dataset?

        if self.gen_dataset.is_multi():
            important_edges = {self.graph_idx: important_edges}
            important_nodes = {self.graph_idx: important_nodes}

        # TODO Misha D. fix the rendering threshold on the front
        self.explanation.add_edges(important_edges)
        self.explanation.add_nodes(important_nodes)

        # print(important_edges)
        # print(important_nodes)
