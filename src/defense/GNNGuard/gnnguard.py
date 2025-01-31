import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv

# from defense.GNNGuard.base_model import BaseModel
from defense.poison_defense import PoisonDefender

from models_builder.gnn_models import FrameworkGNNModelManager
from models_builder.gnn_constructor import FrameworkGNNConstructor
from models_builder.models_zoo import model_configs_zoo
from aux.configs import ModelManagerConfig, ModelModificationConfig, DatasetConfig, DatasetVarConfig, ConfigPattern
from aux.utils import import_by_name, CUSTOM_LAYERS_INFO_PATH, MODULES_PARAMETERS_PATH, hash_data_sha256, \
    TECHNICAL_PARAMETER_KEY, IMPORT_INFO_KEY, OPTIMIZERS_PARAMETERS_PATH


import warnings
import types
# from torch_sparse import coalesce, SparseTensor, matmul

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from scipy.sparse import lil_matrix
import scipy.sparse as sp

import numpy as np

from src.aux.configs import ModelConfig


# class BaseGCN(BaseModel):

#     def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01,
#                 with_bn=False, weight_decay=5e-4, with_bias=True, device=None):

#         super(BaseGCN, self).__init__()

#         assert device is not None, "Please specify 'device'!"
#         self.device = device

#         self.layers = nn.ModuleList([])
#         if with_bn:
#             self.bns = nn.ModuleList()

#         if nlayers == 1:
#             self.layers.append(GCNConv(nfeat, nclass, bias=with_bias))
#         else:
#             self.layers.append(GCNConv(nfeat, nhid, bias=with_bias))
#             if with_bn:
#                 self.bns.append(nn.BatchNorm1d(nhid))
#             for i in range(nlayers-2):
#                 self.layers.append(GCNConv(nhid, nhid, bias=with_bias))
#                 if with_bn:
#                     self.bns.append(nn.BatchNorm1d(nhid))
#             self.layers.append(GCNConv(nhid, nclass, bias=with_bias))

#         self.dropout = dropout
#         self.weight_decay = weight_decay
#         self.lr = lr
#         self.output = None
#         self.best_model = None
#         self.best_output = None
#         self.with_bn = with_bn
#         self.name = 'GCN'

#     def forward(self, x, edge_index, edge_weight=None):
#         x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
#         for ii, layer in enumerate(self.layers):
#             if edge_weight is not None:
#                 adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()
#                 x = layer(x, adj)
#             else:
#                 x = layer(x, edge_index)
#             if ii != len(self.layers) - 1:
#                 if self.with_bn:
#                     x = self.bns[ii](x)
#                 x = F.relu(x)
#                 x = F.dropout(x, p=self.dropout, training=self.training)
#         return F.log_softmax(x, dim=1)

#     def get_embed(self, x, edge_index, edge_weight=None):
#         x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
#         for ii, layer in enumerate(self.layers):
#             if ii == len(self.layers) - 1:
#                 return x
#             if edge_weight is not None:
#                 adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()
#                 x = layer(x, adj)
#             else:
#                 x = layer(x, edge_index)
#             if ii != len(self.layers) - 1:
#                 if self.with_bn:
#                     x = self.bns[ii](x)
#                 x = F.relu(x)
#         return x

#     def initialize(self):
#         for m in self.layers:
#             m.reset_parameters()
#         if self.with_bn:
#             for bn in self.bns:
#                 bn.reset_parameters()

class GNNGuard(PoisonDefender):
    name = 'GNNGuard'

    def __init__(self, model=None, lr=0.01, train_iters=100, attention=False, drop=False, device='cpu', with_bias=False, with_relu=False):
        super().__init__()
        assert device is not None, "Please specify 'device'!"
        self.model = model
        self.with_bias = with_bias
        self.with_relu = with_relu
        self.attention = attention
        self.lr = lr
        self.device = device
        self.drop = drop
        self.train_iters = train_iters
        self.droplearn = nn.Linear(2, 1)
        self.beta = nn.Parameter(torch.rand(1))

    def defense(self, gen_dataset):
        super().defense(gen_dataset=gen_dataset)
        if self.model is None:
            self.model = model_configs_zoo(gen_dataset, 'gcn_gcn_linearized')
            default_config = ModelModificationConfig(
                model_ver_ind=0,
            )
            manager_config = ConfigPattern(
                _config_class="ModelManagerConfig",
                _config_kwargs={
                    "mask_features": [],
                    "optimizer": {
                        "_config_class": "Config",
                        "_class_name": "Adam",
                        "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                        "_class_import_info": ["torch.optim"],
                        "_config_kwargs": {"weight_decay": 5e-4},
                    }
                }
            )
            gnn_model_manager_surrogate = FrameworkGNNModelManager(
                gnn=self.model,
                dataset_path=gen_dataset,
                modification=default_config,
                manager_config=manager_config,
            )
            gnn_model_manager_surrogate.train_model(gen_dataset=gen_dataset, steps=self.train_iters)
            # self.pred_labels = gnn_model_manager_surrogate.run_model(gen_dataset=gen_dataset, mask='all', out='answers')
            self.gnn = gnn_model_manager_surrogate.gnn
        else:
            self.gnn = self.model
        # self.embeddings = self.gnn.get_all_layer_embeddings()


        edge_weights_mem = None
        # print(self.model.forward)
        # print(gen_dataset.data, flush=True)
        # print(gen_dataset.data.edge_index)
        # print(gen_dataset.num_node_features)

        x = gen_dataset.data.x
        edge_index = gen_dataset.data.edge_index
        batch = gen_dataset.data.batch

        self.embeddings = self.get_embeddings(x, edge_index, batch)
        # print(self.embeddings.keys())
        embeddings_count = len(self.embeddings)

        for k in range(-1, embeddings_count-1):
            adj_index, adj_value = self.att_coef(gen_dataset, k=k)

            if edge_weights_mem is None:
                edge_weights = (1 - self.beta) * adj_value
            else:
                edge_weights = self.beta * edge_weights_mem + (1 - self.beta) * adj_value
            edge_weights_mem = edge_weights
            gen_dataset.data.edge_index = adj_index
            gen_dataset.data.edge_weights = adj_value
            # print(adj_value)
        return gen_dataset

    def att_coef(self, gen_dataset, k=-1):
        x = gen_dataset.data.x
        edge_index = gen_dataset.data.edge_index
        batch = gen_dataset.data.batch

        n_node = gen_dataset.data.num_nodes

        fea = self.embeddings[k]

        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]

        fea_copy = fea.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
        sim = sim_matrix[row, col]
        sim[sim<0.1] = 0
        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim

        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')

        """add learnable dropout, make character vector"""
        if self.drop:
            character = np.vstack((att_dense_norm[row, col].A1,
                                    att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T)
            drop_score = self.droplearn(character)
            drop_score = torch.sigmoid(drop_score)
            mm = torch.nn.Threshold(-0.5, 0)
            drop_score = -mm(-drop_score)
            drop_decision = drop_score
            drop_decision = drop_score.clone().requires_grad_()

            drop_mask = lil_matrix((n_node, n_node), dtype=np.float32)
            # print(drop_mask)
            drop_mask[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            drop_mask = drop_mask.tocsr()

            att_dense_norm = att_dense_norm.multiply(drop_mask)  # update, remove the 0 edges

        if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
            degree = (att_dense_norm != 0).sum(1).A1
            lam = 1 / (degree + 1) # degree +1 is to add itself
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight  # add the self loop
        else:
            att = att_dense_norm
        att_adj = np.vstack((row, col))
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)   # exponent, kind of softmax
        # print(np.array(att_edge_weight)[0])
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32)#.cuda()
        att_adj = torch.tensor(att_adj, dtype=torch.int64)#.cuda()

        shape = (n_node, n_node)
        new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape)
        # print(new_adj)
        return (new_adj._indices(), new_adj._values())

    def get_embeddings(self, x, edge_index, batch):
        self.embeddings = self.gnn.get_all_layer_embeddings(x, edge_index, batch)
        self.embeddings[-1] = x
        return self.embeddings


class GuardWrapper(nn.Module):
    def __init__(self, model, drop, attention):
        super(GuardWrapper, self).__init__()
        self.model = model
        self.drop = drop
        self.attention = attention

        # Override the forward method directly
        self.model.forward = self.new_forward

        # Other dynamic attributes
        self.model.att_coef = types.MethodType(self.att_coef, self.model)
        setattr(self.model, "drop", self.drop)
        setattr(self.model, "attention", self.attention)
        setattr(self.model, "gate", nn.Parameter(torch.rand(1)))
        setattr(self.model, "droplearn_1", nn.Linear(2, 1))
        setattr(self.model, "droplearn_2", nn.Linear(2, 1))

        self.model.droplearn_1.reset_parameters()
        self.model.droplearn_2.reset_parameters()

    def new_forward(self, *args, **kwargs):
        """we don't change the edge_index, just update the edge_weight;
            some edge_weight are regarded as removed if it equals to zero"""
        layer_ind = -1
        tensor_storage = {}
        dim_cat = 0
        layer_emb_dict = {}
        save_emb_flag = self._save_emb_flag

        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        print(edge_index)
        feat = x
        adj = edge_index.tocoo()
        adj_memory = None
        # print(list(self.__dict__['_modules'].items()))
        for elem in list(self.__dict__['_modules'].items()):
            layer_name, curr_layer_ind = elem[0].split('_')
            if layer_name=="droplearn":
                continue
            curr_layer_ind = int(curr_layer_ind)
            inp = torch.clone(x)
            loc_flag = False
            if curr_layer_ind != layer_ind:
                if save_emb_flag:
                    loc_flag = True
                zeroing_x_flag = False
                for key, value in self.conn_dict.items():
                    if key[0] == layer_ind and layer_ind not in tensor_storage:
                        tensor_storage[layer_ind] = torch.clone(x)
                layer_ind = curr_layer_ind
                x_copy = torch.clone(x)
                connection_tensor = torch.Tensor()
                for key, value in self.conn_dict.items():

                    if key[1] == curr_layer_ind:
                        if key[1] - key[0] == 1:
                            zeroing_x_flag = True
                        for con in self.conn_dict[key]:
                            if self.embedding_levels_by_layers[key[1]] == 'n' and \
                                    self.embedding_levels_by_layers[key[0]] == 'n':
                                connection_tensor = torch.cat((connection_tensor,
                                                            tensor_storage[key[0]]), 1)
                                dim_cat = 1
                            elif self.embedding_levels_by_layers[key[1]] == 'g' and \
                                    self.embedding_levels_by_layers[key[0]] == 'g':
                                connection_tensor = torch.cat((connection_tensor,
                                                            tensor_storage[key[0]]), 0)
                                dim_cat = 0
                            elif self.embedding_levels_by_layers[key[1]] == 'g' and \
                                    self.embedding_levels_by_layers[key[0]] == 'n':
                                con_pool = import_by_name(con['pool']['pool_type'],
                                                        ["torch_geometric.nn"])
                                tensor_after_pool = con_pool(tensor_storage[key[0]], batch)
                                connection_tensor = torch.cat((connection_tensor,
                                                            tensor_after_pool), 1)
                                dim_cat = 1
                            else:
                                raise Exception(
                                    "Connection from layer type "
                                    f"{self.embedding_levels_by_layers[curr_layer_ind - 1]} to"
                                    f" layer type {self.embedding_levels_by_layers[curr_layer_ind]}"
                                    "is not supported now")


                if zeroing_x_flag:
                    x = connection_tensor
                else:
                    x = torch.cat((x_copy, connection_tensor), dim_cat)


                if self.attention:
                    if layer_name == 'GINConv':
                        if adj_memory is None:
                            adj = self.att_coef(x, adj, is_lil=False,i=layer_ind)
                            edge_index = adj._indices()
                            edge_weight = adj._values()
                            adj_memory = adj
                        elif adj_memory is not None:
                            adj = self.att_coef(x, adj_memory, is_lil=False, i=layer_ind)
                            edge_weight = self.gate * adj_memory._values() + (1 - self.gate) * adj._values()
                            adj_memory = adj
                    elif layer_name == 'GCNConv' or layer_name == 'GATConv':
                        if adj_memory is None:
                            adj = self.att_coef(x, adj, i=0)
                            edge_index = adj._indices()
                            edge_weight = adj._values()
                            adj_memory = adj
                        elif adj_memory is not None:
                            adj = self.att_coef(x, adj_memory, i=layer_ind).to_dense()
                            row, col = adj.nonzero()[:,0], adj.nonzero()[:,1]
                            edge_index = torch.stack((row, col), dim=0)
                            edge_weight = adj[row, col]
                            adj_memory = adj
                else:
                    edge_index = adj._indices()
                    edge_weight = adj._values()


            # QUE Kirill, maybe we should not off UserWarning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                # mid = x
                if layer_name in self.modules_info:
                    code_str = f"getattr(self, elem[0])" \
                                f"({self.modules_info[layer_name][TECHNICAL_PARAMETER_KEY]['forward_parameters']}," \
                                f" edge_weight=edge_weight)"
                    x = eval(f"{code_str}")
                else:
                    x = getattr(self, elem[0])(x)
            if loc_flag:
                layer_emb_dict[layer_ind] = torch.clone(x)

        if save_emb_flag:
            return layer_emb_dict
        return x

    def att_coef(self, fea, edge_index, is_lil=False, i=0):
        print(edge_index)
        if is_lil == False:
            edge_index = edge_index._indices()
        else:
            edge_index = edge_index.tocoo()

        n_node = fea.shape[0]
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]

        fea_copy = fea.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
        sim = sim_matrix[row, col]
        sim[sim<0.1] = 0

        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')


# if __name__ == "__main__":
#     from deeprobust.graph.data import Dataset, Dpr2Pyg
#     # from deeprobust.graph.defense import GCN
#     data = Dataset(root='/tmp/', name='citeseer', setting='prognn')
#     adj, features, labels = data.adj, data.features, data.labels
#     idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
#     model = GCN(nfeat=features.shape[1],
#           nhid=16,
#           nclass=labels.max().item() + 1,
#           dropout=0.5, device='cuda')
#     model = model.to('cuda')
#     pyg_data = Dpr2Pyg(data)[0]

#     # model.fit(features, adj, labels, idx_train, train_iters=200, verbose=True)
#     # model.test(idx_test)

#     from utils import get_dataset
#     pyg_data = get_dataset('citeseer', True, if_dpr=False)[0]

#     import ipdb
#     ipdb.set_trace()

#     model.fit(pyg_data, verbose=True) # train with earlystopping
#     model.test()
#     print(model.predict())