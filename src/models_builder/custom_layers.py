from typing import Optional
import torch
import numpy as np
from IPython.testing.decorators import skip_if_no_x11
from torch import nn
from torch.utils import checkpoint
# from explainers.explainer_results_to_json import ProtExplanationGlobal
from torch_geometric.nn import GMMConv, InstanceNorm
from torch_geometric.utils import is_undirected, sort_edge_index
from torch_sparse import transpose
from src.models_builder.models_utils import apply_attention_to_messages

from explainers.protgnn.MCTS import mcts
import ctypes


class GMM(GMMConv):
    def __init__(self, in_channels, out_channels: int, dim: int, kernel_size: int,
                 **kwargs):
        super().__init__(in_channels, out_channels, dim, kernel_size, **kwargs)

    def forward(self, x, edge_index, edge_attr=None, size=None):
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.size(dim=1), self.dim)
        out = super().forward(x=x, edge_index=edge_index, edge_attr=edge_attr, size=size)
        return out


class PrototypeGraph:
    def __init__(self, base_graph: int = 0, coalition=None):
        if coalition is None:
            coalition = []
        self.base_graph = base_graph
        self.coalition = coalition


class ProtLayer(torch.nn.Module):
    def __init__(self, full_gnn_id, layer_name_in_gnn, in_features, num_classes,
                 num_prototypes_per_class=3, eps=1e-4):
        super().__init__()
        full_gnn = ctypes.cast(full_gnn_id, ctypes.py_object).value
        full_gnn.prot_layer_name = layer_name_in_gnn
        self.num_prototypes_per_class = num_prototypes_per_class
        self.eps = eps
        self.output_dim = num_classes
        self.prototype_shape = (num_classes * num_prototypes_per_class, in_features)
        self.prototype_vectors = torch.nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.prototype_graphs = [PrototypeGraph() for _ in range(num_classes * self.num_prototypes_per_class)]
        self.num_prototypes = self.prototype_shape[0]
        self.last_layer = torch.nn.Linear(self.num_prototypes, num_classes, bias=False)  # do not use bias!

        assert (self.num_prototypes % num_classes == 0)
        # one-hot indication matrix for eac prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes, num_classes)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1
        # initialize last layer
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

    def reset_parameters(self):
        self.last_layer.reset_parameters()
        self.prototype_vectors = torch.nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

    def forward(self, x, full_gnn_id):
        prototype_activations, min_distances = self.prototype_distances(x=x, )
        logits = self.last_layer(prototype_activations)
        full_gnn = ctypes.cast(full_gnn_id, ctypes.py_object).value
        full_gnn.min_distances = min_distances
        return logits

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        """
        The incorrect strength will be actual strength if -0.5 then input -0.5
        """
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def prototype_distances(self, x):
        xp = torch.mm(x, torch.t(self.prototype_vectors))
        distance = -2 * xp + torch.sum(x ** 2, dim=1, keepdim=True) + torch.t(
            torch.sum(self.prototype_vectors ** 2, dim=1, keepdim=True))
        similarity = torch.log((distance + 1) / (distance + self.eps))
        return similarity, distance

    def prototype_subgraph_distances(self, x, prototype):
        distance = torch.norm(x - prototype, p=2, dim=1, keepdim=True) ** 2
        similarity = torch.log((distance + 1) / (distance + self.eps))
        return similarity, distance

    def projection(self, gnn, dataset, data_indices, data, thrsh=10):
        best_graph = None
        best_coalition = None
        gnn.eval()
        for i in range(self.num_prototypes_per_class * self.output_dim):
            count = 0
            best_similarity = 0
            label = i // self.num_prototypes_per_class
            proj_prot = None
            # for j in range(i * 10, len(data_indices)):
            # for j in range(len(data_indices)):
            for j in np.random.permutation(data_indices):
                # graph = dataset[data_indices[j]]
                graph = dataset[j]
                # if dataset.name.lower() == 'graph-sst2':
                #     words = dataset.supplement['sentence_tokens'][str(data_indices[j])]
                if graph.y[0] == label:
                    count += 1
                    coalition, similarity, prot = mcts(graph, gnn, self.prototype_vectors[i])
                    if similarity >= best_similarity:
                        best_similarity = similarity
                        proj_prot = prot
                        # best_data = data
                        best_coalition = coalition
                        # best_words = words
                        # best_graph = data_indices[j]
                        best_graph = j
                if count >= thrsh:
                    break
            if proj_prot is not None:
                self.prototype_vectors.data[i] = proj_prot
                self.prototype_graphs[i] = PrototypeGraph(best_graph, best_coalition)
                print('Projection of prototype completed')
            else:
                print('No objects of specific class in dataset. Projection of prototype failed!')  # To be warning?

    def result_prototypes(self, best_prots: [Optional] = None, best: bool = False, ):
        """
        saving projected prototypes
        """
        # we need weights of last layer in explanation, so we get tensor data without grad and memory pointer
        return self.output_dim, self.num_prototypes_per_class, self.last_layer.weight.data.tolist(), best_prots \
            if best else self.prototype_graphs


class GSATLayer(torch.nn.Module):
    def __init__(
            self,
            full_gnn_id,
            layer_name_in_gnn,
            in_features: int = 16,
            learn_edge_features: bool = False,
            extractor_dropout_p: float = 0.5,
    ):
        super().__init__()
        self.is_inside = False
        self.learn_edge_features = learn_edge_features
        self.extractor = ExtractorMLP(in_features, learn_edge_features, extractor_dropout_p)
        full_gnn = ctypes.cast(full_gnn_id, ctypes.py_object).value
        full_gnn.gsat_layer_name = layer_name_in_gnn

        self.hook_handle = self.register_forward_pre_hook(self.save_input_hook, with_kwargs=True)
        self.handlers = None

    def save_input_hook(self, module, args, kwargs):
        x = kwargs['x']
        if isinstance(x, dict):
            for k, v in x.items():
                if k.startswith("skip"):
                    pass
                else:
                    x = v
        # self.hook_saved_value = x.detach()
        self.hook_saved_value = x.clone()

    def forward(self, x, edge_index, full_gnn_id):
        if isinstance(x, dict):
            for k, v in x.items():
                if k.startswith("skip"):
                    skip_x = v
                else:
                    x = v

        # return x

        if self.is_inside:
            return x
            # return self.extractor(x, edge_index)
        else:
            self.is_inside = True

            emb = x  # layer assume that node/edge embs are passed into
            att_log_logits = self.extractor(emb, edge_index)
            att = self.sampling(att_log_logits, training=True)

            if self.learn_edge_features:
                if is_undirected(edge_index):
                    trans_idx, trans_val = transpose(edge_index, att, None, None, coalesced=False)
                    trans_val_perm = GSATLayer.reorder_like(trans_idx, edge_index, trans_val)
                    edge_att = (att + trans_val_perm) / 2
                else:
                    edge_att = att
            else:
                edge_att = self.lift_node_att_to_edge_att(att, edge_index)

            self.edge_att = edge_att  # for explanation
            self.node_att = att  # for explanation
            # loss = self.gsat_loss(att, clf_logits, batch.y, self.modification.epochs)

            full_gnn = ctypes.cast(full_gnn_id, ctypes.py_object).value

            # full_gnn.gsat_attention = self.edge_att
            if self.handlers is not None:
                for h in self.handlers:
                    h.remove()
                self.handlers = None

            self.handlers = apply_attention_to_messages(full_gnn, self.edge_att)

            # checkpoint.checkpoint(full_gnn, skip_x, edge_index, att)
            full_gnn(skip_x, edge_index, edge_att=self.edge_att)

            if self.handlers is not None:
                for h in self.handlers:
                    h.remove()
                self.handlers = None

            # full_gnn(skip_x, edge_index)

            x = self.hook_saved_value
            self.hook_saved_value = None

            self.is_inside = False

            return x

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    def sampling(self, att_log_logits, training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("Edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att

        # Calculate self-loop attention
        self_loop_att = node_att * node_att

        # Concatenate self-loop attention at the end
        edge_att = torch.cat((edge_att, self_loop_att), dim=0)

        return edge_att


class ExtractorMLP(nn.Module):

    def __init__(
            self,
            hidden_size,
            learn_edge_features: bool = False,
            extractor_dropout_p: float = 0.5,
    ):
        super().__init__()
        self.learn_edge_att = learn_edge_features
        dropout_p = extractor_dropout_p

        if self.learn_edge_att:
            self.feature_extractor = GSATMLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = GSATMLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12)
        else:
            att_log_logits = self.feature_extractor(emb)
        return att_log_logits


class GSATMLP(nn.Sequential):
    # TODO check if specific MLP needed
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(GSATMLP, self).__init__(*m)

    def forward(self, inputs):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                if inputs.shape[0] == 1:  # TODO sort of monkey-patch
                    continue
                inputs = module(inputs)
            else:
                inputs = module(inputs)
        return inputs


class DummyLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        return x
