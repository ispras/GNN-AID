from typing import Optional
import torch
import numpy as np
# from explainers.explainer_results_to_json import ProtExplanationGlobal
from torch_geometric.nn import GMMConv

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
