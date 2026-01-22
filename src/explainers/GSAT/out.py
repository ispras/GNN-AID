from typing import Any, Literal

from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph

from data_structures.explanation import AttributionExplanation
from explainers.explainer import Explainer, finalize_decorator


class GSATExplainer(Explainer):
    name = 'GSAT'

    @staticmethod
    def check_availability(gen_dataset, model_manager):
        """ Availability check for the given dataset and model manager. """
        # return any(isinstance(m, MessagePassing) for m in model_manager.gnn.modules())
        return False

    def __init__(
            self,
            gen_dataset,
            model,
            device,
            expl_type: Literal['binary', 'continuous'] = 'continuous',
            thrsh: float = -1.0
    ):
        Explainer.__init__(self, gen_dataset, model)

        if hasattr(self.model, 'eval'):
            self.model.eval()
        self.device = device
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        self.expl_type = expl_type
        self.thrsh = thrsh

    @finalize_decorator
    def run(self, mode, kwargs, finalize=True):
        # assert self.gen_dataset.is_multi()
        assert mode == "local"
        idx = kwargs.pop('element_idx')

        self.pbar.reset(total=1)
        if self.gen_dataset.is_multi():
            raise NotImplementedError
        else:
            self.raw_explanation = self(
                self.gen_dataset.data.x, self.gen_dataset.data.edge_index,
                node_idx=idx, **kwargs)
        self.pbar.close()

    def get_filtered_k_hop_subgraph(self, node_idx, num_hops, edge_index, attention_scores, threshold):
        subset, edge_index_subgraph, mapping, edge_mask = k_hop_subgraph(
            node_idx, num_hops, edge_index, relabel_nodes=False
        )
        # attention_scores = (attention_scores - attention_scores.mean()) / attention_scores.std()  # normalize attention

        attention_scores = attention_scores[:edge_index.shape[1]]  # remove self-loop (need?)

        edge_index_new = edge_index[:, edge_mask]
        filtered_attention_scores = attention_scores[edge_mask]

        attention_mask = filtered_attention_scores > threshold

        filtered_edge_index = edge_index_new[:, attention_mask.squeeze()]

        return filtered_edge_index

    def __call__(self, x: Tensor, edge_index: Tensor, **kwargs) \
            -> Any:
        self.explained_node = node_idx = kwargs.get('node_idx')

        self.model.get_predictions(x, edge_index).squeeze()
        edge_att = getattr(self.model, self.model.gsat_layer_name).edge_att
        node_att = getattr(self.model, self.model.gsat_layer_name).node_att
        if self.expl_type == 'binary':
            raw_explanation = self.get_filtered_k_hop_subgraph(node_idx, self.model.get_num_hops(), edge_index, edge_att,
                                                               self.thrsh)
        else:
            subset, edge_index_subgraph, mapping, edge_mask = k_hop_subgraph(
                node_idx, self.model.get_num_hops(), edge_index, relabel_nodes=False
            )
            edge_dict = {}
            for i in range(edge_index.shape[1]):
                if not edge_mask[i]:
                    continue
                edge = (edge_index[0, i].item(), edge_index[1, i].item())
                att = edge_att[i].item()
                if  att > self.thrsh:  # Only include edges above threshold
                    edge_dict[edge] = att

            raw_explanation = {'edge_dict': edge_dict, 'node_att': node_att[subset]}  # no self-loops

        return raw_explanation

    def _finalize(self):
        mode = self._run_mode
        # assert mode == "global"
        if self.expl_type == 'binary':
            self.explanation = AttributionExplanation(local=mode, nodes="binary", edges="binary")

            edge_index = self.raw_explanation

            edges_values = {}
            for i in range(edge_index.shape[1]):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                edges_values[f"{src},{dst}"] = 1

            self.explanation.add_edges(edges_values)

            # Nodes
            nodes_values = {}
            for i in range(edge_index.shape[1]):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                nodes_values[src] = 1
                nodes_values[dst] = 1

            self.explanation.add_nodes(nodes_values)

            # Remove unpickable attributes
            self.pbar = None
        else:
            self.explanation = AttributionExplanation(local=mode, nodes="continuous", edges="continuous")

            edge_dict, node_att = self.raw_explanation['edge_dict'], self.raw_explanation['node_att']

            edges_values = {}
            for k, v in edge_dict.items():
                src = k[0]
                dst = k[1]
                edges_values[f"{src},{dst}"] = v

            self.explanation.add_edges(edges_values)

            # Nodes
            nodes_values = {}
            if node_att.shape[0] == 1:
                nodes_values[0] = node_att.item()
            else:
                for i, val in enumerate(node_att.squeeze().tolist()):
                    if val ** 0.5 > self.thrsh:
                        nodes_values[i] = val

            self.explanation.add_nodes(nodes_values)
