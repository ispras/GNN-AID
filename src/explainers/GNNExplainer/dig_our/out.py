import torch
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch_geometric.utils.loop import add_remaining_self_loops
from torch_geometric.nn import MessagePassing

from dig.version import debug
from dig.xgraph.models.utils import subgraph

from explainers.GNNExplainer.dig_our.utils.base_explainer import ExplainerBase

from explainers.explainer import Explainer, finalize_decorator
from data_structures.explanation import AttributionExplanation
from typing import Union


EPS = 1e-15


def cross_entropy_with_logit(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    return cross_entropy(y_pred, y_true.long(), **kwargs)


class GNNExplainer(Explainer, ExplainerBase):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-predictions.
    .. note:: For an example, see `benchmarks/xgraph
        <https://github.com/divelab/DIG/tree/dig/benchmarks/xgraph>`_.
    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        explain_graph (bool, optional): Whether to explain graph classification model
            (default: :obj:`False`)
    """
    name = 'GNNExplainer(dig)'

    @staticmethod
    def check_availability(gen_dataset, model_manager):
        """ Availability check for the given dataset and model manager. """
        # Should have at least 1 MessagePassing module
        return ({'modules', 'flow', 'get_num_hops', 'parameters', 'forward'}.issubset(dir(model_manager.gnn)) and
                any(isinstance(m, MessagePassing) for m in model_manager.gnn.modules()))

    def __init__(self,
                 gen_dataset,
                 model: torch.nn.Module,
                 device,
                 epochs: int = 100,
                 lr: float = 0.01,
                 coff_size: float = 0.001,
                 coff_ent: float = 0.001):
        # TODO can we use device=my_device ?
        ExplainerBase.__init__(self, model, epochs, lr, explain_graph=gen_dataset.is_multi())
        self.table = None  # NOTE: remove unpickable attribute, it is useless for us
        Explainer.__init__(self, gen_dataset, model)
        self.coff_ent = coff_ent
        self.coff_size = coff_size

        self.graph_idx = None

    def __loss__(self, raw_preds: Tensor, x_label: Union[Tensor, int]):
        if self.explain_graph:
            loss = cross_entropy_with_logit(raw_preds, x_label)
        else:
            loss = cross_entropy_with_logit(raw_preds[self.node_idx].reshape(1, -1), x_label)
        m = self.edge_mask.sigmoid()
        loss = loss + self.coff_size * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coff_ent * ent.mean()

        if self.mask_features:
            m = self.node_feat_mask.sigmoid()
            loss = loss + self.coeffs['node_feat_size'] * m.sum()
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def gnn_explainer_alg(self,
                          x: Tensor,
                          edge_index: Tensor,
                          ex_label: Tensor,
                          mask_features: bool = False,
                          **kwargs
                          ) -> Tensor:

        # initialize a mask
        self.to(x.device)
        self.mask_features = mask_features

        # train to get the mask
        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)

        for epoch in range(1, self.epochs + 1):

            if mask_features:
                h = x * self.node_feat_mask.view(1, -1).sigmoid()
            else:
                h = x
            # edge_index_1 = edge_index
            raw_preds = self.model(x=h, edge_index=edge_index, **kwargs)

            loss = self.__loss__(raw_preds, ex_label)
            if epoch % 20 == 0 and debug:
                print(f'Loss:{loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=2.0)
            optimizer.step()
            self.pbar.update(1)

        return self.edge_mask

    def forward(self, mask_features=False, **kwargs):
        r"""
        Run the explainer for a specific graph instance.
        Args:
            x (torch.Tensor): The graph instance's input node features.
            edge_index (torch.Tensor): The graph instance's edge index.
            mask_features (bool, optional): Whether to use feature mask. Not recommended.
                (Default: :obj:`False`)
            **kwargs (dict):
                :obj:`node_idx` （int): The index of node that is pending to be explained.
                (for node classification)
                :obj:`sparsity` (float): The Sparsity we need to control to transform a
                soft mask to a hard mask. (Default: :obj:`0.7`)
        :rtype: (None, list, list)
        .. note::
            (None, edge_masks, related_predictions):
            edge_masks is a list of edge-level explanation for each class;
            related_predictions is a list of dictionary for each class
            where each dictionary includes 4 type predicted probabilities.
        """
        # TODO make idx = kwargs.get('idx') for node and graph explain
        idx = kwargs.get('element_idx')

        if self.gen_dataset.is_multi():
            self.graph_idx = idx
            dataset = self.gen_dataset
            graph = dataset.dataset[self.graph_idx]
            x = graph.x
            edge_index = graph.edge_index
            num_classes = dataset.num_classes
            self.num_nodes = x.size(0)
        else:
            node_idx = idx
            x = self.gen_dataset.data.x
            edge_index = self.gen_dataset.data.edge_index
            num_classes = self.gen_dataset.num_classes

        super().forward(x=x, edge_index=edge_index, **kwargs)
        if hasattr(self.model, 'eval'):
            self.model.eval()

        # добавляет к каждой вершине петлю
        self_loop_edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=self.num_nodes)

        # Only operate on a k-hop subgraph around `node_idx`.
        # Get subgraph and relabel the node, mapping is the relabeled given node_idx.
        if not self.explain_graph:
            node_idx = torch.tensor(node_idx)
            if not node_idx.dim():
                node_idx = node_idx.reshape(-1)
            node_idx = node_idx.to(self.device)
            assert node_idx is not None
            # subset - граф второй окрестности для исходной вершины
            # (напр: node_idx = 2377 соседи: только вершина [372], у 372 соседи [626, 1710, 1834, 2377])
            # => subset = [372, 626, 1710, 1834, 2377]
            self.subset, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, self_loop_edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.model.flow())
            self.node_idx = node_idx
            self.new_node_idx = torch.where(self.subset == node_idx)[0]

        if kwargs.get('edge_masks'):
            edge_masks = kwargs.pop('edge_masks')
            self.__set_masks__(x, self_loop_edge_index)
        else:
            # Assume the mask we will predict
            labels = tuple(i for i in range(num_classes))
            ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)

            # Calculate mask
            edge_masks = []
            for ex_label in ex_labels:
                self.__clear_masks__()
                self.__set_masks__(x, self_loop_edge_index)
                edge_masks.append(self.gnn_explainer_alg(x, edge_index, ex_label))

        hard_edge_masks = [self.control_sparsity(mask, sparsity=kwargs.get('sparsity')).sigmoid()
                           for mask in edge_masks]

        with torch.no_grad():
            related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks, **kwargs)

        self.__clear_masks__()

        return edge_masks, hard_edge_masks, related_preds

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    @finalize_decorator
    def run(self, mode, kwargs, finalize=True):
        assert mode == "local"
        idx = kwargs.pop("element_idx")
        self.pbar.reset(total=self.epochs * self.gen_dataset.num_classes)
        self.raw_explanation = self(element_idx=idx, **kwargs)
        self.pbar.close()

    def _finalize(self):
        mode = self._run_mode
        assert mode == "local"
        self.explanation = AttributionExplanation(
            local=mode, nodes="binary", edges="binary")

        if self.gen_dataset.is_multi():
            # raise NotImplementedError
            # TODO Misha S - what should be here?
            # pred = [self.gnn(data.x, data.edge_index).argmax(-1).item()
            #         for data in self.gen_dataset.dataset]
            # pred = [self.model.get_answer(data.x, data.edge_index).item()
            #         for data in self.gen_dataset.dataset]
            dataset = self.gen_dataset
            graph = dataset.dataset[self.graph_idx]
            x = graph.x
            edge_index = graph.edge_index
            pred = self.model.get_answer(x, edge_index).item()
        else:
            # pred = self.model(self.gen_dataset.data.x, self.gen_dataset.data.edge_index)[
            #     self.explained_node].argmax(-1).item()
            pred = self.model.get_answer(self.gen_dataset.data.x, self.gen_dataset.data.edge_index)[
                self.node_idx].item()

        # Get important
        if self.gen_dataset.is_multi():
            _, hard_edge_masks, _ = self.raw_explanation
            edge_node_mask = list(map(int, hard_edge_masks[pred]))
            edges = self.gen_dataset.dataset[self.graph_idx].edge_index
            num_edges = len(edges[0])
        else:
            _, hard_edge_masks, _ = self.raw_explanation
            edge_node_mask = list(map(int, hard_edge_masks[pred]))
            edges = self.gen_dataset.data.edge_index
            num_edges = len(edges[0])

        # Assign edges, then nodes)
        if self.gen_dataset.is_multi():
            important_edges = {}
            important_nodes = {}

            for i in range(num_edges):
                imp = edge_node_mask[i]
                if imp != 0:
                    edge = edges[0][i], edges[1][i]
                    important_edges[f"{edge[0]},{edge[1]}"] = imp

            for i in range(len(edge_node_mask) - num_edges):
                imp = edge_node_mask[i + num_edges]
                if imp != 0:
                    important_nodes[i] = imp

        else:  # single graph
            important_edges = {}
            important_nodes = {}
            for i in range(num_edges):
                imp = edge_node_mask[i]
                if imp != 0:
                    edge = edges[0][i], edges[1][i]
                    important_edges[f"{edge[0]},{edge[1]}"] = imp

            for i in range(len(edge_node_mask) - num_edges):
                imp = edge_node_mask[i + num_edges]
                if imp != 0:
                    important_nodes[i] = imp

        if self.gen_dataset.is_multi():
            important_edges = {self.graph_idx: important_edges}
            important_nodes = {self.graph_idx: important_nodes}

        self.explanation.add_edges(important_edges)
        self.explanation.add_nodes(important_nodes)

        # Remove unpickable attributes
        self.pbar = None
