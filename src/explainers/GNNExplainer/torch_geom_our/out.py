from math import sqrt
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.nn import MessagePassing
from torch_geometric.explain import ExplainerConfig, Explanation, ModelConfig
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel

from torch_geometric.explain import Explainer as torchExplainerRunner

from explainers.explainer import Explainer, finalize_decorator
from explainers.explanation import AttributionExplanation


class GNNExplainer(Explainer):

    name = 'GNNExplainer(torch-geom)'
    availability_profile = ({'single', 'multi'}, {'modules', 'get_num_hops', 'forward'})

    @staticmethod
    def check_availability(gen_dataset, model_manager):
        """ Availability check for the given dataset and model manager. """
        # Should have at least 1 MessagePassing module
        return\
            {'modules', 'get_num_hops', 'forward'}.issubset(dir(model_manager.gnn)) and\
            any(isinstance(m, MessagePassing) for m in model_manager.gnn.modules())

    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self,
                 gen_dataset,
                 model,
                 device,
                 epochs: int = 100,
                 lr: float = 0.01,
                 node_mask_type: str = 'object',
                 edge_mask_type: str = 'object',
                 mode: str = 'multiclass_classification',
                 return_type: str = 'log_probs',
                 **kwargs):
        Explainer.__init__(self, gen_dataset, model)

        self.coeffs.update(kwargs)

        self.epochs = epochs
        self.node_mask_type = node_mask_type
        self.edge_mask_type = edge_mask_type
        task_level = 'graph' if gen_dataset.is_multi() else 'node'

        self.graph_idx = None
        self.node_idx = None
        self.x = None
        self.edge_index = None
        self.num_classes = gen_dataset.num_classes

        self.explainer = torchExplainerRunner(
            model=model,
            algorithm=GNNExplainerAlgorithm(None, epochs=epochs, lr=lr, kwargs=self.coeffs),
            explanation_type='model',
            node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type,
            model_config=dict(
                mode=mode,
                task_level=task_level,
                return_type=return_type,
            ),
        )

    @finalize_decorator
    def run(self, mode, kwargs, finalize=True):
        assert mode == "local"
        idx = kwargs.pop('element_idx')

        if self.gen_dataset.is_multi():
            self.graph_idx = idx
            graph = self.gen_dataset.dataset[self.graph_idx]
            self.x = graph.x
            self.edge_index = graph.edge_index
        else:
            self.node_idx = idx
            self.x = self.gen_dataset.data.x
            self.edge_index = self.gen_dataset.data.edge_index

        self.pbar.reset(total=self.epochs, mode=mode)
        self.explainer.algorithm.pbar = self.pbar
        if self.gen_dataset.is_multi():
            self.raw_explanation = self.explainer(self.x, self.edge_index)
        else:
            self.raw_explanation = self.explainer(self.x, self.edge_index, index=self.node_idx)
        self.pbar.close()

    def _finalize(self):
        mode = self._run_mode
        assert mode == "local"

        edge_mask = self.raw_explanation.edge_mask
        node_mask = self.raw_explanation.node_mask

        self.explanation = AttributionExplanation(
            local=mode,
            edges="continuous" if self.edge_mask_type=="object" else False,
            nodes="continuous" if self.node_mask_type=="object" else False,
            features="continuous" if self.node_mask_type=="common_attributes" else False)

        important_edges = {}
        important_nodes = {}
        important_features = {}

        if self.edge_mask_type is not None and self.node_mask_type is not None:

            # Multi graphs check is not needed: the explanation format for
            # graph classification and node classification is the same
            eps = 0.001

            # Edges
            if self.edge_mask_type=="object":
                num_edges = edge_mask.size(0)
                assert num_edges == self.edge_index.size(1)
                edges = self.edge_index

                for i in range(num_edges):
                    imp = float(edge_mask[i])
                    if not imp < eps:
                        edge = edges[0][i], edges[1][i]
                        important_edges[f"{edge[0]},{edge[1]}"] = format(imp, '.4f')
            else:  # if "common_attributes" or "attributes"
                raise NotImplementedError(f"Edge mask type '{self.edge_mask_type}' is not yet implemented.")

            # Nodes
            if self.node_mask_type=="object":
                num_nodes = node_mask.size(0)
                assert num_nodes == self.x.size(0)

                for i in range(num_nodes):
                    imp = float(node_mask[i][0])
                    if not imp < eps:
                        important_nodes[i] = format(imp, '.4f')
            # Features
            elif self.node_mask_type=="common_attributes":
                num_features = node_mask.size(1)
                assert num_features == self.x.size(1)

                for i in range(num_features):
                    imp = float(node_mask[0][i])
                    if not imp < eps:
                        important_features[i] = format(imp, '.4f')
            else:  # if "attributes"
                # TODO add functional if node_mask_type=="attributes"
                raise NotImplementedError(f"Node mask type '{self.node_mask_type}' is not yet implemented.")

        if self.gen_dataset.is_multi():
            important_edges = {self.graph_idx: important_edges}
            important_nodes = {self.graph_idx: important_nodes}
            important_features = {self.graph_idx: important_features}

        # TODO Write functions with output threshold
        self.explanation.add_edges(important_edges)
        self.explanation.add_nodes(important_nodes)
        self.explanation.add_features(important_features)

        # print(important_edges)
        # print(important_nodes)


class GNNExplainerAlgorithm(ExplainerAlgorithm):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and node features that play a crucial role in the predictions
    made by a GNN.

    .. note::

        For an example of using :class:`GNNExplainer`, see
        `examples/explain/gnn_explainer.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer.py>`_,
        `examples/explain/gnn_explainer_ba_shapes.py <https://github.com/
        pyg-team/pytorch_geometric/blob/master/examples/
        explain/gnn_explainer_ba_shapes.py>`_, and `examples/explain/
        gnn_explainer_link_pred.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer_link_pred.py>`_.

    Args:
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.GNNExplainer.coeffs`.
    """

    def __init__(self, pbar, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs = kwargs['kwargs']

        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None

        self.pbar = pbar

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        self._train(model, x, edge_index, target=target, index=index, **kwargs)

        node_mask = self._post_process_mask(
            self.node_mask,
            self.hard_node_mask,
            apply_sigmoid=True,
        )
        edge_mask = self._post_process_mask(
            self.edge_mask,
            self.hard_edge_mask,
            apply_sigmoid=True,
        )

        self._clean_model(model)

        return Explanation(node_mask=node_mask, edge_mask=edge_mask)

    def supports(self) -> bool:
        return True

    def _train(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        self._initialize_masks(x, edge_index)

        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters.append(self.edge_mask)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for i in range(self.epochs):
            optimizer.zero_grad()

            h = x if self.node_mask is None else x * self.node_mask.sigmoid()
            y_hat, y = model(h, edge_index, **kwargs), target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)

            loss.backward()
            optimizer.step()

            # In the first iteration, we collect the nodes and edges that are
            # involved into making the prediction. These are all the nodes and
            # edges with gradient != 0 (without regularization applied).
            if i == 0 and self.node_mask is not None:
                if self.node_mask.grad is None:
                    raise ValueError("Could not compute gradients for node "
                                     "features. Please make sure that node "
                                     "features are used inside the model or "
                                     "disable it via `node_mask_type=None`.")
                self.hard_node_mask = self.node_mask.grad != 0.0
            if i == 0 and self.edge_mask is not None:
                if self.edge_mask.grad is None:
                    raise ValueError("Could not compute gradients for edges. "
                                     "Please make sure that edges are used "
                                     "via message passing inside the model or "
                                     "disable it via `edge_mask_type=None`.")
                self.hard_edge_mask = self.edge_mask.grad != 0.0

            self.pbar.update(1)
        self.pbar.close()

    def _initialize_masks(self, x: Tensor, edge_index: Tensor):
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        device = x.device
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        if node_mask_type is None:
            self.node_mask = None
        elif node_mask_type == MaskType.object:
            self.node_mask = Parameter(torch.randn(N, 1, device=device) * std)
        elif node_mask_type == MaskType.attributes:
            self.node_mask = Parameter(torch.randn(N, F, device=device) * std)
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask = Parameter(torch.randn(1, F, device=device) * std)
        else:
            assert False

        if edge_mask_type is None:
            self.edge_mask = None
        elif edge_mask_type == MaskType.object:
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.edge_mask = Parameter(torch.randn(E, device=device) * std)
        else:
            assert False

    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)
        else:
            assert False

        if self.hard_edge_mask is not None:
            assert self.edge_mask is not None
            m = self.edge_mask[self.hard_edge_mask].sigmoid()
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['edge_ent'] * ent.mean()

        if self.hard_node_mask is not None:
            assert self.node_mask is not None
            m = self.node_mask[self.hard_node_mask].sigmoid()
            node_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
            loss = loss + self.coeffs['node_feat_size'] * node_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None
