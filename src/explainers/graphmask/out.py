import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.loop import add_remaining_self_loops
from dig.version import debug
from dig.xgraph.models.utils import subgraph
from torch.nn.functional import cross_entropy
from typing import Union

from explainers.gnnexplainer.dig_our.utils.base_explainer import ExplainerBase
from explainers.explanation import AttributionExplanation
from explainers.graphmask.torch_utils import LagrangianOptimization
from explainers.explainer import Explainer, finalize_decorator

EPS = 1e-15


def cross_entropy_with_logit(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    return cross_entropy(y_pred, y_true.long(), **kwargs)


class GraphMaskExplainer(Explainer, ExplainerBase):
    name = "GraphMask"

    @staticmethod
    def check_availability(gen_dataset, model_manager):
        """ Availability check for the given dataset and model manager. """
        # Should have at least 1 MessagePassing module
        return\
            not gen_dataset.is_multi() and\
            {'modules', 'flow', 'get_num_hops', 'parameters', 'forward'}.issubset(dir(model_manager.gnn)) and\
            any(isinstance(m, MessagePassing) for m in model_manager.gnn.modules())

    def __init__(self,
                 gen_dataset,
                 model: torch.nn.Module,
                 device,
                 epochs: int = 100,
                 lr: float = 0.01,
                 coff_size: float = 0.001,
                 coff_ent: float = 0.001,
                 allowance: float = 0.03):
        # TODO can we use device=my_device ?
        ExplainerBase.__init__(self, model, epochs, lr, explain_graph=gen_dataset.is_multi())
        self.table = None  # NOTE: remove unpickable attribute, it is useless for us
        Explainer.__init__(self, gen_dataset, model)
        # self.coff_ent = coff_ent
        self.coff_size = coff_size
        self.allowance = allowance

    def __loss__(self, raw_preds: Tensor, x_label: Union[Tensor, int]):
        if self.explain_graph:
            loss = cross_entropy_with_logit(raw_preds, x_label)
        else:
            loss = cross_entropy_with_logit(raw_preds[self.node_idx].reshape(1, -1), x_label)
        # m = self.edge_mask.sigmoid()
        # loss = loss + self.coff_size * m.sum()
        # ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        # loss = loss + self.coff_ent * ent.mean()

        # if self.mask_features:
        #     m = self.node_feat_mask.sigmoid()
        #     loss = loss + self.coeffs['node_feat_size'] * m.sum()
        #     ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        #     loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def graph_mask(self,
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
        lagrangian_optimization = LagrangianOptimization(optimizer)
        for epoch in range(1, self.epochs + 1):

            if mask_features:
                h = x * self.node_feat_mask.view(1, -1).sigmoid()
            else:
                h = x
            raw_preds = self.model(x=h, edge_index=edge_index, **kwargs)
            loss = self.__loss__(raw_preds, ex_label)
            g = torch.relu(loss - self.allowance).mean()
            s = torch.sigmoid(self.edge_mask)
            mask_ent = -s * torch.log(s) - (1 - s) * torch.log(1 - s)
            penalty = mask_ent.mean() + 0.5 * s.sum()
            f = penalty * self.coff_size
            lagrangian_optimization.update(f, g)

            if epoch % 20 == 0 and debug:
                print(f'Loss:{loss.item()}')
            # optimizer.zero_grad()
            # loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=2.0)
            # optimizer.step()
            self.pbar.update(1)

        return self.edge_mask

    @finalize_decorator
    def run(self, mode, kwargs, finalize=True):
        assert mode == "local"
        assert not self.gen_dataset.is_multi()
        idx = kwargs.pop('element_idx')
        self.pbar.reset(total=self.epochs * self.gen_dataset.num_classes)
        self.raw_explanation = self(node_idx=idx, **kwargs)
        self.pbar.close()

    def forward(self, mask_features=False, **kwargs):
        x = self.gen_dataset.data.x
        edge_index = self.gen_dataset.data.edge_index
        num_classes = self.gen_dataset.num_classes

        super().forward(x=x, edge_index=edge_index, **kwargs)
        if hasattr(self.model, 'eval'):
            self.model.eval()

        self_loop_edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=self.num_nodes)

        # Only operate on a k-hop subgraph around `node_idx`.
        # Get subgraph and relabel the node, mapping is the relabeled given node_idx.
        if not self.explain_graph:
            node_idx = torch.tensor(kwargs.get('node_idx'))
            if not node_idx.dim():
                node_idx = node_idx.reshape(-1)
            node_idx = node_idx.to(self.device)
            assert node_idx is not None
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
                edge_masks.append(self.graph_mask(x, edge_index, ex_label))

        hard_edge_masks = [self.control_sparsity(mask, sparsity=kwargs.get('sparsity')).sigmoid()
                           for mask in edge_masks]

        with torch.no_grad():
            related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks, **kwargs)

        self.__clear_masks__()

        return edge_masks, hard_edge_masks, related_preds

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def _finalize(self):
        mode = self._run_mode
        assert mode == "local"
        self.explanation = AttributionExplanation(
            local=not self.gen_dataset.is_multi(), nodes="binary", edges="binary")

        if self.gen_dataset.is_multi():
            raise NotImplementedError
            # TODO Danil - what should be here?
            # pred = [self.gnn(data.x, data.edge_index).argmax(-1).item()
            #         for data in self.gen_dataset.dataset]
            # pred = [self.model.get_answer(data.x, data.edge_index).item()
            #         for data in self.gen_dataset.dataset]
        else:
            # pred = self.gnn(self.gen_dataset.data.x, self.gen_dataset.data.edge_index)[
            #     self.node_idx].argmax(-1).item()
            pred = self.model.get_answer(self.gen_dataset.data.x, self.gen_dataset.data.edge_index)[
                self.node_idx].item()

        # Get important
        _, hard_edge_masks, _ = self.raw_explanation
        edge_node_mask = list(map(int, hard_edge_masks[pred]))
        edges = self.gen_dataset.data.edge_index
        num_edges = len(edges[0])

        # Assign edges, then nodes
        if self.gen_dataset.is_multi():
            important_edges = []
            important_nodes = []

            eix = self.gen_dataset.dataset.slices['edge_index']
            graph_ix = -1
            for i in range(num_edges):
                if i >= eix[graph_ix + 1]:
                    important_edges.append({})
                    graph_ix += 1
                imp = edge_node_mask[i]
                if imp != 0:
                    edge = edges[0][i], edges[1][i]
                    important_edges[graph_ix][f"{edge[0]},{edge[1]}"] = imp

            nix = self.gen_dataset.dataset.slices['x']
            graph_ix = -1
            node_ix = 0
            for i in range(len(edge_node_mask) - num_edges):
                if i >= nix[graph_ix + 1]:
                    important_nodes.append({})
                    graph_ix += 1
                    node_ix = 0
                imp = edge_node_mask[i + num_edges]
                if imp != 0:
                    important_nodes[graph_ix][node_ix] = imp
                node_ix += 1

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
        self.explanation.add_edges(important_edges)
        self.explanation.add_nodes(important_nodes)

        # Remove unpickable attributes
        self.pbar = None
