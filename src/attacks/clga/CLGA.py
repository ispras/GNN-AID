from typing import Tuple
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dropout_adj
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

from attacks.clga.differentiable_models.gcn import GCN
from attacks.clga.differentiable_models.model import GRACE
from attacks.poison_attacks import PoisonAttacker
from base.datasets_processing import GeneralDataset
from models_builder.models_utils import apply_decorator_to_graph_layers


class CLGAAttack(PoisonAttacker):
    name = "CLGAAttack"

    def __init__(
            self,
            num_nodes: int = None,
            feature_shape: int = None,
            learning_rate: float = 0.01,
            num_hidden: int = 256,
            num_proj_hidden: int = 32,
            activation: str = "prelu",
            drop_edge_rate_1: float = 0.3,
            drop_edge_rate_2: float = 0.4,
            tau: float = 0.4,
            num_epochs: int = 3000,
            weight_decay: float = 1e-5,
            drop_scheme: str = "degree",
            device: str = "cpu"
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_shape = feature_shape
        self.learning_rate = learning_rate
        self.num_hidden = num_hidden
        self.num_proj_hidden = num_proj_hidden
        self.activation = activation
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.tau = tau
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.drop_scheme = drop_scheme
        self.device = device

        self.modified_adj = None
        self.model = None
        self.optimizer = None

    def _init(self, gen_dataset: GeneralDataset):
        """ Init extra parameters when dataset is known.
        """
        self.num_nodes = gen_dataset.data.x.shape[0]
        self.feature_shape = gen_dataset.num_node_features

    @staticmethod
    def drop_edge(
            edge_index: torch.Tensor,
            drop_prob: float
    ) -> torch.Tensor:
        """
        Perform edge dropout based on the chosen scheme.
        """
        return dropout_adj(edge_index, p=drop_prob)[0]

    def train_gcn(
            self,
            data
    ) -> None:
        """
        Train the GCN model with augmented graphs.
        """
        self.model.train()
        self.optimizer.zero_grad()
        edge_index_1 = self.drop_edge(data.edge_index, self.drop_edge_rate_1)
        edge_index_2 = self.drop_edge(data.edge_index, self.drop_edge_rate_2)
        x_1 = data.x.clone()
        x_2 = data.x.clone()

        z1 = self.model(x_1, edge_index_1)
        z2 = self.model(x_2, edge_index_2)

        loss = self.model.loss(z1, z2)
        loss.backward()
        self.optimizer.step()

    def compute_gradient(
            self,
            data
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Compute gradients of the contrastive loss w.r.t. adjacency matrix.
        """
        self.model.eval()
        edge_index_1 = self.drop_edge(data.edge_index, self.drop_edge_rate_1)
        edge_index_2 = self.drop_edge(data.edge_index, self.drop_edge_rate_2)

        size_1 = edge_index_1.shape[1]
        size_2 = edge_index_2.shape[1]

        # adj_dense_1 = torch.sparse.FloatTensor(
        #     edge_index_1, torch.ones(edge_index_1.shape[1], device=self.device),
        #     (self.num_nodes, self.num_nodes)
        # ).to_dense().requires_grad_(True)
        #
        # adj_dense_2 = torch.sparse.FloatTensor(
        #     edge_index_2, torch.ones(edge_index_2.shape[1], device=self.device),
        #     (self.num_nodes, self.num_nodes)
        # ).to_dense().requires_grad_(True)

        # z1 = self.model(data.x, adj_dense_1)
        # z2 = self.model(data.x, adj_dense_2)

        z1 = self.model(data.x, edge_index_1)
        z2 = self.model(data.x, edge_index_2)

        # edge_index_1.requires_grad = True
        # edge_index_2.requires_grad = True

        loss = self.model.loss(z1, z2)
        loss.backward()

        # grad = torch.zeros_like()
        grad = 0
        for name, layer in self.model.encoder.named_children():
            if isinstance(layer, MessagePassing):
                #print(f"{name}: {layer.get_message_gradients()}")
                for l_name, l_grad in layer.get_message_gradients().items():
                    grad = l_grad

        max_size = max(size_1, size_2)
        max_edge = edge_index_1 if size_1 > size_2 else edge_index_2
        return grad, max_size, max_edge

    def attack(
            self,
            gen_dataset
    ) -> None:
        """
        Execute the CLGA attack.
        """
        self._init(gen_dataset)
        self.model = GRACE(
            encoder=GCN(self.feature_shape, self.num_hidden, 'prelu'),
            num_hidden=self.num_hidden,
            num_proj_hidden=self.num_proj_hidden,
            tau=self.tau
        ).to(self.device)

        apply_decorator_to_graph_layers(self.model)
        apply_decorator_to_graph_layers(self.model.encoder)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        perturbed_edges = [[], []]

        # adj = torch.sparse.FloatTensor(
        #     gen_dataset.dataset.data.edge_index, torch.ones(gen_dataset.dataset.data.edge_index.shape[1], device=self.device),
        #     (self.num_nodes, self.num_nodes)
        # ).to_dense()

        adj = to_dense_adj(gen_dataset.dataset.data.edge_index).squeeze()

        edge_index_set = set((int(x), int(y)) for x, y in
                              zip(gen_dataset.dataset.data.edge_index[0], gen_dataset.dataset.data.edge_index[1]))

        for epoch in tqdm(range(self.num_epochs)):
            self.train_gcn(gen_dataset.dataset.data)

            # grad_1, grad_2 = self.compute_gradient(gen_dataset.dataset.data)
            # grad_sum = grad_1 + grad_2
            grad_sum, max_edge, edge_index_mutated = self.compute_gradient(gen_dataset.dataset.data)
            grad_sum = grad_sum.sum(axis=1)
            grad_sum = grad_sum[:max_edge]

            max_grad_index = torch.argmax(torch.abs(grad_sum))
            # row, col = divmod(max_grad_index.item(), self.num_nodes)
            i = int(edge_index_mutated[0, max_grad_index])
            j = int(edge_index_mutated[1, max_grad_index])

            if (i, j) in edge_index_set:
                if grad_sum[max_grad_index] <= 0:
                    perturbed_edges[0].append(i)
                    perturbed_edges[1].append(j)
            else:
                if grad_sum[max_grad_index] > 0:
                    perturbed_edges[0].append(i)
                    perturbed_edges[1].append(j)

            # if grad_sum[row, col] > 0 and adj[row, col] == 0:
            #     adj[row, col] = 1
            #     adj[col, row] = 1
            # elif grad_sum[row, col] < 0 and adj[row, col] == 1:
            #     adj[row, col] = 0
            #     adj[col, row] = 0

            # perturbed_edges.append((row, col))
            #gen_dataset.dataset.data.edge_index = dense_to_sparse(adj)[0]

        gen_dataset.dataset.data.edge_index = torch.tensor(perturbed_edges)
