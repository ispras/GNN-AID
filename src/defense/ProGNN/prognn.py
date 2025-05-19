from typing import Iterable, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.sparse
from torch.optim.optimizer import required
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm
import torch.nn.functional as F

from src.base.datasets_processing import GeneralDataset
from src.models_builder.models_zoo import model_configs_zoo

from defense.poison_defense import PoisonDefender


def feature_smoothing(
        adj: torch.Tensor,
        X: torch.Tensor
) -> torch.Tensor:
    adj = (adj.T + adj) / 2
    rowsum = adj.sum(1)
    r_inv = rowsum.flatten()
    D = torch.diag(r_inv)
    L = D - adj

    r_inv = r_inv + 1e-3
    r_inv = r_inv.pow(-1 / 2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    L = r_mat_inv @ L @ r_mat_inv

    XLXT = torch.matmul(torch.matmul(X.t(), L), X)
    loss_smooth_feat = torch.trace(XLXT)
    return loss_smooth_feat


class ProGNNDefender(PoisonDefender):
    name = 'ProGNNDefender'

    # TODO re-write with support of sparse matrix

    def __init__(self,
                 symmetric: bool,
                 lr_adj: float,
                 alpha: float,
                 beta: float,
                 epochs: int,
                 lambda_coef: float,
                 phi: float,
                 surr_epochs: int,
                 surr_lr: float,
                 weight_decay: float,
                 data_steps: int,
                 model_steps: int,
                 **kw
                 ) -> None:
        super().__init__()
        self.symmetric = symmetric
        self.lr_adj = lr_adj
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs
        self.lambda_coef = lambda_coef
        self.phi = phi
        self.surr_epochs = surr_epochs
        self.surr_lr = surr_lr
        self.weight_decay = weight_decay
        self.data_steps = data_steps
        self.model_steps = model_steps

    def defense(
            self,
            gen_dataset: GeneralDataset,
            **kw
    ) -> GeneralDataset:

        features = gen_dataset.dataset.data.x
        labels = gen_dataset.dataset.data.y
        idx_train = gen_dataset.dataset.train_mask
        # adj = to_torch_coo_tensor(gen_dataset.dataset.data.edge_index)
        adj = to_dense_adj(gen_dataset.dataset.data.edge_index).squeeze(0)

        self.device = gen_dataset.dataset.data.x.device

        self.model = model_configs_zoo(dataset=gen_dataset, model_name='gcn_gcn')
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.surr_lr, weight_decay=self.weight_decay)

        self.estimator = EstimateAdj(adj, symmetric=self.symmetric).to(self.device)

        self.optimizer_adj = optim.SGD(self.estimator.parameters(),
                                       momentum=0.9, lr=self.lr_adj)

        self.optimizer_l1 = PGDOptimizer(self.estimator.parameters(),
                                         proxs=[prox_l1],
                                         lr=self.lr_adj, alphas=[self.alpha])

        self.optimizer_nuclear = PGDOptimizer(self.estimator.parameters(),
                                              proxs=[prox_nuclear],
                                              lr=self.lr_adj, alphas=[self.beta])

        # Train model
        for epoch in tqdm(range(self.epochs)):
            for i in range(self.data_steps):
                self.train_adj_one_epoch(epoch, features, adj)
            for i in range(self.model_steps):
                self.train_gcn_one_epoch(features, labels, idx_train)

        gen_dataset.dataset.data.edge_index = dense_to_sparse(adj)[0]
        return gen_dataset

    def train_gcn_one_epoch(
            self,
            features: torch.Tensor,
            labels: torch.Tensor,
            idx_train: torch.Tensor
    ) -> None:
        adj = self.estimator.normalize()

        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(features, dense_to_sparse(adj)[0])
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        self.optimizer.step()

    def train_adj_one_epoch(
            self,
            epoch: int,
            features: torch.Tensor,
            adj: torch.Tensor
    ) -> None:
        self.estimator.train()
        self.optimizer_adj.zero_grad()

        loss_fro = torch.norm(self.estimator.estimated_adj - adj, p='fro')

        if self.lambda_coef:
            loss_smooth_feat = feature_smoothing(self.estimator.estimated_adj, features)
        else:
            loss_smooth_feat = 0

        loss_symmetric = torch.norm(self.estimator.estimated_adj - self.estimator.estimated_adj.t(), p="fro")

        loss_diffirential = loss_fro + self.lambda_coef * loss_smooth_feat + self.phi * loss_symmetric

        loss_diffirential.backward()

        self.optimizer_adj.step()
        if self.beta != 0:
            self.optimizer_nuclear.zero_grad()
            self.optimizer_nuclear.step()

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        # feature smoothing
        self.estimator.estimated_adj.data.copy_(torch.clamp(
            self.estimator.estimated_adj.data, min=0, max=1))


def prox_l1(
        data: torch.Tensor,
        alpha: float
) -> torch.Tensor:
    """Proximal operator for l1 norm with sparse tensor support.
    """
    # if not data.is_sparse:
    #     raise ValueError("Input data must be a sparse tensor.")
    #
    # values = data.values()
    # indices = data.indices()
    #
    # prox_values = torch.sign(values) * torch.clamp(torch.abs(values) - alpha, min=0)
    #
    # return torch.sparse_coo_tensor(indices, prox_values, data.size())
    data = torch.mul(torch.sign(data), torch.clamp(torch.abs(data) - alpha, min=0))
    return data


def prox_nuclear(
        data: torch.Tensor,
        alpha: float,
) -> torch.Tensor:
    """Proximal operator for nuclear norm (trace norm).
    """
    device = data.device
    U, S, V = np.linalg.svd(data.cpu())
    U, S, V = torch.FloatTensor(U).to(device), torch.FloatTensor(S).to(device), torch.FloatTensor(V).to(device)

    diag_S = torch.diag(torch.clamp(S - alpha, min=0))
    return torch.matmul(torch.matmul(U, diag_S), V)


class PGDOptimizer(torch.optim.Optimizer):
    """Proximal gradient descent.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining parameter groups
    proxs : iterable
        iterable of proximal operators
    alpha : iterable of coefficients for proximal gradient descent
    lr : float
        learning rate
    momentum : float
        momentum factor (default: 0)
    weight_decay : float
        weight decay (L2 penalty) (default: 0)
    dampening : float
        dampening for momentum (default: 0)
    """

    def __init__(
            self,
            params: Iterable,
            proxs: Iterable,
            alphas: Iterable[float],
            lr=required,
            momentum=0,
            dampening=0,
            weight_decay=0
    ):
        defaults = dict(lr=lr, momentum=0, dampening=0,
                        weight_decay=0, nesterov=False)

        super(PGDOptimizer, self).__init__(params, defaults)

        for group in self.param_groups:
            group.setdefault('proxs', proxs)
            group.setdefault('alphas', alphas)

    def __setstate__(
            self,
            state: dict[str, Any]
    ) -> None:
        super(PGDOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(
            self,
            delta: float = 0,
            closure: Optional[float] = None
    ) -> None:
        for group in self.param_groups:
            lr = group['lr']
            proxs = group['proxs']
            alphas = group['alphas']

            # apply the proximal operator to each parameter in a group
            for param in group['params']:
                for prox_operator, alpha in zip(proxs, alphas):
                    param.data = prox_operator(param.data, alpha=alpha * lr)


class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(
            self,
            adj: torch.Tensor,
            symmetric: bool = False,
    ) -> None:
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = adj.device

    def _init_estimation(
            self,
            adj: torch.Tensor
    ) -> None:
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self) -> torch.Tensor:
        return self.estimated_adj

    def normalize(self) -> torch.Tensor:

        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t()) / 2
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(
            self,
            mx: torch.Tensor
    ) -> torch.Tensor:
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx
