import math
import torch
import numpy as np
import scipy.sparse as sp
import attacks.metattack.utils as utils
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch import optim
from tqdm import tqdm
from models_builder.gnn_models import FrameworkGNNModelManager
from models_builder.models_zoo import model_configs_zoo
from aux.configs import ModelModificationConfig, ConfigPattern
from aux.utils import OPTIMIZERS_PARAMETERS_PATH
from torch_geometric.utils import dense_to_sparse

from attacks.poison_attacks import PoisonAttacker


class BaseMeta(PoisonAttacker):
    name = "BaseMeta"

    """
    Super class for Metattack on GNNs
    Parameters
    ----------
    model:
        surrogate model that will be attacked directly
    num_nodes : int
        number of nodes in the input graph
    train_iters : int
        number of initial training iterations for surrogate model
    attack_iters: int
        number of training iterations for surrogate model for meta-gradient calc
    lambda_ : float
        lambda_ is used to weight the two objectives in Eq. (10) in the paper.
    lr: float
        learning rate for surrogate meta-training
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    undirected : bool
        whether the graph is undirected
    device: str
        'cpu' or 'cuda'
    """

    def __init__(self, num_nodes=None, feature_shape=None, lambda_=0.5, train_iters=200, attack_iters=100, lr=0.1,
                 attack_structure=True, attack_features=False, undirected=False, device='cpu'):
        super().__init__()
        self.model = None
        self.num_nodes = num_nodes
        self.feature_shape = feature_shape
        self.lambda_ = lambda_
        self.train_iters = train_iters
        self.attack_iters = attack_iters
        self.lr = lr
        self.device = device

        self.attack_structure = attack_structure
        self.attack_features = attack_features
        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.modified_adj = None
        self.modified_features = None

        if attack_structure:
            self.undirected = undirected
            assert num_nodes is not None, 'Num_nodes should be given'
            self.adj_changes = Parameter(torch.FloatTensor(num_nodes, num_nodes))
            self.adj_changes.data.fill_(0)

        if attack_features:
            assert feature_shape is not None, 'Feature_shape should be given'
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.feature_changes.data.fill_(0)

    def attack(self, gen_dataset):
        # TODO model choice by user to be implemented.
        #  note: kind of sophisticated task

        # Initial surrogate model training

        # from torch_geometric.data import Data
        # Data().get('adj_t')
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

        self.pred_labels = gnn_model_manager_surrogate.run_model(gen_dataset=gen_dataset, mask='all', out='answers')


    def get_modified_adj(self, ori_adj):
        adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
        if self.undirected:
            adj_changes_square = adj_changes_square + torch.transpose(adj_changes_square, 1, 0)
        adj_changes_square = torch.clamp(adj_changes_square, -1, 1)
        modified_adj = adj_changes_square + ori_adj
        return modified_adj

    def get_modified_features(self, ori_features):
        return ori_features + self.feature_changes

    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.
        """

        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        if self.undirected:
            l_and = l_and + l_and.t()
        flat_mask = 1 - l_and
        return flat_mask

    def self_training_label(self, labels, idx_train):
        # Predict the labels of the unlabeled nodes to use them for self-training.
        output = self.pred_labels
        # labels_self_training = output.argmax(1)
        labels_self_training = self.pred_labels.long().clone().detach()
        labels_self_training[idx_train] = labels[idx_train]
        return labels_self_training


    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        """
        Computes a mask for entries that, if the edge corresponding to the entry is added/removed, would lead to the
        log likelihood constraint to be violated.

        Note that different data type (float, double) can effect the final results.
        """
        t_d_min = torch.tensor(2.0).to(self.device)
        if self.undirected:
            t_possible_edges = np.array(np.triu(np.ones((self.num_nodes, self.num_nodes)), k=1).nonzero()).T
        else:
            t_possible_edges = np.array((np.ones((self.num_nodes, self.num_nodes)) - np.eye(self.num_nodes)).nonzero()).T
        allowed_mask, current_ratio = utils.likelihood_ratio_filter(t_possible_edges,
                                                                    modified_adj,
                                                                    ori_adj, t_d_min,
                                                                    ll_cutoff, undirected=self.undirected)
        return allowed_mask, current_ratio

    def get_adj_score(self, adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff):
        adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
        # Make sure that the minimum entry is 0.
        adj_meta_grad = adj_meta_grad - adj_meta_grad.min()
        # Filter self-loops
        adj_meta_grad = adj_meta_grad - torch.diag(torch.diag(adj_meta_grad, 0))
        # # Set entries to 0 that could lead to singleton nodes.
        singleton_mask = self.filter_potential_singletons(modified_adj)
        adj_meta_grad = adj_meta_grad *  singleton_mask

        if ll_constraint:
            allowed_mask, self.ll_ratio = self.log_likelihood_constraint(modified_adj, ori_adj, ll_cutoff)
            allowed_mask = allowed_mask.to(self.device)
            adj_meta_grad = adj_meta_grad * allowed_mask
        return adj_meta_grad

    def get_feature_score(self, feature_grad, modified_features):
        feature_meta_grad = feature_grad * (-2 * modified_features + 1)
        feature_meta_grad -= feature_meta_grad.min()
        return feature_meta_grad

    # def train_surrogate(self, gen_dataset, initialize=True):
    #     if initialize:
    #         pass

    def reset_parameters(self):
        pass

class MetaAttackFull(BaseMeta):
    """
    Attack GNNs with meta gradients
    """
    name = "MetaAttackFull"

    def __init__(self, num_nodes=None, feature_shape=None, lambda_=0.5, train_iters=200, attack_iters=100, lr=0.1,
                 momentum=0.9, attack_structure=True, attack_features=False, undirected=False, device='cpu',
                 with_bias=False, with_relu=False):
        super().__init__(num_nodes=num_nodes, feature_shape=feature_shape, lambda_=lambda_, train_iters=train_iters,
                         attack_iters=attack_iters, lr=lr, attack_features=attack_features,
                         attack_structure=attack_structure, undirected=undirected, device=device)
        self.with_bias = with_bias
        self.with_relu = with_relu

        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []
        self.momentum = momentum

    def attack(self, gen_dataset, attack_budget=10, ll_constraint=True, ll_cutoff=0.004):
        super().attack(gen_dataset=gen_dataset)

        self.hidden_sizes = [16]   # FIXME get from model architecture
        self.nfeat = gen_dataset.num_node_features
        self.nclass = gen_dataset.num_classes

        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(self.device))
            w_velocity = torch.zeros(weight.shape).to(self.device)
            self.weights.append(weight)
            self.w_velocities.append(w_velocity)

            if self.with_bias:
                bias = Parameter(torch.FloatTensor(nhid).to(self.device))
                b_velocity = torch.zeros(bias.shape).to(self.device)
                self.biases.append(bias)
                self.b_velocities.append(b_velocity)

            previous_size = nhid

        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass).to(self.device))
        output_w_velocity = torch.zeros(output_weight.shape).to(self.device)
        self.weights.append(output_weight)
        self.w_velocities.append(output_w_velocity)

        if self.with_bias:
            output_bias = Parameter(torch.FloatTensor(self.nclass).to(self.device))
            output_b_velocity = torch.zeros(output_bias.shape).to(self.device)
            self.biases.append(output_bias)
            self.b_velocities.append(output_b_velocity)

        self._initialize()

        ori_features = gen_dataset.dataset.data.x
        ori_adj = gen_dataset.dataset.data.edge_index
        labels = gen_dataset.dataset.data.y
        idx_train = gen_dataset.train_mask
        idx_unlabeled = gen_dataset.test_mask

        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)

        labels_self_training = self.self_training_label(labels, idx_train)
        modified_adj = ori_adj
        modified_features = ori_features

        for i in tqdm(range(attack_budget), desc="Perturbing graph"):
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)

            if self.attack_features:
                modified_features = ori_features + self.feature_changes

            adj_norm = utils.normalize_adj_tensor(modified_adj)
            self.inner_train(modified_features, adj_norm, idx_train, idx_unlabeled, labels)

            adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)

            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)
            if self.attack_structure:
                adj_meta_score = self.get_adj_score(adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(feature_grad, modified_features)

            if adj_meta_score.max() >= feature_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)
                row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
                self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
                if self.undirected:
                    self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            else:
                feature_meta_argmax = torch.argmax(feature_meta_score)
                row_idx, col_idx = utils.unravel_index(feature_meta_argmax, ori_features.shape)
                self.feature_changes.data[row_idx][col_idx] += (-2 * modified_features[row_idx][col_idx] + 1)

        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()

        gen_dataset.dataset.data.edge_index = dense_to_sparse(self.modified_adj.int())[0]
        print("TEST")

    def _initialize(self):
        for w, v in zip(self.weights, self.w_velocities):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            v.data.fill_(0)

        if self.with_bias:
            for b, v in zip(self.biases, self.b_velocities):
                stdv = 1. / math.sqrt(w.size(1))
                b.data.uniform_(-stdv, stdv)
                v.data.fill_(0)

    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        self._initialize()

        for ix in range(len(self.hidden_sizes) + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.w_velocities[ix].requires_grad = True

            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True
                self.b_velocities[ix] = self.b_velocities[ix].detach()
                self.b_velocities[ix].requires_grad = True

        for j in range(self.attack_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b

                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

    def get_meta_grad(self, features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training):

        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)

        output = F.log_softmax(hidden, dim=1)

        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
        loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])

        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        print('GCN loss on unlabled data: {}'.format(loss_test_val.item()))
        print('GCN acc on unlabled data: {}'.format(
            utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))
        print('attack loss: {}'.format(attack_loss.item()))

        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]
        return adj_grad, feature_grad


class MetaAttackApprox(BaseMeta):
    """
    Attack GNNs with approximate meta gradients
    """
    name = "MetaAttackApprox"

    def __init__(self, num_nodes=None, feature_shape=None, attack_structure=True, attack_features=False,
                 undirected=False, device='cpu', with_bias=False, lambda_=0.5, train_iters=200, attack_iters=10,
                 lr=0.01, with_relu=False):
        super().__init__(num_nodes=num_nodes, feature_shape=feature_shape, lambda_=lambda_, train_iters=train_iters,
                         attack_iters=attack_iters, lr=lr, attack_features=attack_features,
                         attack_structure=attack_structure, undirected=undirected, device=device)

        self.lr = lr
        self.train_iters = train_iters
        self.attack_iters = attack_iters
        self.adj_meta_grad = None
        self.features_meta_grad = None
        if self.attack_structure:
            self.adj_grad_sum = torch.zeros(num_nodes, num_nodes).to(device)
        if self.attack_features:
            self.feature_grad_sum = torch.zeros(feature_shape).to(device)

        self.with_bias = with_bias
        self.with_relu = with_relu

        self.weights = []
        self.biases = []

    def attack(self, gen_dataset, attack_budget=500, ll_constraint=True, ll_cutoff=0.004):
        super().attack(gen_dataset=gen_dataset)

        self.hidden_sizes = [16]   # FIXME get from model architecture
        self.nfeat = gen_dataset.num_node_features
        self.nclass = gen_dataset.num_classes

        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(self.device))
            bias = Parameter(torch.FloatTensor(previous_size, nhid).to(self.device))
            previous_size = nhid

            self.weights.append(weight)
            self.biases.append(bias)

        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass).to(self.device))
        output_bias = Parameter(torch.FloatTensor(self.nclass).to(self.device))
        self.weights.append(output_weight)
        self.biases.append(output_bias)

        self.optimizer = optim.Adam(self.weights + self.biases, lr=self.lr)  # , weight_decay=5e-4)
        self._initialize()

        ori_features = gen_dataset.dataset.data.x
        ori_adj = gen_dataset.dataset.data.edge_index
        labels = gen_dataset.dataset.data.y
        idx_train = gen_dataset.train_mask
        idx_unlabeled = gen_dataset.test_mask

        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        labels_self_training = self.self_training_label(labels, idx_train)
        self.sparse_features = sp.issparse(ori_features)
        modified_adj = ori_adj
        modified_features = ori_features

        for i in tqdm(range(attack_budget), desc="Perturbing graph"):
            self._initialize()

            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)
                self.adj_grad_sum.data.fill_(0)
            if self.attack_features:
                modified_features = ori_features + self.feature_changes
                self.feature_grad_sum.data.fill_(0)

            self.inner_train(modified_features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training)

            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)

            if self.attack_structure:
                adj_meta_score = self.get_adj_score(self.adj_grad_sum, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(self.feature_grad_sum, modified_features)

            if adj_meta_score.max() >= feature_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)
                row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
                self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
                if self.undirected:
                    self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            else:
                feature_meta_argmax = torch.argmax(feature_meta_score)
                row_idx, col_idx = utils.unravel_index(feature_meta_argmax, ori_features.shape)
                self.feature_changes.data[row_idx][col_idx] += (-2 * modified_features[row_idx][col_idx] + 1)

        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()

        gen_dataset.dataset.data.edge_index = dense_to_sparse(self.modified_adj.int())[0]
        print("TEST")

    def _initialize(self):
        for w, b in zip(self.weights, self.biases):
            # w.data.fill_(1)
            # b.data.fill_(1)
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)

        self.optimizer = optim.Adam(self.weights + self.biases, lr=self.lr)

    def inner_train(self, features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training):
        adj_norm = utils.normalize_adj_tensor(modified_adj)
        for j in range(self.attack_iters):
            # hidden = features
            # for w, b in zip(self.weights, self.biases):
            #     if self.sparse_features:
            #         hidden = adj_norm @ torch.spmm(hidden, w) + b
            #     else:
            #         hidden = adj_norm @ hidden @ w + b
            #     if self.with_relu:
            #         hidden = F.relu(hidden)

            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b
                if self.with_relu:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
            loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])

            if self.lambda_ == 1:
                attack_loss = loss_labeled
            elif self.lambda_ == 0:
                attack_loss = loss_unlabeled
            else:
                attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

            self.optimizer.zero_grad()
            loss_labeled.backward(retain_graph=True)

            if self.attack_structure:
                self.adj_changes.grad.zero_()
                self.adj_grad_sum += torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
            if self.attack_features:
                self.feature_changes.grad.zero_()
                self.feature_grad_sum += torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]

            self.optimizer.step()


        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
        print('GCN loss on unlabled data: {}'.format(loss_test_val.item()))
        print('GCN acc on unlabled data: {}'.format(
            utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))