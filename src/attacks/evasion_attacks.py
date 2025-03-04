from typing import Type, Union

import torch
import torch.nn.functional as F
import numpy as np

from attacks.attack_base import Attacker
from base.datasets_processing import GeneralDataset

# Nettack imports
from src.attacks.nettack.nettack import Nettack
from src.attacks.nettack.utils import preprocess_graph, largest_connected_components, data_to_csr_matrix, train_w1_w2

# PGD imports
from attacks.evasion_attacks_collection.pgd.utils import Projection, RandomSampling
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from models_builder.models_utils import apply_decorator_to_graph_layers
from tqdm import tqdm
from skopt import Optimizer

# FGSM imports
from models_builder.models_utils import apply_decorator_to_graph_layers
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data

# ReWatt imports
from attacks.evasion_attacks_collection.rewatt.utils import *


class EvasionAttacker(
    Attacker
):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__()


class EmptyEvasionAttacker(
    EvasionAttacker
):
    name = "EmptyEvasionAttacker"

    def attack(
            self,
            **kwargs
    ):
        pass


class FGSMAttacker(
    EvasionAttacker
):
    name = "FGSM"

    def __init__(
            self,
            is_feature_attack: bool = False,
            element_idx: int = 0,
            epsilon: float = 0.5,
    ):
        super().__init__()
        self.is_feature_attack = is_feature_attack
        self.element_idx = element_idx
        self.epsilon = epsilon
        self.grad_aggr_type = 'mean'

    def attack(
            self,
            model_manager: Type,
            gen_dataset: GeneralDataset,
            mask_tensor: torch.Tensor
    ):
        if self.is_feature_attack:
            gen_dataset.data.x.requires_grad = True
            model = model_manager.gnn
            model.eval()
            output = model(gen_dataset.data.x, gen_dataset.data.edge_index, gen_dataset.data.batch)
            if gen_dataset.is_multi():
                loss = model_manager.loss_function(output, gen_dataset.dataset[self.element_idx].y)
            else:
                loss = model_manager.loss_function(output[self.element_idx], gen_dataset.data.y[self.element_idx])
            model.zero_grad()
            loss.backward()
            sign_data_grad = gen_dataset.data.x.grad.sign()
            perturbed_data_x = gen_dataset.data.x + self.epsilon * sign_data_grad
            perturbed_data_x = torch.clamp(perturbed_data_x, 0, 1)
            gen_dataset.data.x = perturbed_data_x.detach()
        else:
            if gen_dataset.is_multi():
                graph_idx = self.element_idx

                edge_index = gen_dataset.dataset[graph_idx].edge_index
                y = gen_dataset.dataset[graph_idx].y
                x = gen_dataset.dataset[graph_idx].x

                model = model_manager.gnn
                model.eval()

                first_layer_name, first_layer = next(model.named_children())
                layer = getattr(model, first_layer_name)
                apply_decorator_to_graph_layers(model)

                budget = int(self.epsilon * edge_index.size(1))

                # TODO Some convolutions use the add_self_loops parameter. It is not possible to work with loops,
                #  because even if you remove a loop during the method, it will be used in the model architecture.
                # if layer.add_self_loops is True:
                #     pass

                perturbed_edges = edge_index.clone()
                self_loops_part_size = x.size(0)

                for _ in tqdm(range(budget)):
                    out = model(x, perturbed_edges)
                    loss = -model_manager.loss_function(out, y)
                    loss.backward()

                    grad = layer.message_gradients[first_layer_name]
                    if self.grad_aggr_type == 'mean':
                        grad = torch.mean(grad, dim=1)
                    else:
                        raise ValueError(f"Unsupported grad_aggr_type: {self.grad_aggr_type}")

                    if grad.size(0) != perturbed_edges.size(1):  # model use add_self_loops
                        grad = grad[:-self_loops_part_size]

                    max_index = torch.argmax(grad)
                    perturbed_edges = torch.cat((perturbed_edges[:, :max_index], perturbed_edges[:, max_index + 1:]),
                                                dim=1)
                self.attack_diff = Data(x=x, edge_index=perturbed_edges, y=y)
            else:
                node_idx = self.element_idx

                edge_index = gen_dataset.data.edge_index
                y = gen_dataset.data.y
                x = gen_dataset.data.x

                model = model_manager.gnn
                model.eval()
                num_hops = model.n_layers

                subset, edge_index_subset, inv, edge_mask = k_hop_subgraph(node_idx=node_idx,
                                                                           num_hops=num_hops,
                                                                           edge_index=edge_index,
                                                                           relabel_nodes=True,
                                                                           directed=False)
                node_idx_remap = torch.where(subset == node_idx)[0].item()
                y = y.clone()
                y = y[subset]
                x = x.clone()
                x = x[subset]

                first_layer_name, first_layer = next(model.named_children())
                layer = getattr(model, first_layer_name)
                apply_decorator_to_graph_layers(model)

                budget = int(self.epsilon * edge_index_subset.size(1))

                # TODO Some convolutions use the add_self_loops parameter. It is not possible to work with loops,
                #  because even if you remove a loop during the method, it will be used in the model architecture.
                # if layer.add_self_loops is True:
                #     pass

                perturbed_edges = edge_index_subset.clone()
                self_loops_part_size = subset.size(0)

                for _ in tqdm(range(budget)):
                    out = model(x, perturbed_edges)
                    loss = -model_manager.loss_function(out[node_idx_remap], y[node_idx_remap])
                    loss.backward()

                    grad = layer.message_gradients[first_layer_name]
                    if self.grad_aggr_type == 'mean':
                        grad = torch.mean(grad, dim=1)
                    else:
                        raise ValueError(f"Unsupported grad_aggr_type: {self.grad_aggr_type}")

                    if grad.size(0) != perturbed_edges.size(1):  # model use add_self_loops
                        grad = grad[:-self_loops_part_size]

                    max_index = torch.argmax(grad)
                    perturbed_edges = torch.cat((perturbed_edges[:, :max_index], perturbed_edges[:, max_index + 1:]), dim=1)

                # Update dataset
                edges_to_keep = edge_index[:, ~edge_mask]
                updated_edge_index = torch.cat([edges_to_keep, perturbed_edges], dim=1)
                gen_dataset.data.edge_index = updated_edge_index
                self.attack_diff = gen_dataset
        return gen_dataset


class PGDAttacker(
    EvasionAttacker
):
    name = "PGD"

    def __init__(
            self,
            is_feature_attack: bool = False,
            element_idx: int = 0,
            epsilon: float = 0.5,
            learning_rate: float = 0.001,
            num_iterations: int = 100,
            num_rand_trials: int = 100,
            grad_aggr_type: str = 'mean'
    ):

        super().__init__()
        self.attack_diff = None
        self.is_feature_attack = is_feature_attack  # feature / structure
        self.element_idx = element_idx
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_rand_trials = num_rand_trials
        self.grad_aggr_type = grad_aggr_type

        # TODO check grad_aggr_type correctness
        # raise ValueError(f"Invalid grad_aggr_type: {self.grad_aggr_type}.")

    def attack(
            self,
            model_manager: Type,
            gen_dataset: GeneralDataset,
            mask_tensor: torch.Tensor
    ) -> None:
        if gen_dataset.is_multi():
            self._attack_on_graph(model_manager, gen_dataset)
        else:
            self._attack_on_node(model_manager, gen_dataset)

    def _attack_on_node(
            self,
            model_manager: Type,
            gen_dataset: GeneralDataset
    ) -> None:
        node_idx = self.element_idx

        edge_index = gen_dataset.data.edge_index
        y = gen_dataset.data.y
        x = gen_dataset.data.x

        model = model_manager.gnn
        model.eval()
        num_hops = model.n_layers

        subset, edge_index_subset, inv, edge_mask = k_hop_subgraph(node_idx=node_idx,
                                                                   num_hops=num_hops,
                                                                   edge_index=edge_index,
                                                                   relabel_nodes=True,
                                                                   directed=False)

        if self.is_feature_attack:  # feature attack
            node_idx_remap = torch.where(subset == node_idx)[0].item()
            y = y.clone()
            y = y[subset]
            x = x.clone()
            x = x[subset]
            orig_x = x.clone()
            x.requires_grad = True
            optimizer = torch.optim.Adam([x], lr=self.learning_rate, weight_decay=5e-4)

            for t in tqdm(range(self.num_iterations)):
                out = model(x, edge_index_subset)
                loss = -model_manager.loss_function(out[node_idx_remap], y[node_idx_remap])
                # print(loss)
                model.zero_grad()
                loss.backward()
                x.grad.sign_()
                optimizer.step()
                with torch.no_grad():
                    x.copy_(torch.max(torch.min(x, orig_x + self.epsilon), orig_x - self.epsilon))
                    x.copy_(torch.clamp(x, -self.epsilon, self.epsilon))
            # return the modified lines back to the original tensor x
            gen_dataset.data.x[subset] = x.detach()
            self.attack_diff = gen_dataset
        else:  # structure attack
            node_idx_remap = torch.where(subset == node_idx)[0].item()
            y = y.clone()
            y = y[subset]
            x = x.clone()
            x = x[subset]

            budget = int(self.epsilon * edge_index_subset.size(1))
            perturbed_edges = edge_index_subset.clone()

            space = [(0, 1) for _ in range(perturbed_edges.size(1))]  # binary space {0, 1}^N

            opt = Optimizer(
                dimensions=space,
                acq_func="EI",
                random_state=42
            )

            orig_mask = torch.ones(edge_index_subset.size(1), dtype=torch.bool)
            for i in tqdm(range(self.num_iterations)):
                mask = opt.ask()  # selecting the next point
                mask = torch.tensor(mask, dtype=torch.bool)

                # --- budget control ---
                diff = mask != orig_mask
                change_indices = torch.where(diff)[0]
                num_changes = len(change_indices)

                if num_changes > budget:
                    perturbed_masks = []
                    losses = []

                    for idx in change_indices:
                        temp_mask = mask.clone()
                        temp_mask[idx] = orig_mask[idx]

                        perturbed_edges = edge_index_subset[:, temp_mask]
                        out = model(x, perturbed_edges)
                        loss = -model_manager.loss_function(out[node_idx_remap], y[node_idx_remap])

                        perturbed_masks.append(temp_mask)
                        losses.append(loss.item())

                    # select `eps` of the most "dangerous" changes (maximizing `loss`)
                    sorted_indices = torch.tensor(losses).argsort(descending=True)[:budget]
                    selected_changes = change_indices.clone().detach()[sorted_indices]

                    mask[:] = orig_mask
                    mask[selected_changes] = ~orig_mask[selected_changes]
                # ----------------------

                perturbed_edges = edge_index_subset[:, mask]  # apply mask to edges
                out = model(x, perturbed_edges)
                loss = -model_manager.loss_function(out[node_idx_remap], y[node_idx_remap])
                opt.tell(mask.tolist(), loss.item())  # report the result to the optimizer
            best_mask = opt.Xi[np.argmin(opt.yi)]
            best_mask = torch.tensor(best_mask, dtype=torch.bool)

            # Update dataset
            edges_to_keep = edge_index[:, ~edge_mask]
            perturbed_edges = edge_index_subset[:, best_mask]
            updated_edge_index = torch.cat([edges_to_keep, perturbed_edges], dim=1)

            gen_dataset.data.edge_index = updated_edge_index
            self.attack_diff = gen_dataset

    def _attack_on_graph(
            self,
            model_manager: Type,
            gen_dataset: GeneralDataset
    ):
        graph_idx = self.element_idx

        edge_index = gen_dataset.dataset[graph_idx].edge_index
        y = gen_dataset.dataset[graph_idx].y
        x = gen_dataset.dataset[graph_idx].x

        model = model_manager.gnn
        model.eval()

        if self.is_feature_attack:  # feature attack
            x = x.clone()
            orig_x = x.clone()
            x.requires_grad = True
            optimizer = torch.optim.Adam([x], lr=self.learning_rate, weight_decay=5e-4)

            for t in tqdm(range(self.num_iterations)):
                out = model(x, edge_index)
                loss = -model_manager.loss_function(out, y)
                # print(loss)
                model.zero_grad()
                loss.backward()
                x.grad.sign_()
                optimizer.step()
                with torch.no_grad():
                    x.copy_(torch.max(torch.min(x, orig_x + self.epsilon), orig_x - self.epsilon))
                    x.copy_(torch.clamp(x, -self.epsilon, self.epsilon))
            gen_dataset.dataset[graph_idx].x.copy_(x.detach())
            self.attack_diff = gen_dataset
        else:  # structure attack
            budget = int(self.epsilon * edge_index.size(1))
            perturbed_edges = edge_index.clone()

            space = [(0, 1) for _ in range(perturbed_edges.size(1))]  # binary space {0, 1}^N

            opt = Optimizer(
                dimensions=space,
                acq_func="EI",
                random_state=42
            )

            orig_mask = torch.ones(edge_index.size(1), dtype=torch.bool)
            for i in tqdm(range(self.num_iterations)):
                mask = opt.ask()  # selecting the next point
                mask = torch.tensor(mask, dtype=torch.bool)

                # --- budget control ---
                diff = mask != orig_mask
                change_indices = torch.where(diff)[0]
                num_changes = len(change_indices)

                if num_changes > budget:
                    perturbed_masks = []
                    losses = []

                    for idx in change_indices:
                        temp_mask = mask.clone()
                        temp_mask[idx] = orig_mask[idx]

                        perturbed_edges = edge_index[:, temp_mask]
                        out = model(x, perturbed_edges)
                        loss = -model_manager.loss_function(out, y)

                        perturbed_masks.append(temp_mask)
                        losses.append(loss.item())

                    # select `eps` of the most "dangerous" changes (maximizing `loss`)
                    sorted_indices = torch.tensor(losses).argsort(descending=True)[:budget]
                    selected_changes = change_indices.clone().detach()[sorted_indices]

                    mask[:] = orig_mask
                    mask[selected_changes] = ~orig_mask[selected_changes]
                # ----------------------

                perturbed_edges = edge_index[:, mask]  # apply mask to edges
                out = model(x, perturbed_edges)
                loss = -model_manager.loss_function(out, y)
                opt.tell(mask.tolist(), loss.item())  # report the result to the optimizer
            best_mask = opt.Xi[np.argmin(opt.yi)]
            best_mask = torch.tensor(best_mask, dtype=torch.bool)
            perturbed_edges = edge_index[:, best_mask]

            self.attack_diff = Data(x=x, edge_index=perturbed_edges, y=y)

    def attack_diff(
            self
    ):
        return self.attack_diff


class NettackEvasionAttacker(
    EvasionAttacker
):
    name = "NettackEvasionAttacker"

    def __init__(
            self,
            node_idx: int = 0,
            n_perturbations: Union[int, None] = None,
            perturb_features: bool = True,
            perturb_structure: bool = True,
            direct: bool = True,
            n_influencers: int = 0
    ):

        super().__init__()
        self.attack_diff = None
        self.node_idx = node_idx
        self.n_perturbations = n_perturbations
        self.perturb_features = perturb_features
        self.perturb_structure = perturb_structure
        self.direct = direct
        self.n_influencers = n_influencers

    def attack(
            self,
            model_manager: Type,
            gen_dataset: GeneralDataset,
            mask_tensor: torch.Tensor
    ) -> GeneralDataset:
        # Prepare
        data = gen_dataset.data
        _A_obs, _X_obs, _z_obs = data_to_csr_matrix(data)
        _A_obs = _A_obs + _A_obs.T
        _A_obs[_A_obs > 1] = 1
        lcc = largest_connected_components(_A_obs)

        _A_obs = _A_obs[lcc][:, lcc]

        assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
        assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
        assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        _X_obs = _X_obs[lcc].astype('float32')
        _z_obs = _z_obs[lcc]
        _N = _A_obs.shape[0]
        _K = _z_obs.max() + 1
        _Z_obs = np.eye(_K)[_z_obs]
        _An = preprocess_graph(_A_obs)
        degrees = _A_obs.sum(0).A1

        if self.n_perturbations is None:
            self.n_perturbations = int(degrees[self.node_idx])
        hidden = model_manager.gnn.GCNConv_0.out_channels
        # End prepare

        # Learn matrix W1 and W2
        W1, W2 = train_w1_w2(dataset=gen_dataset, hidden=hidden)

        # Attack
        nettack = Nettack(_A_obs, _X_obs, _z_obs, W1, W2, self.node_idx, verbose=True)

        nettack.reset()
        nettack.attack_surrogate(n_perturbations=self.n_perturbations,
                                 perturb_structure=self.perturb_structure,
                                 perturb_features=self.perturb_features,
                                 direct=self.direct,
                                 n_influencers=self.n_influencers)

        print(f'edges: {nettack.structure_perturbations}')
        print(f'features: {nettack.feature_perturbations}')

        self._evasion(gen_dataset, nettack.feature_perturbations, nettack.structure_perturbations)
        self.attack_diff = gen_dataset

        return gen_dataset

    def attack_diff(
            self
    ):
        return self.attack_diff

    @staticmethod
    def _evasion(
            gen_dataset: GeneralDataset,
            feature_perturbations,
            structure_perturbations
    ):
        cleaned_feat_pert = list(filter(None, feature_perturbations))
        if cleaned_feat_pert:  # list is not empty
            x = gen_dataset.data.x.clone()
            for vertex, feature in cleaned_feat_pert:
                if x[vertex, feature] == 0.0:
                    x[vertex, feature] = 1.0
                elif x[vertex, feature] == 1.0:
                    x[vertex, feature] = 0.0
            gen_dataset.data.x = x

        cleaned_struct_pert = list(filter(None, structure_perturbations))
        if cleaned_struct_pert:  # list is not empty
            edge_index = gen_dataset.data.edge_index.clone()
            # add edges
            for edge in cleaned_struct_pert:
                edge_index = torch.cat((edge_index,
                                        torch.tensor((edge[0], edge[1]), dtype=torch.int32).to(torch.int64).unsqueeze(
                                            1)), dim=1)
                edge_index = torch.cat((edge_index,
                                        torch.tensor((edge[1], edge[0]), dtype=torch.int32).to(torch.int64).unsqueeze(
                                            1)), dim=1)

            gen_dataset.data.edge_index = edge_index


class NettackGroupEvasionAttacker(
    EvasionAttacker
):
    name = "NettackGroupEvasionAttacker"

    def __init__(
            self,
            node_idxs: list,
            **kwargs
    ):
        super().__init__()
        self.node_idxs = node_idxs  # kwargs.get("node_idxs")
        assert isinstance(self.node_idxs, list)
        self.n_perturbations = kwargs.get("n_perturbations")
        self.perturb_features = kwargs.get("perturb_features")
        self.perturb_structure = kwargs.get("perturb_structure")
        self.direct = kwargs.get("direct")
        self.n_influencers = kwargs.get("n_influencers")
        self.attacker = NettackEvasionAttacker(0, **kwargs)

    def attack(
            self,
            model_manager: Type,
            gen_dataset: GeneralDataset,
            mask_tensor: torch.Tensor
    ) -> GeneralDataset:
        for node_idx in self.node_idxs:
            self.attacker.node_idx = node_idx
            gen_dataset = self.attacker.attack(model_manager, gen_dataset, mask_tensor)
        return gen_dataset


class ReWattAttacker(
    EvasionAttacker
):
    name = "ReWatt"

    def __init__(
            self,
            element_idx: int = 0,
            eps: int = 0.1,
            epochs: int = 100,
            mlp_hidden: int = 16,
            h_method: str = 'sum',
            pooling_method: str = 'mean',
    ):
        super().__init__()
        self.element_idx = element_idx
        self.eps = eps
        self.epochs = epochs
        self.mlp_hidden = mlp_hidden
        self.h_method = h_method
        self.pooling_method = pooling_method

        self.attack_diff = None
        self.my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attack(
            self,
            model_manager: Type,
            gen_dataset: GeneralDataset,
            mask_tensor: torch.Tensor
    ):
        model = model_manager.gnn
        model.eval()

        if gen_dataset.is_multi():
            graph_idx = self.element_idx
            node_idx=None
            edge_index = gen_dataset.dataset[graph_idx].edge_index
            y = gen_dataset.dataset[graph_idx].y.squeeze()
            x = gen_dataset.dataset[graph_idx].x
            y_prob = torch.softmax(model(x, edge_index), dim=1).squeeze().max().item()
        else:
            node_idx = self.element_idx
            y = gen_dataset.data.y[node_idx]
            x = gen_dataset.data.x
            edge_index = gen_dataset.data.edge_index
            y_prob = torch.softmax(model(x, edge_index)[node_idx], dim=0).max().item()

        # the attack makes sense when the model's prediction is correct !!!
        # we use y_prob in the attack because in case the attack fails to change the class, we have saved
        # the state of the graph that most reduces the probability of a correct prediction.
        initial_graph_state = GraphState(x, edge_index, y, y_prob)
        env = GraphEnvironment(model, initial_graph_state, eps=self.eps, node_idx=node_idx)

        # TODO check that embeddings will get before pooling if graph_glassification task
        penultimate_layer_embeddings_dim = model.get_all_layer_embeddings(x, edge_index)[model.n_layers - 2].size(1)

        policy = ReWattPolicyNet(gnn_model=model,
                                 penultimate_layer_embeddings_dim=penultimate_layer_embeddings_dim,
                                 node_idx=node_idx,
                                 mlp_hidden=self.mlp_hidden,
                                 h_method=self.h_method,
                                 pooling_method=self.pooling_method,
                                 device=self.my_device)

        agent = ReWattAgent(policy, env, lr=1e-3, gamma=0.99)
        attacked_graph = agent.train(epochs=self.epochs)

        if gen_dataset.is_multi():
            self.attack_diff = Data(x=x, edge_index=attacked_graph.edge_index, y=y)
        else:
            gen_dataset.data.edge_index = attacked_graph.edge_index
            self.attack_diff = gen_dataset
