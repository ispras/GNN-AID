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
from tqdm import tqdm


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
            epsilon: float = 0.1
    ):
        super().__init__()
        self.epsilon = epsilon

    def attack(
            self,
            model_manager: Type,
            gen_dataset: GeneralDataset,
            mask_tensor: torch.Tensor
    ):
        gen_dataset.data.x.requires_grad = True
        output = model_manager.gnn(gen_dataset.data.x, gen_dataset.data.edge_index, gen_dataset.data.batch)
        loss = model_manager.loss_function(output[mask_tensor],
                                           gen_dataset.data.y[mask_tensor])
        model_manager.gnn.zero_grad()
        loss.backward()
        sign_data_grad = gen_dataset.data.x.grad.sign()
        perturbed_data_x = gen_dataset.data.x + self.epsilon * sign_data_grad
        perturbed_data_x = torch.clamp(perturbed_data_x, 0, 1)
        gen_dataset.data.x = perturbed_data_x.detach()
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
            num_rand_trials: int = 100
    ):

        super().__init__()
        self.attack_diff = None
        self.is_feature_attack = is_feature_attack  # feature / structure
        self.element_idx = element_idx
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_rand_trials = num_rand_trials

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
            pass

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
            pass

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
