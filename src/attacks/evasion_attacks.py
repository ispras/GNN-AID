import torch
import torch.nn.functional as F
import numpy as np

from attacks.attack_base import Attacker

# Nettack imports
from src.attacks.nettack.nettack import Nettack
from src.attacks.nettack.utils import preprocess_graph, largest_connected_components, data_to_csr_matrix, train_w1_w2


class EvasionAttacker(Attacker):
    def __init__(self, **kwargs):
        super().__init__()


class EmptyEvasionAttacker(EvasionAttacker):
    name = "EmptyEvasionAttacker"

    def attack(self, **kwargs):
        pass


class FGSMAttacker(EvasionAttacker):
    name = "FGSM"

    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def attack(self, model_manager, gen_dataset, mask_tensor):
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


class NettackEvasionAttacker(EvasionAttacker):
    name = "NettackEvasionAttacker"

    def __init__(self,
                 node_idx=0,
                 n_perturbations=None,
                 perturb_features=True,
                 perturb_structure=True,
                 direct=True,
                 n_influencers=0
                 ):

        super().__init__()
        self.attack_diff = None
        self.node_idx = node_idx
        self.n_perturbations = n_perturbations
        self.perturb_features = perturb_features
        self.perturb_structure = perturb_structure
        self.direct = direct
        self.n_influencers = n_influencers

    def attack(self, model_manager, gen_dataset, mask_tensor):
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

    def attack_diff(self):
        return self.attack_diff

    @staticmethod
    def _evasion(gen_dataset, feature_perturbations, structure_perturbations):
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
                                        torch.tensor((edge[0], edge[1]), dtype=torch.int32).to(torch.int64).unsqueeze(1)), dim=1)
                edge_index = torch.cat((edge_index,
                                        torch.tensor((edge[1], edge[0]), dtype=torch.int32).to(torch.int64).unsqueeze(1)), dim=1)

            gen_dataset.data.edge_index = edge_index

class NettackGroupEvasionAttacker(EvasionAttacker):
    name = "NettackGroupEvasionAttacker"
    def __init__(self,node_idxs, **kwargs):
        super().__init__()
        self.node_idxs = node_idxs # kwargs.get("node_idxs")
        assert isinstance(self.node_idxs, list)
        self.n_perturbations = kwargs.get("n_perturbations")
        self.perturb_features = kwargs.get("perturb_features")
        self.perturb_structure = kwargs.get("perturb_structure")
        self.direct = kwargs.get("direct")
        self.n_influencers = kwargs.get("n_influencers")
        self.attacker = NettackEvasionAttacker(0, **kwargs)

    def attack(self, model_manager, gen_dataset, mask_tensor):
        for node_idx in self.node_idxs:
            self.attacker.node_idx = node_idx
            gen_dataset = self.attacker.attack(model_manager, gen_dataset, mask_tensor)
        return gen_dataset