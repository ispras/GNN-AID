import torch

from typing import Type
from attacks.attack_base import Attacker
from base.datasets_processing import GeneralDataset
from data_structures.mi_results import MIResultsStore


class MIAttacker(
    Attacker
):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__()


class EmptyMIAttacker(
    MIAttacker
):
    name = "EmptyMIAttacker"

    def attack(
            self,
            **kwargs
    ):
        pass


class NaiveMIAttacker(MIAttacker):
    name = "NaiveMIAttacker"

    def __init__(self, threshold: float = 0.75, **kwargs):
        super().__init__(**kwargs)
        self.thrsh = threshold
        self.results = MIResultsStore()

    def attack(
        self,
        model_manager: Type,
        gen_dataset: GeneralDataset,
        mask_tensor: torch.Tensor,
    ):
        model_manager.gnn.eval()

        data = gen_dataset.data
        with torch.no_grad():
            probs = model_manager.get_predictions(data.x, data.edge_index)
            max_probs = torch.max(probs, dim=1).values
            inferred_train_mask = max_probs > self.thrsh
            self.results.add(mask_tensor, inferred_train_mask)

        return self.results
