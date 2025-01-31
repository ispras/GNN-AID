import numpy as np

from base.datasets_processing import GeneralDataset
from defense.defense_base import Defender


class PoisonDefender(
    Defender
):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__()

    def defense(
            self,
            **kwargs
    ):
        pass


class BadRandomPoisonDefender(
    PoisonDefender
):
    name = "BadRandomPoisonDefender"

    def __init__(
            self,
            n_edges_percent: float = 0.1
    ):
        self.defense_diff = None

        super().__init__()
        self.n_edges_percent = n_edges_percent

    def defense(
            self,
            gen_dataset: GeneralDataset
    ) -> GeneralDataset:
        edge_index = gen_dataset.data.edge_index
        random_indices = np.random.choice(
            edge_index.shape[1],
            int(edge_index.shape[1] * (1 - self.n_edges_percent)),
            replace=False
        )
        total_indices_array = np.arange(edge_index.shape[1])
        indices_to_remove = np.setdiff1d(total_indices_array, random_indices)
        edge_index_diff = edge_index[:, indices_to_remove]
        edge_index = edge_index[:, random_indices]
        gen_dataset.data.edge_index = edge_index
        self.defense_diff = edge_index_diff
        return gen_dataset

    def defense_diff(
            self
    ):
        return self.defense_diff


class EmptyPoisonDefender(
    PoisonDefender
):
    name = "EmptyPoisonDefender"

    def defense(
            self,
            gen_dataset: GeneralDataset
    ) -> GeneralDataset:
        return gen_dataset
