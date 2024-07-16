import numpy as np

from base.datasets_processing import GeneralDataset


class Defender:
    name = "Defender"

    def __init__(self, gen_dataset: GeneralDataset, model):
        self.gen_dataset = gen_dataset
        self.model = model

    def defense(self):
        pass

    def defense_diff(self):
        pass

    @staticmethod
    def check_availability(gen_dataset, model_manager):
        return False


class EvasionDefender(Defender):
    def __init__(self, gen_dataset: GeneralDataset, model, evasion_defense_config=None):
        super().__init__(gen_dataset, model)
        self.evasion_defense_config = evasion_defense_config


class MIDefender(Defender):
    def __init__(self, gen_dataset: GeneralDataset, model, mi_defense_config=None):
        super().__init__(gen_dataset, model)
        self.mi_defense_config = mi_defense_config


class PoisonDefender(Defender):
    def __init__(self, gen_dataset: GeneralDataset, model, poison_defense_config=None):
        super().__init__(gen_dataset, model)
        self.poison_defense_config = poison_defense_config


class BadRandomPoisonDefender(Defender):
    def __init__(self, gen_dataset: GeneralDataset, model, n_edges_percent=0.1):
        super().__init__(gen_dataset=gen_dataset, model=model)
        self.defense_diff = None
        self.n_edges_percent = n_edges_percent

    def defense(self):
        edge_index = self.gen_dataset.data.edge_index
        random_indices = np.random.choice(
            edge_index.shape[1],
            int(edge_index.shape[1] * (1 - self.n_edges_percent)),
            replace=False
        )
        total_indices_array = np.arange(edge_index.shape[1])
        indices_to_remove = np.setdiff1d(total_indices_array, random_indices)
        edge_index_diff = edge_index[:, indices_to_remove]
        edge_index = edge_index[:, random_indices]
        self.gen_dataset.data.edge_index = edge_index
        self.defense_diff = edge_index_diff

    def defense_diff(self):
        return self.defense_diff
