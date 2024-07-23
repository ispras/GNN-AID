import numpy as np

from attacks.attack_base import Attacker


class PoisonAttacker(Attacker):
    def __init__(self, **kwargs):
        super().__init__()


class RandomPoisonAttack(PoisonAttacker):
    name = "RandomPoisonAttack"

    def __init__(self, n_edges_percent=0.1):
        self.attack_diff = None

        super().__init__()
        self.n_edges_percent = n_edges_percent

    def attack(self, gen_dataset):
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
        self.attack_diff = edge_index_diff
        return gen_dataset

    def attack_diff(self):
        return self.attack_diff


class EmptyPoisonAttacker(PoisonAttacker):
    name = "EmptyPoisonAttacker"

    def attack(self, **kwargs):
        pass