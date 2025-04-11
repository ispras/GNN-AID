from base.datasets_processing import GeneralDataset
from models_builder.gnn_models import GNNModelManager


class Attacker:
    name = "Attacker"

    def __init__(
            self
    ):
        pass

    def attack(
            self,
            **kwargs
    ):
        pass

    def attack_diff(
            self
    ):
        pass

    @staticmethod
    def check_availability(
            gen_dataset: GeneralDataset,
            model_manager: GNNModelManager
    ):
        return True  # TODO implement in subclasses


