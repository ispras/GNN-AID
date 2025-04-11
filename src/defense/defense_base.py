from base.datasets_processing import GeneralDataset
from models_builder.gnn_models import GNNModelManager


class Defender:
    name = "Defender"

    def __init__(
            self
    ):
        pass

    def defense_diff(
            self
    ):
        pass

    @staticmethod
    def check_availability(
            gen_dataset: GeneralDataset,
            model_manager: GNNModelManager
    ):
        return True  # TODO implement in subclasses


