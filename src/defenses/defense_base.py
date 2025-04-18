from base.datasets_processing import GeneralDataset
from models_builder.gnn_models import GNNModelManager


class Defender:
    """ Base class for all defense methods.
    """
    name = "Defender"

    def __init__(
            self
    ):
        pass

    def defense_diff(
            self
    ):
        """ TODO Kirill add function docstring
        """
        pass

    @staticmethod
    def check_availability(
            gen_dataset: GeneralDataset,
            model_manager: GNNModelManager
    ):
        """ Check whether the method can be applied for the given dataset and model manager.

        :param gen_dataset: dataset
        :param model_manager: model manager
        """
        return True  # TODO implement in all subclasses
