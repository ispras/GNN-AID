from gnn_aid.data_structures.graph_modification_artifacts import GraphModificationArtifact
from gnn_aid.datasets.gen_dataset import GeneralDataset
from gnn_aid.models_builder.gnn_models import GNNModelManager


class Defender:
    """ Base class for all defense methods.
    """
    name = "Defender"

    def __init__(
            self
    ):
        self.defense_diff = None

    def dataset_diff(
            self
    ) -> GraphModificationArtifact:
        diff = GraphModificationArtifact()

        # diff.remove_nodes([0, 1])
        # diff.add_node(999, torch.tensor([0.1, 0.2, 0.3]))
        # diff.change_node_feature(2, 0, 0.5)
        #
        # diff.add_edge(2, 999, torch.tensor([1.0]))
        # diff.remove_edge(4, 5)

        self.defense_diff = diff
        return diff

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
