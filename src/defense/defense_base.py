from typing import Type

from base.datasets_processing import GeneralDataset
from data_structures.graph_modification_artifacts import GraphModificationArtifact


class Defender:
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
            model_manager: Type
    ):
        return False


