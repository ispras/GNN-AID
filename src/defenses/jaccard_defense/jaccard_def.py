import torch
import numpy as np

from datasets_block.gen_dataset import GeneralDataset
from data_structures.graph_modification_artifacts import GraphModificationArtifact
from defenses.poison_defense import PoisonDefender


class JaccardDefender(
    PoisonDefender
):
    """
    Poison defense based on removing edges between dissimilar nodes
    """
    name = 'JaccardDefender'

    def __init__(self, threshold):
        super().__init__()
        self.thrsh = threshold
        self.remove_edge_index = None

    def defense(
            self,
            gen_dataset: GeneralDataset,
            **kwargs
    ) -> GeneralDataset:
        """
        Modify input graph by removing edges between dissimilar nodes
        :param gen_dataset: input graph dataset
        :return: modified graph (only adjacency matrix modified)
        """

        def is_binary_tensor(X: torch.Tensor) -> bool:
            return torch.all((X == 0) | (X == 1)).item()

        assert is_binary_tensor(gen_dataset.dataset.data.x), "The features should be presented in binary form"

        # TODO need to check whether features binary or not. Consistency required - Cora has 'unknown' features e.g.
        # self.drop_edges(batch)
        edge_index = gen_dataset.dataset.data.edge_index.tolist()
        #new_edge_mask = torch.zeros_like(gen_dataset.dataset.data.edge_index).bool()
        new_edge_index = [[],[]]
        self.remove_edge_index = [[], []]
        for i in range(len(edge_index[0])):
            if self.jaccard_index(gen_dataset.dataset.data.x, edge_index[0][i], edge_index[1][i]) > self.thrsh:
                # new_edge_mask[0,i] = True
                # new_edge_mask[1,i] = True
                new_edge_index[0].append(edge_index[0][i])
                new_edge_index[1].append(edge_index[1][i])
            else:
                self.remove_edge_index[0].append(edge_index[0][i])
                self.remove_edge_index[1].append(edge_index[1][i])
        # gen_dataset.dataset.data.edge_index *= new_edge_mask.float()
        gen_dataset.dataset.data.edge_index = torch.tensor(new_edge_index).long()
        return gen_dataset

    def jaccard_index(

            self,
            x,
            u,
            v
    ) -> float:
        """
        Computes jaccard index of 'u' and 'v' objects based on their features
        :param x: feature matrix
        :param u: index of object from dataset
        :param v: index of object from dataset
        :return:
        """
        im1 = x[u,:].detach().cpu().numpy().astype(bool)
        im2 = x[v,:].detach().cpu().numpy().astype(bool)
        intersection = np.logical_and(im1, im2)
        union = np.logical_or(im1, im2)
        return intersection.sum() / float(union.sum())

    def dataset_diff(
            self
    ) -> GraphModificationArtifact:
        diff = GraphModificationArtifact()

        try:
            src_nodes = self.remove_edge_index[0]
            dst_nodes = self.remove_edge_index[1]

            assert len(src_nodes) == len(dst_nodes), (
                "Mismatch in source and target edge lengths: "
                f"{len(src_nodes)} vs {len(dst_nodes)}"
            )

            edges_to_remove = [
                [src, dst] for src, dst in zip(src_nodes, dst_nodes)
            ]

            diff.remove_edges(edges_to_remove)
            self.defense_diff = diff

        except Exception as e:
            raise RuntimeError(
                f"Failed to build dataset diff from remove_edge_index: {e}"
            ) from e

        return self.defense_diff
