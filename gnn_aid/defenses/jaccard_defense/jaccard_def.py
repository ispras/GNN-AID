import torch
import numpy as np
from typing import Optional

from gnn_aid.datasets.gen_dataset import GeneralDataset
from gnn_aid.data_structures.graph_modification_artifacts import GraphModificationArtifact
from gnn_aid.defenses.poison_defense import PoisonDefender
from gnn_aid.data_structures.configs import Task


def _is_binary_tensor(X: torch.Tensor) -> bool:
    return torch.all((X == 0) | (X == 1)).item()


class JaccardDefender(PoisonDefender):
    """
    Poison defense based on removing edges between dissimilar nodes.
    """
    name = 'JaccardDefender'

    def __init__(self, threshold: float, binarize_threshold: Optional[float] = None):
        """
        :param threshold: Jaccard similarity threshold (edges with similarity <= threshold are removed)
        :param binarize_threshold: Optional threshold to binarize non-binary features
        """
        super().__init__()
        self.threshold = threshold
        self.binarize_threshold = binarize_threshold
        self.removed_edges_train = None
        self.original_num_edges = None

    def defense(
            self,
            gen_dataset: GeneralDataset,
            **kwargs
    ) -> GeneralDataset:
        task = gen_dataset.dataset_var_config.task

        if task in [Task.EDGE_PREDICTION, Task.EDGE_REGRESSION]:
            if not hasattr(gen_dataset, 'train_mask') or gen_dataset.train_mask is None:
                raise RuntimeError("JaccardDefender for link tasks requires train_test_split() to be called first")

        self.original_num_edges = gen_dataset.data.edge_index.size(1)

        x = self._prepare_features(gen_dataset.data.x)

        if task in [Task.EDGE_PREDICTION, Task.EDGE_REGRESSION]:
            gen_dataset = self._defense_link_task(gen_dataset, x)
        else:
            gen_dataset = self._defense_standard_task(gen_dataset, x)

        return gen_dataset

    def _prepare_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.binarize_threshold is not None:
            x = (x > self.binarize_threshold).float()
        elif not _is_binary_tensor(x):
            raise ValueError(
                "JaccardDefender requires binary features"
            )
        return x

    def _defense_link_task(
            self,
            gen_dataset: GeneralDataset,
            x: torch.Tensor
    ) -> GeneralDataset:
        train_edge_label_index = gen_dataset.edge_label_index[:, gen_dataset.train_mask]

        filtered_train_edges, removed_edges = self._filter_edges_jaccard(train_edge_label_index, x)
        self.removed_edges_train = removed_edges

        gen_dataset.data.edge_index = filtered_train_edges

        num_removed = removed_edges.size(1) if removed_edges is not None else 0
        print(f"JaccardDefender: Removed {num_removed}/{train_edge_label_index.size(1)} "
              f"training edges (threshold={self.threshold})")

        return gen_dataset

    def _defense_standard_task(
            self,
            gen_dataset: GeneralDataset,
            x: torch.Tensor
    ) -> GeneralDataset:
        filtered_edges, removed_edges = self._filter_edges_jaccard(
            gen_dataset.data.edge_index, x
        )
        self.removed_edges_train = removed_edges  # Reusing field for simplicity

        gen_dataset.data.edge_index = filtered_edges

        num_removed = removed_edges.size(1) if removed_edges is not None else 0
        print(f"JaccardDefender: Removed {num_removed}/{self.original_num_edges} edges "
              f"(threshold={self.threshold})")

        return gen_dataset

    def _filter_edges_jaccard(
            self,
            edge_index: torch.Tensor,
            x: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if edge_index.size(1) == 0:
            return edge_index, None

        src_feats = x[edge_index[0]]
        dst_feats = x[edge_index[1]]

        intersection = (src_feats * dst_feats).sum(dim=1)  # AND
        union = ((src_feats + dst_feats) > 0).sum(dim=1).float()  # OR

        union = torch.where(union == 0, torch.ones_like(union), union)

        jaccard_scores = intersection / union

        keep_mask = jaccard_scores > self.threshold
        filtered_edges = edge_index[:, keep_mask]
        removed_edges = edge_index[:, ~keep_mask] if (~keep_mask).any() else None

        return filtered_edges, removed_edges

    def dataset_diff(self) -> GraphModificationArtifact:
        diff = GraphModificationArtifact()

        if self.removed_edges_train is not None and self.removed_edges_train.size(1) > 0:
            edges_to_remove = self.removed_edges_train.t().tolist()
            diff.remove_edges(edges_to_remove)
            self.defense_diff = diff
        else:
            # No edges removed
            self.defense_diff = diff

        return self.defense_diff
