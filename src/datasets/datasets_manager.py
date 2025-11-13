import json
from pathlib import Path
from typing import Tuple

import torch_geometric

from aux.declaration import Declare
from aux.utils import import_by_name
from data_structures.configs import DatasetConfig, DatasetVarConfig, FeatureConfig, Task
from datasets.dataset_info import DatasetInfo
from datasets.gen_dataset import GeneralDataset
from datasets.ptg_datasets import LibPTGDataset


class DatasetManager:
    """
    Class for working with datasets. Methods: get - loads dataset in torch_geometric format for gnn
    along the path specified in full_name as tuple. Currently also supports automatic loading and
    processing of all datasets from torch_geometric.datasets
    """

    @staticmethod
    def get_by_config(
            dataset_config: DatasetConfig,
            dataset_var_config: DatasetVarConfig = None,
            **params
    ) -> GeneralDataset:
        """
        Get GeneralDataset by dataset config. Convenient to use from the frontend.

        :param dataset_config:
        :param dataset_var_config:
        :param params: additional parameters to init dataset class
        """
        path = Declare.dataset_info_path(dataset_config)

        # Check special cases when there is no metainfo file but we know where to get class
        if not path.exists():
            from datasets.ptg_datasets import LibPTGDataset
            if dataset_config.full_name[0] == LibPTGDataset.data_folder:
                class_name = LibPTGDataset.__name__
                import_from = LibPTGDataset.__module__
            else:
                raise RuntimeError(f"No metainfo file found at '{path}'.")

        else:
            # Read metainfo
            info = DatasetInfo.read(path)
            class_name = info.class_name
            import_from = info.import_from
            if class_name is None or import_from is None:
                raise RuntimeError(f"Metainfo file does not contain field 'class_name' or 'import_from'."
                                   f" They must be specified in metainfo file, check it {path}")

        klass = import_by_name(class_name, [import_from])
        dataset = klass(dataset_config=dataset_config, **params)

        # Build dataset
        if dataset_var_config:
            dataset.build(dataset_var_config)

        return dataset

    @staticmethod
    def add_labeling(
            dataset_config: DatasetConfig,
            task: Task,
            labeling_name: str,
            labeling_dict: dict,
            value: int | list | None = None,
            force_rewrite = False
    ) -> None:
        """
        Adds a new labeling to datasets with a specified dataset_config.

        Args:
            dataset_config (DatasetConfig):
            task (Task):
            labeling_name (str): name for a new labeling
            labeling_dict (dict): dictionary with labels {node/edge/graph -> value}
            value (int | list | None, optional): possible value depending on the task:
             number of classes or regression value bounds. Will be induced if omitted
            force_rewrite (bool): if True, will rewrite labeling if exists
        """
        info = DatasetInfo.read(Declare.dataset_info_path(dataset_config))

        if task in info.labelings and labeling_name in info.labelings[task] and not force_rewrite:
            raise NameError(f"Labeling '{labeling_name}' for task {task} already exists for dataset"
                            f" {dataset_config}.")

        # Some checks
        if task.is_node_level():
            assert info.count == 1
            assert len(labeling_dict) == info.nodes[0]
        elif task.is_edge_level():
            assert info.count == 1
        elif task.is_graph_level():
            assert len(labeling_dict) == info.count
        else:
            raise ValueError(f"Adding labelings for task {task} is not supported.")

        if value is None:
            if task.is_classification():
                value = max(labeling_dict.values()) + 1
            if task.is_regression():
                value = [min(labeling_dict.values()), max(labeling_dict.values())]

        # Save labeling_dict to file and update metainfo
        path = Declare.dataset_root_dir(dataset_config)[0] / 'raw' / 'labels' / task / labeling_name
        path.parent.mkdir(parents=True)
        with open(path, 'w') as f:
            json.dump(labeling_dict, f, indent=1)

        if task not in info.labelings:
            info.labelings[task] = {}
        info.labelings[task][labeling_name] = value
        info.save(Declare.dataset_info_path(dataset_config))
