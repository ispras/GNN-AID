import json
from pathlib import Path
from typing import Union, Tuple

import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data

from aux.custom_decorators import timing_decorator
from aux.declaration import Declare
from aux.utils import tmp_dir, import_by_name
from data_structures.configs import DatasetConfig, DatasetVarConfig, FeatureConfig
from datasets.dataset_info import DatasetInfo
from datasets.gen_dataset import GeneralDataset


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
    def get_by_full_name(
            full_name: Tuple[str, ...] = None,
            **dvc_kwargs
    ) -> [GeneralDataset, torch_geometric.data.Data, Path]:
        """
        Get built `PTGDataset` by its full name and additional kwargs for dataset var config.
        Starts the creation of an object from raw data or takes already saved datasets in prepared
        form. NOTE: works only for `PTGDataset`.

        Args:
            full_name: full name of graph data as a tuple of strings
            **dvc_kwargs: kwargs for `DatasetVarConfig`, optional.
             If given, try to create var config and call dataset.build()

        Returns: GeneralDataset, a list of tensors with data, and
        the path where the dataset is saved.

        """
        from datasets.ptg_datasets import PTGDataset
        dc = DatasetConfig(full_name=full_name)
        dataset = DatasetManager.get_by_config(dc)
        # if not isinstance(dataset, PTGDataset):
        #     raise RuntimeError(f"get_by_full_name suits only for {PTGDataset.__name__} datasets."
        #                        f" You are trying to get {dataset.__class__.__name__}."
        #                        f" Use get_by_config() instead.")

        if dvc_kwargs:
            features = dvc_kwargs.get('features', FeatureConfig())
            # features = FeatureConfig(**dvc_kwargs.get('features', {}))
            dvc_kwargs['features'] = features

        cfg = PTGDataset.default_dataset_var_config.to_savable_dict()
        cfg.update(**dvc_kwargs)
        dataset_var_config = DatasetVarConfig(**cfg)

        dataset.build(dataset_var_config=dataset_var_config)

        dataset.train_test_split(percent_train_class=dvc_kwargs.get("percent_train_class", 0.8),
                                 percent_test_class=dvc_kwargs.get("percent_test_class", 0.2))

        # IMP Kirill suggest to return only dataset, else is its parts
        return dataset, dataset.data, dataset.prepared_dir
