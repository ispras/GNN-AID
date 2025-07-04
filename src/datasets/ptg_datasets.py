import inspect
import os
import shutil
from pathlib import Path
from time import time
from typing import Union, List, Dict

import torch
from torch_geometric.data import Data

from aux.utils import import_by_name, TORCH_GEOM_GRAPHS_PATH
from datasets.gen_dataset import GeneralDataset, LocalDataset
from datasets.dataset_info import DatasetInfo
from data_structures.configs import DatasetConfig, DatasetVarConfig, ConfigPattern

PTG_FEATURE_NAME = "unknown"


class PTGDataset(GeneralDataset):
    """
    Generalisation and a wrapper over a single ptg Dataset.
    Features and labels are defined at initialisation.
    You should extend from this class.
    """
    default_dataset_var_config = DatasetVarConfig(
        features={'attr': {PTG_FEATURE_NAME: 'vector'}},
        labeling="origin",
        dataset_ver_ind=0
    )

    def __init__(
            self,
            dataset_config: Union[ConfigPattern, DatasetConfig],
            # **ptg_kwargs
    ):
        """
        :param dataset_config: dataset config dictionary
        # :param ptg_kwargs: additional args to init torch_geometric.data.Dataset
        """
        super(PTGDataset, self).__init__(dataset_config)

    def _compute_dataset_data(
            self
    ) -> None:
        self.dataset_var_config = PTGDataset.default_dataset_var_config.copy()

    def _compute_dataset_var_data(
            self
    ) -> None:
        raise RuntimeError(f"{self.__class__.__name__} does not compute var data.")

    def node_attributes(
            self,
            attrs: List[str] = None
    ) -> Dict[str, Union[list, torch.Tensor]]:
        """ Get node attributes as a dict {name -> list}"""
        assert attrs is None or attrs == [PTG_FEATURE_NAME]
        return {PTG_FEATURE_NAME: [data.x.tolist() for data in self.dataset]}

    def edge_attributes(
            self,
            attrs: List[str] = None
    ) -> Dict[str, Union[list, torch.Tensor]]:
        """ Get edge attributes as a dict {name -> list}"""
        raise NotImplementedError()


class LocalPTGDataset(PTGDataset):
    """
    A single ptg Dataset defined online by tensors.
    """
    data_folder = 'locally-created-graphs'

    def __init__(
            self,
            data_list: List[Data] = None,
            name: Union[str, None] = None,
            dataset_config: DatasetConfig = None
    ):
        """
        :param data_list: list of ready torch_geometric.data.Data objects
        :param name: unique dataset name
        :param dataset_config: is optional here
        """
        self.data_list = data_list

        if dataset_config:
            assert dataset_config.full_name[0] == self.data_folder
        else:
            if data_list is None:
                raise RuntimeError(f"{self.__class__.__name__}.__init__() must have specified"
                                   f" 'data_list' or 'dataset_config'")
            group = 'single-graph' if len(data_list) == 1 else 'multiple-graphs'
            # TODO misha Add hetero info
            name = name or 'graph_' + str(time())
            dataset_config = DatasetConfig((self.data_folder, group, name))
            # dataset_config = DatasetConfig((self.data_folder, group, name), {'data_list': data_list})

        super(LocalPTGDataset, self).__init__(dataset_config)

    def _compute_dataset_data(
            self
    ) -> None:
        super()._compute_dataset_data()
        # Create local ptg dataset

        if self.results_dir.exists():
            # Just read
            self.info = DatasetInfo.read(self.info_path)
            self.dataset = LocalDataset(None, self.results_dir)

        else:
            self.dataset = LocalDataset(self.data_list, self.results_dir)
            self.info = DatasetInfo.induce(self.dataset)
            self.info.save(self.info_path)

    def _compute_dataset_var_data(
            self
    ) -> None:
        pass  # Nothing to compute


class LibPTGDataset(PTGDataset):
    """
    A single ptg Dataset from Pytorch-Geometric library.
    """
    data_folder = 'ptg-library-graphs'

    def __init__(
            self,
            dataset_config: DatasetConfig,
            # domain: str,
            # group: str,
            # name: str,
            **params
    ):
        """
        :param group: group name, e.g. 'TUDataset'
        :param name: dataset name, e.g. 'MUTAG'
        :param params: optional init parameters
        """
        # assert domain in ['single-graph', 'multiple-graphs']
        first, self._domain, self._group, self._name = dataset_config.full_name
        assert first == self.data_folder
        self._params = params

        super(LibPTGDataset, self).__init__(dataset_config)

    def move_processed(
            self,
            processed: Union[str, Path]
    ) -> None:
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True)
            os.rename(processed, self.results_dir)
        else:
            shutil.rmtree(processed)

    def move_raw(
            self,
            raw: Union[str, Path]
    ) -> None:
        if Path(raw) == self.raw_dir:
            return
        if not self.raw_dir.exists():
            self.raw_dir.mkdir(parents=True)
            os.rename(raw, self.raw_dir)
        else:
            raise RuntimeError(f"raw_dir '{self.raw_dir}' already exists")

    def _compute_dataset_data(
            self
    ) -> None:
        super()._compute_dataset_data()

        if self.results_dir.exists():
            # Just read
            self.info = DatasetInfo.read(self.info_path)
            self.dataset = LocalDataset(None, self.results_dir, **self._params)

        elif is_in_torch_geometric_datasets((self._domain, self._group, self._name)):
            # Download specific dataset
            # TODO Kirill, all torch-geometric datasets
            if self._group in ["pytorch-geometric-other"]:
                dataset_cls = import_by_name(self._name, ['torch_geometric.datasets'])
                if 'root' in str(inspect.signature(dataset_cls.__init__)):
                    self.dataset = dataset_cls(root=str(self.root_dir), **self._params)
                    self.move_processed(self.root_dir / 'processed')
                else:
                    # TODO misha or Kirill have get params,
                    #  https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.BAShapes.html#torch_geometric.datasets.BAShapes
                    #  e.g. BAShapes, other/PCPNetDataset etc
                    self.dataset = dataset_cls(**self._params)
                    if not os.path.exists(self.results_dir):
                        os.makedirs(self.results_dir)
                    torch.save(obj=(self.dataset.data, self.dataset.slices),
                               f=self.results_dir / 'data.pt')
            else:
                dataset_cls = import_by_name(self._group, ['torch_geometric.datasets'])
                self.dataset = dataset_cls(root=self.root_dir.parent, name=self._name, **self._params)
                # QUE Kirill, maybe we can do it some other way
                if self.name == 'PROTEINS':
                    torch.save((self.dataset.data, self.dataset.slices),
                               self.dataset.processed_paths[0])
                if self._group in ["GEDDataset"]:
                    root = self.root_dir.parent
                else:
                    root = self.root_dir

                self.move_processed(root / 'processed')
                self.move_raw(root / 'raw')

            # Define and save DatasetInfo
            self.info = DatasetInfo.induce(self.dataset)
            self.info.save(self.info_path)
        else:
            raise RuntimeError()

    def _compute_dataset_var_data(
            self
    ) -> None:
        pass  # Nothing to compute


def is_in_torch_geometric_datasets(
        full_name: tuple = None
) -> bool:
    from data_structures.prefix_storage import PrefixStorage
    with open(TORCH_GEOM_GRAPHS_PATH, 'r') as f:
        return PrefixStorage.from_json(f.read()).check(full_name)
