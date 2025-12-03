import copy
import inspect
import warnings
from pathlib import Path
from time import time
from typing import Union, List, Dict

import torch
from torch_geometric.data import Data, HeteroData, Dataset, InMemoryDataset

from gnn_aid.aux.declaration import Declare
from gnn_aid.aux.utils import import_by_name, shape
from gnn_aid.data_structures.configs import (
    DatasetConfig, DatasetVarConfig, ConfigPattern, FeatureConfig, Task)
from .dataset_info import DatasetInfo
from .gen_dataset import GeneralDataset, LocalDataset

PTG_FEATURE_NAME = "unknown"


class PTGDataset(GeneralDataset):
    """
    Generalisation and a wrapper over a single ptg Dataset.
    Features and labels are defined at initialisation.
    Extend this class if you have a dataset extending :class:`torch_geometric.data.Dataset`.

    """
    default_dataset_var_config = DatasetVarConfig(
        task=None,  # not defined
        features=FeatureConfig(node_attr=[PTG_FEATURE_NAME]),
        # features=FeatureConfig(),
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
        """
        super(PTGDataset, self).__init__(dataset_config)

    def _compute_dataset_data(
            self
    ) -> None:
        self.dataset_var_config = PTGDataset.default_dataset_var_config.copy()
        results_exist = self.prepared_dir.exists()

        self._define_ptg_dataset()  # creates tensors and save them to self.prepared_dir

        assert isinstance(self.dataset, Dataset)
        assert isinstance(self.dataset[0], Data)
        # TODO Data can have no attribute 'x', do we support this?
        assert self.dataset[0].x is not None, "Data objects must have 'x' attribute"
        assert self.dataset[0].edge_index is not None, "Data objects must have 'edge_index' attribute"

        x_shape = shape(self.dataset[0].x)
        node_feats = x_shape[1] if len(x_shape) > 1 else 1
        assert node_feats == self.dataset.num_node_features
        self.node_attr_slices = {PTG_FEATURE_NAME: (0, node_feats)}

        if results_exist:
            # Read info
            self.info = DatasetInfo.read(self.metainfo_path)

        else:  # first time
            # Define and save DatasetInfo
            self.info = self.induce_dataset_info()
            self.info.save(self.metainfo_path)

            # # Save json with config in data/
            # _, files_paths = Declare.dataset_root_dir(self.dataset_config)
            # with open(files_paths[0], "w") as f:
            #     f.write(self.dataset_config.json_for_config())

            # Save json with configs in datasets/
            _, files_paths = Declare.dataset_prepared_dir(
                self.dataset_config, self.dataset_var_config)
            with open(files_paths[0], "w") as f:
                f.write(self.dataset_config.json_for_config())
            with open(files_paths[1], "w") as f:
                f.write(self.dataset_var_config.json_for_config())

    def _compute_dataset_var_data(
            self
    ) -> None:
        # PTG graphs have tensors by default
        # Only different task is allowed
        # assert self.dataset_var_config.features == PTGDataset.default_dataset_var_config.features
        assert self.dataset_var_config.task in [
            Task.NODE_CLASSIFICATION, Task.GRAPH_CLASSIFICATION, Task.EDGE_PREDICTION]

    def _define_ptg_dataset(
            self
    ) -> None:
        """ This function creates all tensors and save them to self.prepared_dir
        """
        raise RuntimeError("This should be implemented in subclass")

    def node_attributes(
            self,
            attrs: List[str] = None
    ) -> Dict[str, Union[list, torch.Tensor]]:
        """ Get node attributes as a dict {name -> list}"""
        # return {}  # features are not attributes
        assert attrs is None or attrs == [] or attrs == [PTG_FEATURE_NAME]
        return {PTG_FEATURE_NAME: [data.x.tolist() for data in self.dataset]}

    def edge_attributes(
            self,
            attrs: List[str] = None
    ) -> Dict[str, Union[list, torch.Tensor]]:
        """ Get edge attributes as a dict {name -> list}"""
        return {}  # features are not attributes

    def induce_dataset_info(
            self
    ) -> DatasetInfo:
        """ Induce metainfo for the PTG dataset.
        """
        res = DatasetInfo()
        res.count = len(self.dataset)
        # from datasets.ptg_datasets import is_graph_directed
        # res.directed = is_graph_directed(dataset[0])
        msg = ""
        data0 = self.dataset[0]
        if data0.edge_index is not None:
            res.directed = data0.is_directed()  # FIXME misha check correct
        if isinstance(data0, HeteroData):
            res.hetero = True
            node_types = data0.node_types
            res.nodes = [{nt: data0[nt].num_nodes for nt in node_types}]
            res.node_attributes = {
                nt: {
                    # "names": [],
                    # "types": [],
                    # "values": []
                    "names": [PTG_FEATURE_NAME],
                    "types": ["vector"],
                    "values": [len(data0[nt].x[0])]
                } for nt in node_types
            }
            edge_types = [','.join([f'"{x}"' for x in et]) for et in data0.edge_types]
            res.edge_attributes = {
                et: {
                    "names": [],
                    "types": [],
                    "values": []
                } for et in edge_types
            }
            # TODO add edge attributes
            res.labelings = {}
            for nt in node_types:
                if hasattr(data0[nt], 'y'):
                    res.labelings[nt] = {"origin": int(max(data0[nt].y)) + 1}

        elif isinstance(data0, Data):
            res.hetero = False
            res.nodes = [data.num_nodes for data in self.dataset]

            if hasattr(self.dataset, 'num_classes'):
                if self.dataset.num_classes == 0:
                    raise NotImplementedError("Datasets without classes are not supported")
                classification_task = Task.NODE_CLASSIFICATION if len(self.dataset) == 1 else Task.GRAPH_CLASSIFICATION
                # TODO if edges have attributes, add edge classification
                res.labelings = {
                    classification_task: {"origin": self.dataset.num_classes},
                }
            else:
                res.labelings = "?"
                msg += f"Cannot get num_classes Dataset of type {self.dataset.__class__}. "

            if self.dataset[0].x is not None:
                res.node_attributes = {
                    "names": [PTG_FEATURE_NAME],
                    "types": ["vector"],
                    "values": [shape(self.dataset[0].x)[0]]
                }

            else:
                res.node_attributes = {
                    "names": ["?"],
                    "types": ["?"],
                    "values": ["?"]
                }
                msg += "Cannot induce node_attributes for such type of Data that have no 'x'" \
                       " attribute. "

        else:
            raise NotImplementedError(
                f"Cannot induce metainfo for Dataset that contains objects of type"
                f" {self.dataset[0].__class__}. Only torch_geometric.data.Data is supported.")

        if msg:
            msg = "Cannot induce metainfo for this kind of data." \
                  " Metainfo file is created, but you need to finish it manually. " + msg
            warnings.warn(msg)
        else:
            res.check()
        return res


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
        :param data_list: list of ready :class:`torch_geometric.data.data.Data` objects
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

    def _define_ptg_dataset(
            self
    ) -> None:
        # Create local ptg dataset

        if self.prepared_dir.exists():
            self.dataset = LocalDataset(None, self.prepared_dir)
        else:
            self.dataset = LocalDataset(self.data_list, self.prepared_dir)


class LibPTGDataset(PTGDataset):
    """
    A single ptg Dataset from Pytorch-Geometric library.
    """
    data_folder = 'ptg-library-graphs'

    def __init__(
            self,
            dataset_config: DatasetConfig,
            **ptg_init_kwargs
    ):
        """
        :param dataset_config: dataset config dictionary
        :param ptg_init_kwargs: additional parameters to init ptg class, e.g. constructor params or
         torch_geometric.transforms. Possible only if dataset is not created yet.
        """
        first, self._domain, self._group, *rest = dataset_config.full_name
        self._name = rest[0] if len(rest) > 0 else None

        assert first == self.data_folder
        assert self._domain in ["Homogeneous", "Heterogeneous", "Synthetic"]
        if self._domain == "Heterogeneous":
            raise NotImplementedError("Heterogeneous ptg datasets are not supported yet")

        self._ptg_init_kwargs = copy.deepcopy(ptg_init_kwargs)

        super(LibPTGDataset, self).__init__(dataset_config)

    def _define_ptg_dataset(
            self
    ) -> None:
        dataset_cls = import_by_name(self._group, ['torch_geometric.datasets'])

        has_root = 'root' in str(inspect.signature(dataset_cls.__init__).parameters.keys())
        has_name = 'name' in str(inspect.signature(dataset_cls.__init__).parameters.keys())
        if has_root:
            self._ptg_init_kwargs['root'] = str(self.raw_dir)
        if has_name:
            self._ptg_init_kwargs['name'] = self._name

        # If tensors exist, load as a LocalDataset
        if self.prepared_dir.exists():
            if has_root:
                # PTG dataset should omit downloading and processing
                self.dataset = dataset_cls(**self._ptg_init_kwargs)

            else:
                # Just load as LocalDataset to avoid re-generating tensors within PTG class

                # if self._ptg_init_kwargs:
                #     warnings.warn(f"Dataset {self.dataset_config} already exists, so ptg init kwargs"
                #                   f" will be ignored.")

                # NOTE: this works if `processed_file_names` does not use self - we pass None.
                # We do not want to create object since it may make unnecessary computations
                # processed_file_names = lambda: dataset_cls.processed_file_names.fget(None)
                self.dataset = LocalDataset(None, prepared_dir=self.prepared_dir)
            return

        self.dataset = dataset_cls(**self._ptg_init_kwargs)

        if has_root:
            # Move (link) processed data to self.prepared_dir
            self.move_processed_to_prepared()

        elif issubclass(dataset_cls, InMemoryDataset):
            # InMemoryDatasets do not save data by default

            # Dataset is generated, we save it manually
            self.dataset = dataset_cls(**self._ptg_init_kwargs)
            self.prepared_dir.mkdir(parents=True, exist_ok=True)

            # Take self.data since it applies transforms if any to get final data.
            torch.save(obj=(self.data, self.dataset.slices), f=self.prepared_dir / 'data.pt')

            # And create an empty raw folder so framework can recognize a dataset
            self.raw_dir.mkdir(parents=True)

        else:
            # The problem is that self.data will read all dataset into memory, but it could
            # be too large.
            raise NotImplementedError("Non InMemoryDataset without root are not supported yet")

        if self.data.x is not None and len(shape(self.data.x)) == 1:
            # Some datasets, e.g. AQSOL, have Data.x of shape (N) instead of (N, 1)
            self.data.x = torch.reshape(self.data.x, (shape(self.data.x)[0], 1))

    def move_processed_to_prepared(
            self
    ) -> None:
        """ Move (link) processed data to self.prepared_dir
        """
        self.prepared_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.prepared_dir != self.dataset.processed_dir \
                and Path(self.dataset.processed_dir).is_dir():
            # Create link to original processed files
            # (we do not move them to avoid torch graph calling process() each time)
            self.prepared_dir.symlink_to(self.dataset.processed_dir, target_is_directory=True)
