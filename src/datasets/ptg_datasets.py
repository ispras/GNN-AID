import inspect
import os
import shutil
from pathlib import Path
from time import time
from typing import Union, List, Dict

import torch
from torch_geometric.data import Data, HeteroData

from aux.utils import import_by_name, TORCH_GEOM_GRAPHS_PATH
from datasets.gen_dataset import GeneralDataset, LocalDataset
from datasets.dataset_info import DatasetInfo
from data_structures.configs import DatasetConfig, DatasetVarConfig, ConfigPattern, FeatureConfig

PTG_FEATURE_NAME = "unknown"


class PTGDataset(GeneralDataset):
    """
    Generalisation and a wrapper over a single ptg Dataset.
    Features and labels are defined at initialisation.
    Extend this class if you have a dataset extending :class:`torch_geometric.data.Dataset`.

    """
    default_dataset_var_config = DatasetVarConfig(
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

        self.node_attr_slices = {PTG_FEATURE_NAME: (0, self.dataset.data.x.shape[1])}

        if results_exist:
            # Read info
            self.info = DatasetInfo.read(self.metainfo_path)

        else:  # first time
            self.prepared_dir.parent.mkdir(parents=True, exist_ok=True)
            if self.prepared_dir != self.dataset.processed_dir\
                    and Path(self.dataset.processed_dir).is_dir():
                # Create link to original processed files
                # (we do not move them to avoid torch graph calling process() each time)
                self.prepared_dir.symlink_to(self.dataset.processed_dir, target_is_directory=True)

            # Define and save DatasetInfo
            self.info = self.induce_dataset_info()
            self.info.save(self.metainfo_path)

    def _compute_dataset_var_data(
            self
    ) -> None:
        pass  # PTG graphs have tensors by default

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
        # res.directed = is_graph_directed(dataset.get(0))
        data = self.dataset.get(0)
        res.directed = data.is_directed()  # FIXME misha check correct
        if isinstance(data, HeteroData):
            res.hetero = True
            node_types = data.node_types
            res.nodes = [{nt: data[nt].num_nodes for nt in node_types}]
            res.node_attributes = {
                nt: {
                    # "names": [],
                    # "types": [],
                    # "values": []
                    "names": [PTG_FEATURE_NAME],
                    "types": ["vector"],
                    "values": [len(data[nt].x[0])]
                } for nt in node_types
            }
            edge_types = [','.join([f'"{x}"' for x in et]) for et in data.edge_types]
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
                if hasattr(data[nt], 'y'):
                    res.labelings[nt] = {"origin": int(max(data[nt].y)) + 1}
        else:
            res.hetero = False
            res.nodes = [len(self.dataset.get(ix).x) for ix in range(len(self.dataset))]
            res.node_attributes = {
                # "names": [],
                # "types": [],
                # "values": []
                "names": [PTG_FEATURE_NAME],
                "types": ["vector"],
                "values": [len(self.dataset.get(0).x[0])]
            }
            res.labelings = {"origin": self.dataset.num_classes}
            res.node_attr_slices = res.get_attributes_slices_form_attributes(res.node_attributes, res.edge_attributes)

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
            **params
    ):
        """
        :param dataset_config: dataset config dictionary
        :param params: additional parameters to init ptg class if needed
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
        """ Move ptg processed files to folder when tenors are stored """
        if not self.prepared_dir.exists():
            self.prepared_dir.mkdir(parents=True)
            os.rename(processed, self.prepared_dir)
        else:
            shutil.rmtree(processed)

    def move_raw(
            self,
            raw: Union[str, Path]
    ) -> None:
        """ Move ptg raw files to folder when raw files are stored """
        if Path(raw) == self.raw_dir:
            return
        if not self.raw_dir.exists():
            self.raw_dir.mkdir(parents=True)
            os.rename(raw, self.raw_dir)
        else:
            raise RuntimeError(f"raw_dir '{self.raw_dir}' already exists")

    def _define_ptg_dataset(
            self
    ) -> None:

        if is_in_torch_geometric_datasets((self._domain, self._group, self._name)):
            # Download specific dataset
            # TODO add all torch-geometric datasets
            if self._group in ["pytorch-geometric-other"]:
                dataset_cls = import_by_name(self._name, ['torch_geometric.datasets'])
                if 'root' in str(inspect.signature(dataset_cls.__init__)):
                    self.dataset = dataset_cls(root=str(self.root_dir), **self._params)
                    self.move_processed(self.dataset.processed_dir)
                else:
                    # TODO misha or Kirill have get params,
                    #  https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.BAShapes.html#torch_geometric.datasets.BAShapes
                    #  e.g. BAShapes, other/PCPNetDataset etc
                    self.dataset = dataset_cls(**self._params)
                    if not os.path.exists(self.prepared_dir):
                        os.makedirs(self.prepared_dir)
                    torch.save(obj=(self.dataset.data, self.dataset.slices),
                               f=self.prepared_dir / 'data.pt')
            else:
                dataset_cls = import_by_name(self._group, ['torch_geometric.datasets'])
                self.dataset = dataset_cls(root=self.root_dir.parent, name=self._name, **self._params)
                # QUE Kirill, maybe we can do it some other way
                if self._name == 'PROTEINS':
                    torch.save((self.dataset.data, self.dataset.slices),
                               self.dataset.processed_paths[0])
                if self._group in ["GEDDataset"]:
                    root = self.root_dir.parent
                else:
                    root = self.root_dir

            #     self.move_processed(self.dataset.processed_dir)
            #     self.move_raw(self.dataset.raw_dir)
            #
            # # Define and save DatasetInfo
            # self.info = self.induce_dataset_info()
            # self.info.save(self.metainfo_path)
        else:
            raise RuntimeError()


def is_in_torch_geometric_datasets(
        full_name: tuple = None
) -> bool:
    from data_structures.prefix_storage import FixedKeysPrefixStorage
    with open(TORCH_GEOM_GRAPHS_PATH, 'r') as f:
        return FixedKeysPrefixStorage.from_json(f.read(), ).check(full_name)
