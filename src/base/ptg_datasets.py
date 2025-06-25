import inspect
import os
import shutil
from pathlib import Path
from time import time
from typing import Union, List

import torch
from torch_geometric.data import Data

from aux.utils import import_by_name, TORCH_GEOM_GRAPHS_PATH
from base.gen_dataset import DatasetInfo, GeneralDataset, LocalDataset
from aux.configs import DatasetConfig, DatasetVarConfig, ConfigPattern


class aPTGDataset(GeneralDataset):
    """
    Generalisation and a wrapper over a single ptg Dataset.
    Features and labels are defined at initialisation.
    You should extend from this class.
    """
    default_dataset_var_config = DatasetVarConfig(
        features={'attr': {'unknown': 'vector'}},
        labeling="origin",
        dataset_ver_ind=0
    )

    def __init__(
            self,
            dataset_config: Union[ConfigPattern, DatasetConfig],
            **ptg_kwargs
    ):
        """
        :param dataset_config: dataset config dictionary
        :param ptg_kwargs: additional args to init torch_geometric.data.Dataset
        """
        # self.dataset_config = dataset_config
        # # todo create and save metainfo
        # print(self.info_path)

        super(aPTGDataset, self).__init__(dataset_config)

    def _compute_dataset_data(
            self
    ) -> None:
        self.dataset_var_config = aPTGDataset.default_dataset_var_config.copy()

    def _compute_dataset_var_data(
            self
    ) -> None:
        raise RuntimeError(f"{self.__class__.__name__} does not compute var data.")


class LocalPTGDataset(aPTGDataset):
    """
    A single ptg Dataset defined online by tensors.
    """
    def __init__(
            self,
            data_list: List[Data],
            name: Union[str, None] = None,
    ):
        """
        :param data_list: list of ready torch_geometric.data.Data objects
        :param name: unique dataset name
        """
        self.data_list = data_list

        domain = 'single-graph' if len(data_list) == 1 else 'multiple-graphs'
        group = 'local-ptg-datasets'
        name = name or 'graph_' + str(time())
        dataset_config = DatasetConfig((domain, group, name),)

        super(LocalPTGDataset, self).__init__(dataset_config)

    def _compute_dataset_data(
            self
    ) -> None:
        super()._compute_dataset_data()
        # Create local ptg dataset

        self.dataset = LocalDataset(self.data_list, self.results_dir)
        self.info = DatasetInfo.induce(self.dataset)
        self.info.save(self.info_path)

        # Define auxiliary fields
        self.dataset_data = {}
        if self.is_hetero():
            raise NotImplementedError

        else:
            self.dataset_data['edges'] = [data.edge_index for data in self.data_list]
            self.dataset_data['node_attributes'] = [data.x for data in self.data_list]

    def _compute_dataset_var_data(
            self
    ) -> None:
        # Define auxiliary fields
        self.dataset_var_data = {}
        if self.is_hetero():
            raise NotImplementedError

        else:
            self.dataset_var_data['node_features'] = [data.x for data in self.data_list]
            # self.dataset_var_data['edge_features'] = self.dataset_data['edge_attributes'] = []
            self.dataset_var_data['labels'] = [data.y for data in self.data_list]


class LibPTGDataset(aPTGDataset):
    """
    A single ptg Dataset from Pytorch-Geometric library.
    """
    def __init__(
            self,
            domain: str,
            group: str,
            name: str,
            **params
    ):
        """
        :param group: group name, e.g. 'TUDataset'
        :param name: dataset name, e.g. 'MUTAG'
        :param params: optional init parameters
        """
        assert domain in ['single-graph', 'multiple-graphs']
        self._domain = domain
        self._group = group
        self._name = name
        self._params = params
        dataset_config = DatasetConfig((domain, group, name),)

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

        elif is_in_torch_geometric_datasets(self.dataset_config.full_name):
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

        # Define auxiliary fields
        self.dataset_data = {}
        if self.is_hetero():
            raise NotImplementedError

        else:
            self.dataset_data['edges'] = [data.edge_index for data in self.dataset]
            self.dataset_data['node_attributes'] = [data.x for data in self.dataset]

    def _compute_dataset_var_data(
            self
    ) -> None:
        # Define auxiliary fields
        self.dataset_var_data = {}
        if self.is_hetero():
            raise NotImplementedError

        else:
            self.dataset_var_data['node_features'] = [data.x for data in self.dataset]
            # self.dataset_var_data['edge_features'] = self.dataset_data['edge_attributes'] = []
            self.dataset_var_data['labels'] = [data.y for data in self.dataset]
        if not self.is_multi():
            self.dataset_var_data['node_features'] = self.dataset_var_data['node_features'][0]
            # self.dataset_var_data['edge_features'] = []
            self.dataset_var_data['labels'] = self.dataset_var_data['labels'][0]


# class PTGDataset(
#     GeneralDataset
# ):
#     """ Contains a PTG dataset.
#     """
#     attr_name = 'unknown'
#     dataset_var_config = DatasetVarConfig(
#         features={'attr': {attr_name: 'vector'}},
#         labeling="origin",
#         dataset_ver_ind=0
#     )
# 
#     def __init__(
#             self,
#             dataset_config: Union[ConfigPattern, DatasetConfig],
#             **kwargs
#     ):
#         """
#         :param dataset_config: dataset config dictionary
#         :param kwargs: additional args to init torch dataset class
#         """
#         super(PTGDataset, self).__init__(dataset_config)
#         self.dataset_var_config = PTGDataset.dataset_var_config.copy()
# 
#         dataset_group = dataset_config.group
#         dataset_name = dataset_config.full_name[-1]
# 
#         if dataset_group == 'api':
#             if self.api_path.exists():
#                 self.info = DatasetInfo.read(self.info_path)
# 
#                 with self.api_path.open('r') as f:
#                     api = json.load(f)
# 
#                 import_path = Path(api['import_path'])
# 
#                 # Parse import path and locate module
#                 import sys
#                 imp = None
#                 parts = import_path.parts
#                 # If submodule of current project
#                 if all(root_dir.parts[i] == parts[i] for i in range(root_dir_len)):
#                     # Remove extension, replace '/' -> '.'
#                     imp = '.'.join(
#                         list(parts[root_dir_len: -1]) + [import_path.stem])
#                 else:
#                     raise NotImplementedError(
#                         "User dataset should be implemented as a part of the project")
#                 #     # Check whether it is in python path and add relative to it
#                 #     for ppath in sys.path:
#                 #         ppath = Path(ppath)
#                 #         # if pythonpath is prefix
#                 #         if all(ppath.parts[i] == parts[i] for i in range(len(ppath.parts))):
#                 #             imp = '.'.join(
#                 #                 list(parts[len(ppath.parts) + 1: -1]) + [import_path.stem])
#                 #             break
#                 #
#                 # if imp is None:
#                 #     # Not found - add to python path
#                 #     path = import_path.parent.absolute()
#                 #     sys.path.append(str(path))
#                 #     imp = '.'.join(
#                 #         list(parts[len(path.parts) + 1: -1]) + [import_path.stem])
# 
#                 from pydoc import locate
#                 self.dataset: Dataset = locate(f"{imp}.{api['obj_name']}")
#                 if self.dataset is None:
#                     raise ImportError(f"Couldn't import user dataset from {imp} as {api['obj_name']}")
#                 else:
#                     print(f"Imported user dataset from {imp} as {api['obj_name']}")
#             else:
#                 # Do not know how to load data, hope to get dataset later
#                 pass
# 
#         elif self.results_dir.exists():
#             # Just read
#             self.info = DatasetInfo.read(self.info_path)
#             self.dataset = LocalDataset(self.results_dir, **kwargs)
# 
#         # else:
#         #     if is_in_torch_geometric_datasets(dataset_config.full_name()):
#         #         # Download specific dataset
#         #         # TODO Kirill, all torch-geometric datasets
#         #         if dataset_group in ["pytorch-geometric-other"]:
#         #             dataset_cls = import_by_name(dataset_name, ['torch_geometric.datasets'])
#         #             if 'root' in str(inspect.signature(dataset_cls.__init__)):
#         #                 self.dataset = dataset_cls(root=str(self.root_dir), **kwargs)
#         #                 self.move_processed(self.root_dir / 'processed')
#         #             else:
#         #                 # TODO misha or Kirill have get params,
#         #                 #  https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.BAShapes.html#torch_geometric.datasets.BAShapes
#         #                 #  e.g. BAShapes, other/PCPNetDataset etc
#         #                 self.dataset = dataset_cls(**kwargs)
#         #                 if not os.path.exists(self.results_dir):
#         #                     os.makedirs(self.results_dir)
#         #                 torch.save(obj=(self.dataset.data, self.dataset.slices),
#         #                            f=self.results_dir / 'data.pt')
#         #         else:
#         #             dataset_cls = import_by_name(dataset_group, ['torch_geometric.datasets'])
#         #             self.dataset = dataset_cls(root=self.root_dir.parent, name=dataset_name, **kwargs)
#         #             # QUE Kirill, maybe we can do it some other way
#         #             if dataset_name == 'PROTEINS':
#         #                 torch.save((self.dataset.data, self.dataset.slices), self.dataset.processed_paths[0])
#         #             if dataset_group in ["GEDDataset"]:
#         #                 root = self.root_dir.parent
#         #             else:
#         #                 root = self.root_dir
#         #
#         #             self.move_processed(root / 'processed')
#         #             self.move_raw(root / 'raw')
#         #
#         #         # Define and save DatasetInfo
#         #         self.info = DatasetInfo.induce(self.dataset)
#         #         self.info.save(self.info_path)
#         #
#         #     else:  # Not lib graph nor the folder exists
#         #         # Do not know how to load data, hope to get dataset later
#         #         pass
#         #         # raise FileNotFoundError(
#         #         #     f"No data found for dataset '{self.dataset_config.full_name}'")
# 
#         if self.info.hetero:
#             self.node_types = list(self.info.nodes[0].keys())
#             self.edge_types = list(self.info.edge_attributes.keys())
# 
#     def move_processed(
#             self,
#             processed: Union[str, Path]
#     ) -> None:
#         if not self.results_dir.exists():
#             self.results_dir.mkdir(parents=True)
#             os.rename(processed, self.results_dir)
#         else:
#             shutil.rmtree(processed)
# 
#     def move_raw(
#             self,
#             raw: Union[str, Path]
#     ) -> None:
#         if Path(raw) == self.raw_dir:
#             return
#         if not self.raw_dir.exists():
#             self.raw_dir.mkdir(parents=True)
#             os.rename(raw, self.raw_dir)
#         else:
#             raise RuntimeError(f"raw_dir '{self.raw_dir}' already exists")
# 
#     def _compute_dataset_data(
#             self,
#             center=None,
#             depth: Union[int, None] = None
#     ) -> None:
#         # TODO misha hetero
#         name_type = self.dataset_var_config.features['attr']
#         # node_attributes == PTG features
#         assert len(name_type) == 1
#         num = len(self.dataset)
#         data_list = [self.dataset.get(ix) for ix in range(num)]
#         is_directed = self.info.directed
# 
#         # FIXME misha self.dataset_data['edges'] == ptg edge_index, make them one
#         edges_list = []
#         if self.is_multi():
#             for data in data_list:
#                 edges_list.append(data.edge_index.T.tolist())
# 
#             node_attributes = {
#                 list(name_type.keys())[0]: [data.x.tolist() for data in data_list]
#             }
# 
#         else:
#             assert len(data_list) == 1
#             data = data_list[0]
# 
#             if self.info.hetero:
#                 node_attributes = {
#                     nt: {
#                         list(name_type.keys())[0]: [data[nt].x.tolist()]
#                     } for nt in self.node_types
#                 }
#             else:
#                 node_attributes = {
#                     list(name_type.keys())[0]: [data.x.tolist()]
#                 }
# 
#         for data in data_list:
#             edges = []
#             if self.info.hetero:
#                 from base.heterographs import edge_type_from_str
# 
#                 edges = {}
#                 for et in self.edge_types:
#                     edges[et] = []
#                     et_3 = edge_type_from_str(et)
#                     dataset_edge_index = data[et_3].edge_index.tolist()
#                     for i in range(len(dataset_edge_index[0])):
#                         if not is_directed:
#                             if dataset_edge_index[0][i] <= dataset_edge_index[1][i]:
#                                 edges[et].append([dataset_edge_index[0][i], dataset_edge_index[1][i]])
#                         else:
#                             edges[et].append([dataset_edge_index[0][i], dataset_edge_index[1][i]])
# 
#             else:
#                 dataset_edge_index = data.edge_index.tolist()
#                 for i in range(len(dataset_edge_index[0])):
#                     if not is_directed:
#                         if dataset_edge_index[0][i] <= dataset_edge_index[1][i]:
#                             edges.append([dataset_edge_index[0][i], dataset_edge_index[1][i]])
#                     else:
#                         edges.append([dataset_edge_index[0][i], dataset_edge_index[1][i]])
# 
#             edges_list.append(edges)
# 
#         self.dataset_data = {
#             'edges': edges_list,
#             'node_attributes': node_attributes,
#             # 'info': self.info.to_dict()
#         }
#         # if self.info.name == "":
#         #     self.dataset_data['info']['name'] = '/'.join(self.dataset_config.full_name())
# 
#     def build(
#             self,
#             dataset_var_config: dict = None
#     ) -> None:
#         """ PTG dataset is already built
#         """
#         # Use cached ptg dataset. Only default dataset_var_config is allowed.
#         assert self.dataset_var_config == dataset_var_config


# class LocalDataset(
#     InMemoryDataset
# ):
#     """ Locally saved PTG Dataset.
#     """
#
#     def __init__(
#             self,
#             results_dir: Union[str, Path],
#             process_func: Union[Callable, None] = None,
#             **kwargs
#     ):
#         """
#
#         :param results_dir:
#         :param process_func:
#         :param kwargs:
#         """
#         self.results_dir = results_dir
#         if process_func:
#             self.process = process_func
#         # Init and process if needed
#         super().__init__(None, **kwargs)
#
#         # Load
#         self.data, *rest_data = torch.load(self.processed_paths[0])
#         self.slices = None
#         try:
#             self.slices = rest_data[0]
#             # TODO can use rest_data[1] ?
#         except IndexError:
#             pass
#
#     @property
#     def processed_file_names(
#             self
#     ) -> str:
#         return 'data.pt'
#
#     def process(
#             self
#     ) -> None:
#         raise RuntimeError("Dataset is supposed to be processed and saved earlier.")
#         # torch.save(self.collate(self.data_list), self.processed_paths[0])
#
#     @property
#     def processed_dir(
#             self
#     ) -> str:
#         return self.results_dir


def is_in_torch_geometric_datasets(
        full_name: tuple = None
) -> bool:
    from aux.prefix_storage import PrefixStorage
    with open(TORCH_GEOM_GRAPHS_PATH, 'r') as f:
        return PrefixStorage.from_json(f.read()).check(full_name)
