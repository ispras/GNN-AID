import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Union, Callable

import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
from torch_geometric.data.data import BaseData

from aux.utils import import_by_name, root_dir, root_dir_len
from base.datasets_processing import GeneralDataset, is_in_torch_geometric_datasets, DatasetInfo
from aux.configs import DatasetConfig, DatasetVarConfig, ConfigPattern


class PTGDataset(
    GeneralDataset
):
    """ Contains a PTG dataset.
    """
    attr_name = 'unknown'
    dataset_var_config = DatasetVarConfig(
        features={'attr': {attr_name: 'other'}},
        labeling="origin",
        dataset_ver_ind=0
    )

    def __init__(
            self,
            dataset_config: Union[ConfigPattern, DatasetConfig],
            **kwargs
    ):
        """
        :param dataset_config: dataset config dictionary
        :param kwargs: additional args to init torch dataset class
        """
        super(PTGDataset, self).__init__(dataset_config)
        self.dataset_var_config = PTGDataset.dataset_var_config.copy()

        dataset_group = dataset_config.group
        dataset_name = dataset_config.graph

        if dataset_group == 'api':
            if self.api_path.exists():
                self.info = DatasetInfo.read(self.info_path)

                with self.api_path.open('r') as f:
                    api = json.load(f)

                import_path = Path(api['import_path'])

                # Parse import path and locate module
                import sys
                imp = None
                parts = import_path.parts
                # If submodule of current project
                if all(root_dir.parts[i] == parts[i] for i in range(root_dir_len)):
                    # Remove extension, replace '/' -> '.'
                    imp = '.'.join(
                        list(parts[root_dir_len: -1]) + [import_path.stem])
                else:
                    raise NotImplementedError(
                        "User dataset should be implemented as a part of the project")
                #     # Check whether it is in python path and add relative to it
                #     for ppath in sys.path:
                #         ppath = Path(ppath)
                #         # if pythonpath is prefix
                #         if all(ppath.parts[i] == parts[i] for i in range(len(ppath.parts))):
                #             imp = '.'.join(
                #                 list(parts[len(ppath.parts) + 1: -1]) + [import_path.stem])
                #             break
                #
                # if imp is None:
                #     # Not found - add to python path
                #     path = import_path.parent.absolute()
                #     sys.path.append(str(path))
                #     imp = '.'.join(
                #         list(parts[len(path.parts) + 1: -1]) + [import_path.stem])

                from pydoc import locate
                self.dataset: Dataset = locate(f"{imp}.{api['obj_name']}")
                if self.dataset is None:
                    raise ImportError(f"Couldn't import user dataset from {imp} as {api['obj_name']}")
                else:
                    print(f"Imported user dataset from {imp} as {api['obj_name']}")
            else:
                # Do not know how to load data, hope to get dataset later
                pass

        elif self.results_dir.exists():
            # Just read
            self.info = DatasetInfo.read(self.info_path)
            self.dataset = LocalDataset(self.results_dir, **kwargs)

        else:
            if is_in_torch_geometric_datasets(dataset_config.full_name()):
                # Download specific dataset
                # TODO Kirill, all torch-geometric datasets
                if dataset_group in ["pytorch-geometric-other"]:
                    dataset_cls = import_by_name(dataset_name, ['torch_geometric.datasets'])
                    if 'root' in str(inspect.signature(dataset_cls.__init__)):
                        self.dataset = dataset_cls(root=str(self.root_dir), **kwargs)
                        self.move_processed(self.root_dir / 'processed')
                    else:
                        # TODO misha or Kirill have get params,
                        #  https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.BAShapes.html#torch_geometric.datasets.BAShapes
                        #  e.g. BAShapes, other/PCPNetDataset etc
                        self.dataset = dataset_cls(**kwargs)
                        if not os.path.exists(self.results_dir):
                            os.makedirs(self.results_dir)
                        torch.save(obj=(self.dataset.data, self.dataset.slices),
                                   f=self.results_dir / 'data.pt')
                else:
                    dataset_cls = import_by_name(dataset_group, ['torch_geometric.datasets'])
                    self.dataset = dataset_cls(root=self.root_dir.parent, name=dataset_name, **kwargs)
                    # QUE Kirill, maybe we can do it some other way
                    if dataset_name == 'PROTEINS':
                        torch.save((self.dataset.data, self.dataset.slices), self.dataset.processed_paths[0])
                    if dataset_group in ["GEDDataset"]:
                        root = self.root_dir.parent
                    else:
                        root = self.root_dir

                    self.move_processed(root / 'processed')
                    self.move_raw(root / 'raw')

                # Define and save DatasetInfo
                self.info = DatasetInfo.induce(self.dataset)
                self.info.save(self.info_path)

            else:  # Not lib graph nor the folder exists
                # Do not know how to load data, hope to get dataset later
                pass
                # raise FileNotFoundError(
                #     f"No data found for dataset '{self.dataset_config.full_name()}'")

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
            self,
            center=None,
            depth: Union[int, None] = None
    ) -> None:
        # assert len(name_type) == 1  # FIXME
        dataset_data = super()._compute_dataset_data()
        # FIXME add features

        return dataset_data

    def build(
            self,
            dataset_var_config: dict = None
    ) -> None:
        """ PTG dataset is already built
        """
        # Use cached ptg dataset. Only default dataset_var_config is allowed.
        assert self.dataset_var_config == dataset_var_config


class LocalDataset(
    InMemoryDataset
):
    """ Locally saved PTG Dataset.
    """

    def __init__(
            self,
            results_dir: Union[str, Path],
            process_func: Union[Callable, None] = None,
            **kwargs
    ):
        """

        :param results_dir:
        :param process_func:
        :param kwargs:
        """
        self.results_dir = results_dir
        if process_func:
            self.process = process_func
        # Init and process if needed
        super().__init__(None, **kwargs)

        # Load
        self.data, *rest_data = torch.load(self.processed_paths[0])
        self.slices = None
        try:
            self.slices = rest_data[0]
            # TODO can use rest_data[1] ?
        except IndexError:
            pass

    @property
    def processed_file_names(
            self
    ) -> str:
        return 'data.pt'

    def process(
            self
    ) -> None:
        raise RuntimeError("Dataset is supposed to be processed and saved earlier.")
        # torch.save(self.collate(self.data_list), self.processed_paths[0])

    @property
    def processed_dir(
            self
    ) -> str:
        return self.results_dir


def is_graph_directed(
        data: Union[Data, BaseData]
) -> bool:
    """ Detect whether graph is directed or not (for each edge i->j, exists j->i).
    """
    # Note: this does not work correctly. E.g. for TUDataset/MUTAG it incorrectly says directed.
    # return not data.is_undirected()

    edges = data.edge_index.tolist()
    edges_set = set()
    directed_flag = True
    undirected_edges = 0
    for i, elem in enumerate(edges[0]):
        if (edges[1][i], edges[0][i]) not in edges_set:
            edges_set.add((edges[0][i], edges[1][i]))
        else:
            undirected_edges += 1
    if undirected_edges == len(edges[0]) / 2:
        directed_flag = False
    return directed_flag
