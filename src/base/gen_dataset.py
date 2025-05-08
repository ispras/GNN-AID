import inspect
import json
import shutil
import os
import numpy as np
from collections import OrderedDict
from pathlib import Path
from time import time
from typing import Union, List, Callable

import torch
import torch_geometric
from torch import default_generator, randperm, tensor
# from torch._C import default_generator
# from torch._C._VariableFunctions import randperm
from torch_geometric.data import Dataset, InMemoryDataset, Data, HeteroData
from torch_geometric.data.data import BaseData

from aux.configs import DatasetConfig, DatasetVarConfig, ConfigPattern
from aux.custom_decorators import timing_decorator
from aux.declaration import Declare
from aux.utils import TORCH_GEOM_GRAPHS_PATH, tmp_dir, import_by_name


class DatasetInfo:
    """
    Description for a dataset.
    Some fields are obligate, others are not.
    """

    def __init__(
            self
    ):
        self.name: str = None
        self.count: int = None
        self.directed: bool = None
        self.hetero: bool = False
        self.nodes: list = None
        self.remap: bool = False
        self.node_attributes: OrderedDict = None
        self.edge_attributes: OrderedDict = None
        self.labelings: dict = None
        self.node_attr_slices: dict = None
        self.edge_attr_slices: dict = None
        self.node_info: dict = {}
        self.edge_info: dict = {}
        self.graph_info: dict = {}

    def check_validity(
            self
    ) -> None:
        """ Check existing fields have allowed values. """
        assert self.count > 0
        assert len(self.node_attributes) > 0

        ntv_triples = []
        if self.hetero:
            for attributes in [self.node_attributes, self.edge_attributes]:
                for entity_attrs in attributes.values():
                    ntv_triples.extend(list(zip(entity_attrs["names"],
                                                entity_attrs["types"],
                                                entity_attrs["values"])))
                # ntv_triples.extend(list(zip(sum(attributes["names"].values(), []),
                #                             sum(attributes["types"].values(), []),
                #                             sum(attributes["values"].values(), []))))
        else:
            assert set(self.node_attributes.keys()) == {"names", "types", "values"}
            for attributes in [self.node_attributes, self.edge_attributes]:
                if attributes:
                    ntv_triples.extend(list(zip(attributes["names"], attributes["types"], attributes["values"])))

        for name, type, value in ntv_triples:
            assert isinstance(name, str)
            assert type in {"continuous", "categorical", "vector", "other"}
            if type == "continuous":
                assert isinstance(value, list) and len(value) == 2 and value[0] < value[1]
            elif type == "continuous":
                assert isinstance(value, list)
            elif type == "vector":
                assert isinstance(value, int) and value > 0
            elif type == "other":
                assert value in ["str", None]
        assert len(self.labelings) > 0
        if self.hetero:
            labelings = []
            for kv in self.labelings.values():
                labelings.extend(list(kv.items()))
        else:
            labelings = list(self.labelings.items())
        for k, v in labelings:
            assert isinstance(k, str)
            assert isinstance(v, int) and v >= 1  # 1 stands for regression

    def check_consistency(
            self
    ) -> None:
        """ Check existing fields are consistent. """
        assert self.count == len(self.nodes)

        if self.hetero:
            node_types = list(self.nodes[0].keys())
            edge_types = list(self.edge_attributes.keys())
            assert list(self.node_attributes.keys()) == node_types
            assert list(self.edge_attributes.keys()) == edge_types
            for nt in node_types:
                node_attributes = self.node_attributes[nt]
                assert len(node_attributes["names"]) == len(node_attributes["types"]) == len(node_attributes["values"])
            for et in edge_types:
                edge_attributes = self.edge_attributes[et]
                assert len(edge_attributes["names"]) == len(edge_attributes["types"]) == len(edge_attributes["values"])

            node_types = set(node_types)
            for et in edge_types:
                s, _, d = et.split(',')
                s = s[1:-1]
                d = d[1:-1]
                assert s in node_types and d in node_types
        else:
            assert len(self.node_attributes["names"]) == len(self.node_attributes["types"]) == len(
                self.node_attributes["values"])
            if self.edge_attributes:
                assert len(self.edge_attributes["names"]) == len(self.edge_attributes["types"]) == len(
                    self.edge_attributes["values"])

    def check_sufficiency(
            self
    ) -> None:
        """ Check all obligate fields are defined. """
        for attr in self.__dict__.keys():
            if attr is None:
                raise ValueError(f"Attribute '{attr}' of metainfo should be defined.")

    def check_consistency_with_dataset(
            self,
            dataset: Dataset
    ) -> None:
        """ Check if metainfo fields are consistent with PTG dataset. """
        assert self.count == len(dataset)
        assert self.directed == is_graph_directed(dataset.get(0))
        assert self.remap is False
        if self.hetero:
            from torch_geometric.data import HeteroData
            assert isinstance(dataset.get(0), HeteroData)
            # TODO misha hetero
        else:
            assert len(self.node_attributes["names"]) == 1
            assert self.node_attributes["types"][0] == "vector"
        # TODO check features values range

    def check(
            self
    ) -> None:
        """ Check metainfo is sufficient, consistent, and valid. """
        self.check_sufficiency()
        self.check_consistency()
        self.check_validity()

    def to_dict(
            self
    ) -> dict:
        """ Return info as a dictionary. """
        return dict(self.__dict__)

    def save(
            self,
            path: Union[str, Path]
    ) -> None:
        """ Save into file non-null info. """
        not_nones = {k: v for k, v in self.__dict__.items() if v is not None}
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open('w') as f:
            json.dump(not_nones, f, indent=1, ensure_ascii=False)

    @staticmethod
    def induce(
            dataset: Dataset
    ) -> object:
        """ Induce metainfo from a given PTG dataset.
        """
        res = DatasetInfo()
        res.count = len(dataset)
        # from base.ptg_datasets import is_graph_directed
        # res.directed = is_graph_directed(dataset.get(0))
        data = dataset.get(0)
        res.directed = data.is_directed()  # FIXME misha check correct
        if isinstance(data, HeteroData):
            res.hetero = True
            node_types = data.node_types
            res.nodes = [{nt: data[nt].num_nodes for nt in node_types}]
            res.node_attributes = {
                nt: {
                    "names": ["unknown"],
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
            res.nodes = [len(dataset.get(ix).x) for ix in range(len(dataset))]
            res.node_attributes = {
                "names": ["unknown"],
                "types": ["vector"],
                "values": [len(dataset.get(0).x[0])]
            }
            res.labelings = {"origin": dataset.num_classes}
            res.node_attr_slices = res.get_attributes_slices_form_attributes(res.node_attributes, res.edge_attributes)

        res.check()
        return res

    @staticmethod
    def read(
            path: Union[str, Path]
    ) -> object:
        """ Read info from a file. """
        with path.open('r') as f:
            a_dict = json.load(f, object_pairs_hook=OrderedDict)
        res = DatasetInfo()
        for k, v in a_dict.items():
            setattr(res, k, v)
        res.node_attr_slices, res.edge_attr_slices = res.get_attributes_slices_form_attributes(
            res.node_attributes, res.edge_attributes)
        res.check()
        return res

    @staticmethod
    def get_attributes_slices_form_attributes(
            node_attributes: dict,
            edge_attributes: dict,
    ) -> (dict, dict):
        if isinstance(next(iter(node_attributes.values())), dict):
            # TODO misha hetero
            return None, None

        node_attr_slices = {}
        if node_attributes:
            start_attr_index = 0
            for i in range(len(node_attributes['names'])):
                if node_attributes['types'][i] == 'vector':
                    attr_len = node_attributes['values'][i]
                elif node_attributes['types'][i] == 'categorical':
                    attr_len = len(node_attributes['values'][i])
                elif node_attributes['types'][i] == 'continuous':
                    attr_len = 1
                else:
                    attr_len = 0
                node_attr_slices[node_attributes['names'][i]] = (
                    start_attr_index, start_attr_index + attr_len)
                start_attr_index = start_attr_index + attr_len

        edge_attr_slices = {}
        if edge_attributes:
            start_attr_index = 0
            for i in range(len(edge_attributes['names'])):
                if edge_attributes['types'][i] == 'vector':
                    attr_len = edge_attributes['values'][i]
                elif edge_attributes['types'][i] == 'categorical':
                    attr_len = len(edge_attributes['values'][i])
                elif edge_attributes['types'][i] == 'continuous':
                    attr_len = 1
                else:
                    attr_len = 1
                edge_attr_slices[edge_attributes['names'][i]] = (
                    start_attr_index, start_attr_index + attr_len)
                start_attr_index = start_attr_index + attr_len

        return node_attr_slices, edge_attr_slices


class VisiblePart:
    def __init__(
            self,
            gen_dataset,
            center: [int, list, tuple] = None,
            depth: [int] = None
    ):
        """ Compute a part of dataset specified by a center node/graph and a depth

        :param gen_dataset:
        :param center: central node/graph or a list of nodes/graphs
        :param depth: neighborhood depth or number of graphs before and after center to take,
         e.g. center=7, depth=2 will give 5,6,7,8,9 graphs
        :return:
        """
        #         neigh   graph   graphs
        # nodes   [[n]]     n      [n]
        # graphs    -       -      [g]
        # edges   [[e]]   [[e]]   [[e]]
        self.graphs = None
        self.nodes = None
        self.edges = None

        self._ixes = None  # node or graph ids to include to the result

        if gen_dataset.is_multi():
            if center is not None:  # Get several graphs
                if isinstance(center, list):
                    self._ixes = center
                else:
                    if depth is None:
                        depth = 3
                    self._ixes = range(
                        max(0, center - depth),
                        min(gen_dataset.info.count, center + depth + 1))
            else:  # Get all graphs
                self._ixes = range(gen_dataset.info.count)

            self.graphs = list(self._ixes)
            self.nodes = [gen_dataset.info.nodes[ix] for ix in self._ixes]
            self.edges = [gen_dataset.dataset_data['edges'][ix] for ix in self._ixes]

        else:  # single
            assert gen_dataset.info.count == 1

            if center is not None:  # Get neighborhood
                if isinstance(center, list):
                    raise NotImplementedError
                if depth is None:
                    depth = 2

                if gen_dataset.info.hetero:
                    # TODO misha hetero
                    raise NotImplementedError

                    nodes = {0: {center[0]: {center[1]}}}  # {depth: {type: set of ids}}
                    edges = {0: {}}  # incoming edges, {depth: {type: list}}
                    prev_nodes = set()  # Nodes in neighborhood Up to depth=d-1

                    all_edges = gen_dataset.dataset_data['edges'][0]
                    for d in range(1, depth + 1):
                        ns = nodes[d - 1]
                        es_next = []
                        ns_next = set()
                        for i, j in all_edges:
                            # Get all incoming edges * -> j
                            if j in ns and i not in prev_nodes:
                                es_next.append((i, j))
                                if i not in ns:
                                    ns_next.add(i)

                            if not gen_dataset.info.directed:
                                # Check also outcoming edge i -> *, excluding already added
                                if i in ns and j not in prev_nodes:
                                    es_next.append((j, i))
                                    if j not in ns:
                                        ns_next.add(j)

                        prev_nodes.update(ns)
                        nodes[d] = ns_next
                        edges[d] = es_next

                    self.nodes = [list(ns) for ns in nodes.values()]
                    self.edges = [list(es) for es in edges.values()]
                    self._ixes = [n for ns in self.nodes for n in ns]

                else:  # homo
                    nodes = {0: {center}}  # {depth: set of ids}
                    edges = {0: []}  # incoming edges
                    prev_nodes = set()  # Nodes in neighborhood Up to depth=d-1

                    all_edges = gen_dataset.dataset_data['edges'][0]
                    for d in range(1, depth + 1):
                        ns = nodes[d - 1]
                        es_next = []
                        ns_next = set()
                        for i, j in all_edges:
                            # Get all incoming edges * -> j
                            if j in ns and i not in prev_nodes:
                                es_next.append((i, j))
                                if i not in ns:
                                    ns_next.add(i)

                            if not gen_dataset.info.directed:
                                # Check also outcoming edge i -> *, excluding already added
                                if i in ns and j not in prev_nodes:
                                    es_next.append((j, i))
                                    if j not in ns:
                                        ns_next.add(j)

                        prev_nodes.update(ns)
                        nodes[d] = ns_next
                        edges[d] = es_next

                    self.nodes = [list(ns) for ns in nodes.values()]
                    self.edges = [list(es) for es in edges.values()]
                    self._ixes = [n for ns in self.nodes for n in ns]

            else:  # Get whole graph
                self.edges = gen_dataset.dataset_data['edges']
                self.nodes = gen_dataset.info.nodes[0]
                if gen_dataset.info.hetero:
                    self._ixes = {nt: list(range(self.nodes[nt])) for nt in gen_dataset.node_types}
                else:
                    self._ixes = list(range(self.nodes))

    def ixes(
            self
    ) -> list:
        # if isinstance(self._ixes, list):
        #     for i in self._ixes:
        #         yield i
        # elif isinstance(self._ixes, dict):
        #     for nt, ixes in self._ixes.items():
        #         for i in ixes:
        #             yield nt, i
        return self._ixes

    def as_dict(
            self
    ) -> dict:
        res = {}
        if self.nodes:
            res['nodes'] = self.nodes
        if self.edges:
            res['edges'] = self.edges
        if self.graphs:
            res['graphs'] = self.graphs
        return res

    def filter(
            self,
            array
    ) -> dict:
        """ Suppose ixes = [2,4]: [a, b, c, d] ->  {2: b, 4: d}
        """
        return {ix: array[ix] for ix in self._ixes}


class GeneralDataset:
    """ Generalisation of Pytorch-geometric and user-defined datasets: custom, VK, etc.
    """

    def __init__(
            self,
            dataset_config: Union[DatasetConfig, ConfigPattern]
    ):
        """
        Args:
            dataset_config: DatasetConfig dict from frontend
        """
        # = Configs
        self.dataset_config = dataset_config
        self.dataset_var_config: DatasetVarConfig = None
        self.info: DatasetInfo = None

        # = Variable part
        self.visible_part: VisiblePart = None  # index of visible nodes/graphs at frontend

        self.dataset: Dataset = None  # Current PTG dataset
        self.dataset_data: dict = None  # structure data, proxy for dataset tensors, contains 'edges', 'node_attributes', 'edge_attributes'
        self.dataset_var_data: dict = None  # var data, proxy for dataset tensors, contains 'node_features', 'edge_features', 'labels'

        from base.dataset_stats import DatasetStats
        self.stats = DatasetStats(self)  # dict of {stat -> value}  # fixme misha - should keep it in prepared?

        # Data split
        self.percent_test_class = None  # FIXME misha do we need it here? it is in manager_config
        self.percent_train_class = None

        self.train_mask = None
        self.val_mask = None
        self.test_mask = None

        # Build graph structure
        self._compute_dataset_data()

    @property
    def name(
            self
    ) -> str:
        """ Last folder name. """
        return self.dataset_config.full_name[-1]

    @property
    def root_dir(
            self
    ) -> Path:
        """ Dataset root directory with folders 'raw' and 'prepared'. """
        # FIXME Misha, dataset_prepared_dir return path and files_paths not only path
        return Declare.dataset_root_dir(self.dataset_config)[0]

    @property
    def results_dir(
            self
    ) -> Path:
        """ Path to 'prepared/../' folder where tensor data is stored. """
        # FIXME Misha, dataset_prepared_dir return path and files_paths not only path
        return Path(Declare.dataset_prepared_dir(self.dataset_config, self.dataset_var_config)[0])

    @property
    def raw_dir(
            self
    ) -> Path:
        """ Path to 'raw/' folder where raw data is stored. """
        return self.root_dir / 'raw'

    @property
    def api_path(
            self
    ) -> Path:
        """ Path to '.api' file. Could be not present. """
        return self.root_dir / '.api'

    @property
    def info_path(
            self
    ) -> Path:
        """ Path to 'metainfo' file. """
        return self.root_dir / 'metainfo'

    @property
    def data(
            self
    ) -> Data:
        # fixme access
        return self.dataset._data

    @property
    def num_classes(
            self
    ) -> int:
        return self.dataset.num_classes

    @property
    def num_node_features(
            self
    ) -> int:
        return self.dataset.num_node_features

    @property
    def labels(
            self
    ) -> torch.Tensor:
        # fixme why do we need it? maybe other field make also
        if self.dataset_var_data['labels'] is None:
            # NOTE: this is a copy from torch_geometric.data.dataset v=2.3.1
            from torch_geometric.data.dataset import _get_flattened_data_list
            data_list = _get_flattened_data_list([data for data in self.dataset])
            self.dataset_var_data['labels'] = torch.cat([data.y for data in data_list if 'y' in data], dim=0)
        return self.dataset_var_data['labels']

    def __len__(
            self
    ) -> int:
        return self.info.count

    def domain(
            self
    ) -> str:
        return self.dataset_config.domain

    def is_multi(
            self
    ) -> bool:
        """ Return whether this dataset is multiple-graphs or single-graph. """
        return self.info.count > 1

    def is_hetero(
            self
    ) -> bool:
        """ Return whether this dataset is hetero graph. """
        return self.info.hetero

    def build(
            self,
            dataset_var_config: Union[ConfigPattern, DatasetVarConfig]
    ) -> None:
        """ Create node feature tensors from attributes based on dataset_var_config.
        """
        if dataset_var_config == self.dataset_var_config:
            return
        self.dataset_var_config = dataset_var_config
        self._compute_dataset_var_data()

    def get_dataset_data(
            self,
            part: Union[dict, None] = None
    ) -> dict:
        """ Get DatasetData for specified graphs or nodes
        """
        edges_list = []
        node_attributes = {}
        res = {
            'edges': edges_list,
            'node_attributes': node_attributes,
        }

        visible_part = self.visible_part if part is None else VisiblePart(self, **part)

        res.update(visible_part.as_dict())

        # Get needed part of self.dataset_data
        ixes = visible_part.ixes()
        if self.is_multi():
            for a, vals_list in self.dataset_data['node_attributes'].items():
                node_attributes[a] = {ix: vals_list[ix] for ix in ixes}

        else:
            if isinstance(visible_part.nodes, list):  # neighborhood
                for a, vals_list in self.dataset_data['node_attributes'].items():
                    node_attributes[a] = [{
                        n: (vals_list[0][n] if n in vals_list[0] else None) for n in ixes}]

            else:  # whole graph
                res['node_attributes'] = self.dataset_data['node_attributes']

        return res

    def _compute_dataset_data(
            self
    ) -> None:
        """ Build graph(s) structure - edge index
        Structure according to https://docs.google.com/spreadsheets/d/1fNI3sneeGoOFyIZP_spEjjD-7JX2jNl_P8CQrA4HZiI/edit#gid=1096434224
        """
        raise RuntimeError("This should be implemented in subclass")

    def set_visible_part(
            self,
            part: dict
    ) -> None:
        if self.dataset_data is None:
            self._compute_dataset_data()

        self.visible_part = VisiblePart(self, **part)

    def get_dataset_var_data(
            self,
            part: Union[dict, None] = None
    ) -> dict:
        """ Get DatasetVarData for specified graphs or nodes
        """
        if self.dataset_var_data is None:
            self._compute_dataset_var_data()

        # Get needed part of self.dataset_var_data
        features = {}
        labels = {}
        dataset_var_data = {
            "features": features,
            "labels": labels,
        }

        visible_part = self.visible_part if part is None else VisiblePart(self, **part)

        for ix in visible_part.ixes():
            # TODO IMP misha replace with getting data from tensors instead of keeping the whole data
            features[ix] = self.dataset_var_data['node_features'][ix]
            labels[ix] = self.dataset_var_data['labels'][ix]

        return dataset_var_data

    def _compute_dataset_var_data(
            self
    ) -> None:
        """ Build graph(s) tensors - features and labels
        """
        raise RuntimeError("This should be implemented in subclass")

        # FIXME version fail in torch-geom 2.3.1
        # self.dataset.num_classes = int(self.dataset_data["info"]["labelings"][self.dataset_var_config.labeling])

        labels = []
        node_features = []
        for ix in range(len(self.dataset)):
            data = self.dataset.get(ix)
            labels.append(data.y.tolist())
            node_features.append(data.x.tolist())

        if self.is_multi():
            self.dataset_var_data = {
                "features": node_features,
                "labels": labels,
            }
            # self.dataset_var_data = {
            #     i: {
            #         "features": node_features[i],
            #         "labels": labels[i],
            #     } for i in range(len(self.dataset))
            # }
        else:
            self.dataset_var_data = {
                "features": node_features if self.is_multi() else node_features[0],
                "labels": labels if self.is_multi() else labels[0],
            }

    def get_stat(
            self,
            stat: str
    ) -> Union[int, float, dict, str]:
        """ Get statistics.
        """
        return self.stats.get(stat)

    def _compute_stat(
            self,
            stat: str
    ) -> None:
        """ Compute a non-standard statistics.
        """
        # Should be defined in a subclass
        raise NotImplementedError()

    def is_one_hot_able(
            self
    ) -> bool:
        """ Return whether features are 1-hot encodings. If yes, nodes can be colored.
        """
        assert self.dataset_var_config

        if not self.is_multi():
            return True

        res = False
        features = self.dataset_var_config.features
        if len(features.keys()) == 1:
            # 1-hot over nodes and no attributes is OK
            if 'str_g' in features:
                if features['str_g'] == 'one_hot':
                    res = True

            # Only 1 categorical attr is OK
            elif 'attr' in features:
                if len(features['attr']) == 1:
                    attr = list(features['attr'].keys())[0]
                    if features['attr'][attr] == 'categorical':
                        res = True
                    elif features['attr'][attr] == 'vector':
                        # Check honestly each feature vector
                        feats = self.dataset_var_data['node_features']
                        res = all(all(all(x == 1 or x == 0 for x in f) for f in feat) for feat in feats) and \
                              all(all(sum(f) == 1 for f in feat) for feat in feats)

        return res

    def train_test_split(
            self,
            percent_train_class: float = 0.8,
            percent_test_class: float = 0.2
    ) -> None:
        """ Compute train-validation-test split of graphs/nodes. """
        self.percent_train_class = percent_train_class
        self.percent_test_class = percent_test_class
        percent_val_class = 1 - percent_train_class - percent_test_class  # - 1.1e-15

        if percent_val_class < -1.1e-15:
            raise Exception("percent_train_class + percent_test_class > 1")
        train_mask = torch.BoolTensor([False] * self.labels.size(dim=0))
        val_mask = torch.BoolTensor([False] * self.labels.size(dim=0))
        test_mask = torch.BoolTensor([False] * self.labels.size(dim=0))

        labeled_nodes_numbers = [n for n, y in enumerate(self.labels) if y != -1]
        num_train = int(percent_train_class * len(labeled_nodes_numbers))
        num_test = int(percent_test_class * len(labeled_nodes_numbers))
        num_eval = len(labeled_nodes_numbers) - num_train - num_test
        if percent_val_class <= 0 and num_eval > 0:
            num_test += num_eval
            num_eval = 0
        split = randperm(num_train + num_eval + num_test, generator=default_generator).tolist()

        for elem in split[:num_train]:
            train_mask[labeled_nodes_numbers[elem]] = True
        for elem in split[num_train: num_train + num_eval]:
            val_mask[labeled_nodes_numbers[elem]] = True
        for elem in split[num_train + num_eval:]:
            test_mask[labeled_nodes_numbers[elem]] = True

        # labeled_nodes_numbers = [n for n, y in enumerate(self.dataset_var_data["labels"]) if y != -1]
        # for i in labeled_nodes_numbers:
        #     test_mask[i] = True
        # t = int(percent_train_class * len(labeled_nodes_numbers))
        # for i in np.random.choice(labeled_nodes_numbers, t, replace=False):
        #     train_mask[i] = True
        #     test_mask[i] = False
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.val_mask = val_mask
        self.dataset.data.train_mask = train_mask
        self.dataset.data.test_mask = test_mask
        self.dataset.data.val_mask = val_mask

    def save_train_test_mask(
            self,
            path: Union[str, Path]
    ) -> None:
        """ Save current train/test mask to a given path (together with the model). """
        if path is not None:
            path /= 'train_test_split'
            path.parent.mkdir(exist_ok=True, parents=True)
            torch.save([self.train_mask, self.val_mask, self.test_mask,
                        (self.percent_train_class, self.percent_test_class)], path)
        else:
            assert Exception("Path for save file is None")


class LocalDataset(
    InMemoryDataset
):
    """ Locally saved PTG Dataset.
    """

    def __init__(
            self,
            data_list: List[Data],
            results_dir: Union[str, Path],
            process_func: Union[Callable, None] = None,
    ):
        """
        :param data_list: optionally, list of ready torch_geometric.data.Data objects
        :param results_dir: directory where tensors are stored
        :param process_func: optionally, custom process() function, which converts raw files into tensors
        """
        self.data_list = data_list
        self.results_dir = results_dir
        if process_func:
            self.process = process_func
        # Init and process
        super().__init__(None)

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
        torch.save(self.collate(self.data_list), self.processed_paths[0])

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


if __name__ == '__main__':
    print("test dataset")
    from base.ptg_datasets import is_in_torch_geometric_datasets, LocalPTGDataset, aPTGDataset

    # dc = DatasetConfig(('single', 'test1'), init_kwargs={"a": 10, "b": 'line'})
    # ptg = aPTGDataset(dc)

    # LocalPTGDataset
    x = tensor([[0, 0], [1, 0], [1, 0]])
    edge_index = tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    y = tensor([0, 1, 1])

    # Single
    data_list = [Data(x=x, edge_index=edge_index, y=y)]
    dataset = LocalPTGDataset(data_list)

    dvc = aPTGDataset.default_dataset_var_config

    # # LibPTGDataset
    # dataset = LibPTGDataset(domain='single-graph', group='Planetoid', name='Cora')
    # dvc = LibPTGDataset.default_dataset_var_config

    # # KnownFormatDataset
    # dc = DatasetConfig(('single-graph', 'example'), init_kwargs={"a": 10, "b": 'line'})
    # from base.custom_datasets import KnownFormatDataset
    # dataset = KnownFormatDataset(dc)
    #
    # dvc = DatasetVarConfig(features={'attr': {'a': 'as_is'}}, labeling='binary', dataset_ver_ind=0)

    print(dataset.get_dataset_data({}))

    dataset.build(dvc)
    print(dataset.get_dataset_var_data({}))

    # gen_dataset_s = DatasetManager.register_torch_geometric_local(dataset)