import json
import os
import shutil
from pathlib import Path
from typing import Union, List, Dict

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from data_structures.configs import DatasetConfig, ConfigPattern, FeatureConfig
from datasets.dataset_info import DatasetInfo
from datasets.gen_dataset import LocalDataset, GeneralDataset


class KnownFormatDataset(
    GeneralDataset
):
    """
    Custom dataset in 'ij' format or one of popular formats (see
    :class:`~datasets.dataset_converter.DatasetConverter.supported_formats`).
    User defines functions for building graph from raw files and feature creation.

    Example of usage

    .. code-block:: python

        from datasets.datasets_manager import DatasetManager

        # Define dataset config - where to get raw data
        dc = DatasetConfig(('example', 'single-graph', 'example'))
        dataset = DatasetManager.get_by_config(dc)

        # Get graph data from a 1-neighborhood of node 0
        dataset.set_visible_part({'center': 0, 'depth': 1})
        print(dataset.visible_part.get_dataset_data())
        >>> DatasetData[
        >>>  edges: [[], [(1, 0)]]
        >>>  nodes: [[0], [1]]
        >>>  graphs: None
        >>>  node_attributes: {'a': [{0: 1, 1: 1}], 'b': [{0: 'A', 1: 'A'}]}
        >>> ]

        # Define var config and build tensors
        dvc = DatasetVarConfig(features=FeatureConfig(node_attr=['a', 'b']),
                               labeling='binary', dataset_ver_ind=0)
        dataset.build(dvc)

        # Get var data from the neighborhood
        print(dataset.visible_part.get_dataset_var_data())
        >>> DatasetVarData[
        >>>  labels: {0: 1, 1: 1}
        >>>  node_features: {0: [1.0, 1.0, 0.0, 0.0], 1: [1.0, 1.0, 0.0, 0.0]}
        >>> ]
    """

    def __init__(
            self,
            dataset_config: Union[ConfigPattern, DatasetConfig],
            default_node_attr_value: dict = None,
            default_edge_attr_value: dict = None,
    ):
        """
        Args:
            dataset_config (DatasetConfig | ConfigPattern): config to define dataset
            default_node_attr_value (dict): dict as {attr -> value} with default node attributes values to
             apply where missing.
            default_edge_attr_value (dict): dict as {attr -> value} with default edge attributes values to
             apply where missing.
        """
        self._default_node_attr_value = default_node_attr_value
        self._default_edge_attr_value = default_edge_attr_value
        self.node_map = None  # Optional nodes mapping: node_map[i] = original id of node i

        self._ptg_edge_index: list = None  # ptg tensors
        self._node_attributes: dict = None  # python lists
        # self._edge_attributes: dict = None

        super(KnownFormatDataset, self).__init__(dataset_config)

    @property
    def raw_dir(
            self
    ) -> Path:
        """ Path to folder where raw ij data is stored. """
        if self._format == 'ij':
            return self.root_dir / 'raw'
        else:
            return self.root_dir / 'converted'

    @property
    def node_attributes_dir(
            self
    ) -> Path:
        """ Path to dir with node attributes. """
        return self.raw_dir / 'node_attributes'

    @property
    def edge_attributes_dir(
            self
    ) -> Path:
        """ Path to dir with edge attributes. """
        return self.raw_dir / 'edge_attributes'

    @property
    def labels_dir(
            self
    ) -> Path:
        """ Path to dir with labels. """
        return self.raw_dir / 'labels'

    @property
    def edges_path(
            self
    ) -> Path:
        """ Path to file with edge list. """
        return self.raw_dir / 'edges.ij'

    @property
    def edge_index_path(
            self
    ) -> Path:
        """ Path to file with edge indices, for multiple graphs. """
        return self.raw_dir / 'edge_index'

    @property
    def edges(
            self
    ) -> List[torch.Tensor]:
        return self._ptg_edge_index

    def node_attributes(
            self,
            attrs: List[str] = None
    ) -> Dict[str, Union[list, torch.Tensor]]:
        """ Get node attributes as a dict {name -> list}"""
        if attrs is None:
            attrs = sorted(self._node_attributes.keys())
        return {a: self._node_attributes[a] for a in attrs}

    def edge_attributes(
            self,
            attrs: List[str] = None
    ) -> Dict[str, Union[list, torch.Tensor]]:
        """ Get edge attributes as a dict {name -> list}"""
        raise NotImplementedError()

    def check_validity(
            self
    ) -> None:
        """ Check that dataset files (graph and attributes) are valid and consistent with .info.
        """
        # Assuming info is OK
        count = self.info.count
        # Check edges
        if self.is_multi():
            with open(self.edges_path, 'r') as f:
                num_edges = sum(1 for _ in f)
            with open(self.edge_index_path, 'r') as f:
                edge_index = json.load(f)
                assert all(i <= num_edges for i in edge_index)
                assert num_edges == edge_index[-1]
                assert count == len(edge_index)

        # Check nodes
        all_nodes = [set() for _ in range(count)]  # sets of nodes
        if self.is_multi():
            with open(self.edges_path, 'r') as f:
                start = 0
                for ix, end in enumerate(edge_index):
                    for _ in range(end - start):
                        all_nodes[ix].update(map(int, f.readline().split()))
                    if self.info.remap:
                        assert len(all_nodes[ix]) == self.info.nodes[ix]
                    else:
                        assert all_nodes[ix] == set(range(self.info.nodes[ix]))
                    start = end
        else:
            with open(self.edges_path, 'r') as f:
                for line in f.readlines():
                    all_nodes[0].update(map(int, line.split()))
                if self.info.remap:
                    assert len(all_nodes[0]) == self.info.nodes[0]
                else:
                    assert all_nodes[0] == set(range(self.info.nodes[0]))

        # Check node attributes
        for ix, attr in enumerate(self.info.node_attributes["names"]):
            with open(self.node_attributes_dir / attr, 'r') as f:
                node_attributes = json.load(f)
            if not self.is_multi():
                node_attributes = [node_attributes]
            for i, attributes in enumerate(node_attributes):
                assert all_nodes[i] == set(map(int, attributes.keys()))
                if self.info.node_attributes["types"][ix] == "continuous":
                    v_min, v_max = self.info.node_attributes["values"][ix]
                    assert all(isinstance(v, (int, float, complex)) for v in attributes.values())
                    assert min(attributes.values()) >= v_min
                    assert max(attributes.values()) <= v_max
                elif self.info.node_attributes["types"][ix] == "categorical":
                    real = set(attributes.values())
                    gold = set(self.info.node_attributes["values"][ix])
                    assert real.issubset(gold), \
                        f"Real node attributes ({real}) differ from ones specified in metainfo ({gold})"

        # Check edge attributes
        for ix, attr in enumerate(self.info.edge_attributes["names"]):
            with open(self.edge_attributes_dir / attr, 'r') as f:
                edge_attributes = json.load(f)
            if not self.is_multi():
                edge_attributes = [edge_attributes]
            for i, attributes in enumerate(edge_attributes):
                # TODO check edges
                if self.info.edge_attributes["types"][ix] == "continuous":
                    v_min, v_max = self.info.edge_attributes["values"][ix]
                    assert all(isinstance(v, (int, float, complex)) for v in attributes.values())
                    assert min(attributes.values()) >= v_min
                    assert max(attributes.values()) <= v_max
                elif self.info.edge_attributes["types"][ix] == "categorical":
                    assert set(attributes.values()).issubset(
                        set(self.info.edge_attributes["values"][ix]))

        # Check labels
        for labelling, _ in self.info.labelings.items():
            with open(self.labels_dir / labelling, 'r') as f:
                labels = json.load(f)
            if self.is_multi():  # graph labels
                assert set(range(count)) == set(map(int, labels.keys()))
            else:  # nodes labels
                assert all_nodes[0] == set(map(int, labels.keys()))

    def _convert_to_ij(
            self
    ) -> None:
        # Check if ij files exist
        if self.edges_path.exists():
            return

        # Convert the data if necessary, write it to 'converted/' directory
        self.raw_dir.mkdir(exist_ok=False)
        from datasets.dataset_converter import DatasetConverter
        DatasetConverter.format_to_ij(
            self.info, self.root_dir / 'raw', self.raw_dir,
            self._default_node_attr_value, self._default_edge_attr_value)

        try:
            self.check_validity()
        except AssertionError as e:
            # Dataset is not valid - remove all converted files
            shutil.rmtree(self.raw_dir)
            raise e

    def _compute_dataset_data(
            self
    ) -> None:
        """ Read edges and attributes from raw files.
        """
        self.info = DatasetInfo.read(self.metainfo_path)
        self._format = self.info.format or 'ij'
        if self._format != 'ij':
            self._convert_to_ij()

        if not self.edges_path.exists():
            raise FileNotFoundError(
                f"File with edges not found at {self.edges_path}. Check that all files exist and"
                f" correct graph format is specified in metainfo.")

        self._ptg_edge_index = []

        # Read edges and attributes
        if self.is_multi():
            self._read_multi()
        else:
            self._read_single()
        self._read_attributes()

        self._infer_feature_slices_form_attributes()

    def _infer_feature_slices_form_attributes(
            self,
    ) -> None:
        """ Set correspondence of attributes to components of feature vector.
        """
        node_attributes = self.info.node_attributes

        if len(node_attributes) > 0 and\
                isinstance(next(iter(node_attributes.values())), dict):
            # TODO misha hetero
            return

        self.node_attr_slices = {}
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
                self.node_attr_slices[node_attributes['names'][i]] = (
                    start_attr_index, start_attr_index + attr_len)
                start_attr_index = start_attr_index + attr_len

        # edge_attributes = self.info.edge_attributes
        # self.edge_attr_slices = {}
        # if edge_attributes:
        #     start_attr_index = 0
        #     for i in range(len(edge_attributes['names'])):
        #         if edge_attributes['types'][i] == 'vector':
        #             attr_len = edge_attributes['values'][i]
        #         elif edge_attributes['types'][i] == 'categorical':
        #             attr_len = len(edge_attributes['values'][i])
        #         elif edge_attributes['types'][i] == 'continuous':
        #             attr_len = 1
        #         else:
        #             attr_len = 1
        #         self.edge_attr_slices[edge_attributes['names'][i]] = (
        #             start_attr_index, start_attr_index + attr_len)
        #         start_attr_index = start_attr_index + attr_len

    def _read_single(
            self
    ) -> None:
        """ Read edges, remap node, create edge index - for single graph case.
        """
        node_map = {}
        ptg_edge_index = [[], []]
        node_index = 0
        with open(self.edges_path, 'r') as f:
            for line in f.readlines():
                node_index = self._read_edge(line, node_index, node_map, ptg_edge_index)

        self._ptg_edge_index = [torch.tensor(np.asarray(ptg_edge_index))]
        if self.info.remap:
            if len(node_map) < self.info.nodes[0]:
                labeling_path = self.labels_dir / os.listdir(self.labels_dir)[0]
                with open(labeling_path, 'r') as f:
                    labeling_dict = json.load(f)
                for node in labeling_dict.keys():
                    node = int(node)
                    if node not in node_map:
                        node_map[node] = node_index
                        node_index += 1
            # assert node_index == self.info.nodes[0]
            # Original ids in the order of appearance
            self.node_map = list(node_map.keys())
            # self.info.node_info = {"id": self.node_map}

        assert node_index == self.info.nodes[0]
        assert len(self._ptg_edge_index) == self.info.count

    def _read_multi(
            self
    ) -> None:
        """ Read edges, remap node, create edge index - for multi graph case.
        """
        count = self.info.count
        node_maps = []  # list of node_maps

        # Read edges
        with open(self.edge_index_path, 'r') as f:
            edge_index = json.load(f)

        with open(self.edges_path, 'r') as f:
            g_ix = 0
            node_index = 0
            ptg_edge_index = [[], []]  # Over each graph
            node_map = {}
            node_maps.append(node_map)
            for l, line in enumerate(f.readlines()):
                node_index = self._read_edge(line, node_index, node_map, ptg_edge_index)

                if l == edge_index[g_ix] - 1:
                    if self.info.remap:
                        if len(node_maps[g_ix]) < self.info.nodes[g_ix]:
                            # Get the full nodes list from 1st labeling
                            labeling_path = self.labels_dir / os.listdir(self.labels_dir)[0]
                            with open(labeling_path, 'r') as f:
                                labeling_dict = json.load(f)
                            for node in labeling_dict.keys():
                                node = int(node)
                                if node not in node_maps[g_ix]:
                                    node_maps[g_ix][node] = node_index
                                    node_index += 1
                    self._ptg_edge_index.append(torch.tensor(np.asarray(ptg_edge_index)))
                    g_ix += 1
                    if g_ix == count:
                        break
                    node_index = 0
                    ptg_edge_index = [[], []]
                    node_map = {}
                    node_maps.append(node_map)

        if self.info.remap:
            # Original ids in the order of appearance
            self.node_map = []
            for node_map in node_maps:
                self.node_map.append(list(node_map.keys()))
            # self.info.node_info = {"id": self.node_map}

        assert sum(len(_) for _ in node_maps) == sum(self.info.nodes)
        assert len(self._ptg_edge_index) == self.info.count

    def _read_edge(
            self,
            line: str,
            node_index: int,
            node_map: dict,
            ptg_edge_index: list
    ) -> int:
        i, j = map(int, line.split())
        if i not in node_map:
            node_map[i] = node_index
            node_index += 1
        if j not in node_map:
            node_map[j] = node_index
            node_index += 1
        if self.info.remap:
            i = node_map[i]
            j = node_map[j]
        ptg_edge_index[0].append(i)
        ptg_edge_index[1].append(j)
        if not self.info.directed:
            ptg_edge_index[0].append(j)
            ptg_edge_index[1].append(i)
        return node_index

    def _read_attributes(
            self
    ) -> None:
        """ Read node attributes and remap them.
        """
        self._node_attributes = {}  # {attr -> [{node -> value} for each graph]}
        for a in self.info.node_attributes["names"]:
            self._node_attributes[a] = []
            with open(self.node_attributes_dir / a, 'r') as f:
                self._node_attributes[a] = []
                orig_node_attributes = json.load(f)
                if not self.is_multi():
                    orig_node_attributes = [orig_node_attributes]
                for g in range(self.info.count):
                    node_attributes = {}
                    for ix, orig in self._iter_nodes(g):
                        node_attributes[ix] = orig_node_attributes[g][orig]
                    self._node_attributes[a].append(node_attributes)

    def _compute_dataset_var_data(
            self
    ) -> None:
        """ Build PTG Dataset based on dataset_var_config.
        """
        self.dataset = LocalDataset(None, self.prepared_dir, process_func=self._create_ptg)

    def _create_ptg(
            self
    ) -> None:
        """ Create PTG Dataset and save tensors
        """
        data_list = []
        for ix in range(self.info.count):
            node_features = self._feature_tensor(ix)
            labels = self._labeling_tensor(ix)
            x = torch.tensor(node_features, dtype=torch.float)
            y = torch.tensor(labels)
            data = Data(
                x=x, edge_index=self._ptg_edge_index[ix], y=y,
                num_classes=self.info.labelings[self.dataset_var_config.labeling]
            )
            data_list.append(data)

        # Build slices and save
        self.prepared_dir.mkdir(exist_ok=True, parents=True)
        torch.save(InMemoryDataset.collate(data_list), self.prepared_dir / 'data.pt')

    def _iter_nodes(
            self,
            graph: int = None
    ) -> [int, str]:
        """ Iterate over nodes according to mapping. Yields pairs of (node_index, original_id)
        """
        if self.node_map is not None:
            node_map = self.node_map[graph] if self.is_multi() else self.node_map
            for ix, orig in enumerate(node_map):
                yield ix, str(orig)
        else:
            for n in range(self.info.nodes[graph or 0]):
                yield n, str(n)

    def _labeling_tensor(
            self,
            g_ix: int = None
    ) -> list:
        """ Returns list of labels (not tensors) """
        y = []
        # Read labels
        labeling_path = self.labels_dir / self.dataset_var_config.labeling
        with open(labeling_path, 'r') as f:
            labeling_dict = json.load(f)

        if self.is_multi():
            if labeling_dict[str(g_ix)] is not None:
                y.append(labeling_dict[str(g_ix)])
            else:
                y.append(-1)
        else:
            for _, orig in self._iter_nodes():
                if labeling_dict[orig] is not None:
                    y.append(labeling_dict[orig])
                else:
                    y.append(-1)

        return y

    def _feature_tensor(
            self,
            g_ix: int = None
    ) -> list:
        """ Returns list of features (not tensors) for graph g_ix.
        """
        feature_config: FeatureConfig = self.dataset_var_config.features
        node_struct = feature_config.node_struct or []
        node_attr = feature_config.node_attr or []
        edge_attr = feature_config.edge_attr or []
        graph_attr = feature_config.graph_attr or []

        if len(feature_config) == 0:
            raise RuntimeError(f"{FeatureConfig.__name__} must not be empty")

        for fc in [node_struct, node_attr, edge_attr, graph_attr]:
            if not isinstance(fc, list):
                raise TypeError(f"{self.__class__.__name__} expects lists as feature configs,"
                                f" but {type(fc)} is given")

        if self.is_multi():
            nodes = self.info.nodes[g_ix]
        else:  # single
            nodes = self.info.nodes[0]

        node_features = [[] for _ in range(nodes)]  # List of vectors

        # Transform structure to node features
        for elem in node_struct:
            if elem == FeatureConfig.one_hot:
                # 1-hot encoding of all nodes
                for n in range(nodes):
                    vec = [0] * nodes
                    vec[n] = 1
                    node_features[n].extend(vec)

            elif elem == FeatureConfig.ten_ones:
                # add 10 ones to all nodes
                for n in range(nodes):
                    node_features[n].extend([1] * 10)

            elif elem == FeatureConfig.degree:
                raise NotImplementedError

            elif elem == FeatureConfig.clustering:
                raise NotImplementedError

            else:
                raise RuntimeError(f"Unknown feature config for node_struct: '{elem}'")

        # Transform attributes for nodes, edges, graph
        def one_hot(
                x: int,
                values: list
        ) -> list:
            res = [0] * len(values)
            for ix, v in enumerate(values):
                if x == v:
                    res[ix] = 1
                    break
            return res

        def assign_feats(attrs, func):
            # for n, orig in self._iter_nodes(g_ix):
            for n in range(len(attrs)):
                value = attrs[n]
                node_features[n].extend(func(value))

        for attr in node_attr:
            node_attributes_info = self.info.node_attributes
            ix = node_attributes_info["names"].index(attr)
            _type = node_attributes_info["types"][ix]

            if _type == "categorical":
                def func(x): return one_hot(x, node_attributes_info["values"][ix])
            elif _type == "continuous" or _type == "vector":
                def func(x): return x if isinstance(x, list) else [x]
            else:
                raise RuntimeError(f"{self.__class__.__name__} cannot convert attribute of type"
                                   f" '{_type}' to feature.")

            node_attributes = self.node_attributes([attr])[attr]
            if self.is_multi():
                node_attributes = node_attributes[g_ix]
            else:
                node_attributes = node_attributes[0]
            assign_feats(node_attributes, func)

        # TODO in future, same for edge and graph attributes

        if len(node_features[0]) == 0:
            raise RuntimeError("Feature vector size must be > 0")
        return node_features
