import json
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from aux.declaration import Declare
from base.datasets_processing import GeneralDataset, DatasetInfo
from aux.configs import DatasetConfig, DatasetVarConfig, ConfigPattern
from base.ptg_datasets import LocalDataset


class CustomDataset(
    GeneralDataset
):
    """ User-defined dataset in 'ij' format.
    """
    def __init__(
            self,
            dataset_config: Union[ConfigPattern, DatasetConfig]
    ):
        """
        Args:
            dataset_config: DatasetConfig dict from frontend
        """
        super().__init__(dataset_config)

        # assert self.labels_dir.exists()
        self.info = DatasetInfo.read(self.info_path)
        self.node_map = None  # Optional nodes mapping: node_map[i] = original id of node i
        self.edge_index = None

    @property
    def node_attributes_dir(
            self
    ) -> Path:
        """ Path to dir with node attributes. """
        return self.root_dir / 'raw' / (self.name + '.node_attributes')

    @property
    def edge_attributes_dir(
            self
    ) -> Path:
        """ Path to dir with edge attributes. """
        return self.root_dir / 'raw' / (self.name + '.edge_attributes')

    @property
    def labels_dir(
            self
    ) -> Path:
        """ Path to dir with labels. """
        return self.root_dir / 'raw' / (self.name + '.labels')

    @property
    def edges_path(
            self
    ) -> Path:
        """ Path to file with edge list. """
        return self.root_dir / 'raw' / (self.name + '.ij')

    @property
    def edge_index_path(
            self
    ) -> Path:
        """ Path to dir with labels. """
        return self.root_dir / 'raw' / (self.name + '.edge_index')

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
                    for _ in range(end-start):
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
                    assert set(attributes.values()).issubset(set(self.info.node_attributes["values"][ix]))

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
                    assert set(attributes.values()).issubset(set(self.info.edge_attributes["values"][ix]))

        # Check labels
        for labelling, n_classes in self.info.labelings.items():
            with open(self.labels_dir / labelling, 'r') as f:
                labels = json.load(f)
            if self.is_multi():  # graph labels
                assert set(range(count)) == set(map(int, labels.keys()))
            else:  # nodes labels
                assert all_nodes[0] == set(map(int, labels.keys()))

    def build(
            self,
            dataset_var_config: Union[ConfigPattern, DatasetVarConfig]
    ) -> None:
        """ Build ptg dataset based on dataset_var_config and create DatasetVarData.
        """
        if dataset_var_config == self.dataset_var_config:
            # PTG is cached
            return

        self.dataset_var_data = None
        self.stats.update_var_config()
        self.dataset_var_config = dataset_var_config
        self.dataset = LocalDataset(self.results_dir, process_func=self._create_ptg)

    def _compute_stat(
            self,
            stat: str
    ) -> dict:
        """ Compute some additional stats
        """
        if stat == "attr_corr":
            if self.node_attributes_dir.exists():
                # Read all continuous attrs
                node_attributes = self.info.node_attributes
                attr_node_attrs = {}  # {attr -> {node -> attr value}}
                for ix, a in enumerate(node_attributes["names"]):
                    if node_attributes["types"][ix] != "continuous": continue
                    with open(self.node_attributes_dir / a, 'r') as f:
                        attr_node_attrs[a] = json.load(f)

                # FIXME misha - for single graph [0]
                edges = self.edge_index
                node_map = (lambda i: str(self.node_map[i])) if self.node_map else lambda i: str(i)

                # Compute mean and std over edges
                in_attr_mean = {}
                in_attr_denom = {}
                out_attr_mean = {}
                out_attr_denom = {}
                for a, node_attrs in attr_node_attrs.items():
                    ins = []
                    outs = []
                    for i, j in zip(*edges):
                        i = int(i)
                        j = int(j)
                        outs.append(node_attrs[node_map(i)])
                        ins.append(node_attrs[node_map(j)])
                    in_attr_mean[a] = np.mean(ins)
                    in_attr_denom[a] = (np.sum(np.array(ins)**2) - len(edges)*in_attr_mean[a]**2)**0.5
                    out_attr_mean[a] = np.mean(outs)
                    out_attr_denom[a] = (np.sum(np.array(outs)**2) - len(edges)*out_attr_mean[a]**2)**0.5

                # Compute corr
                attrs = list(attr_node_attrs.keys())
                # Matrix of corr numerators
                pearson_corr = np.zeros((len(attrs), len(attrs)), dtype=float)
                for i, out_a in enumerate(attrs):
                    out_node_attrs = attr_node_attrs[out_a]
                    for j, in_a in enumerate(attrs):
                        in_node_attrs = attr_node_attrs[in_a]
                        corr = 0
                        for x, y in zip(*edges):
                            x = int(x)
                            y = int(y)
                            corr += (out_node_attrs[node_map(x)] - out_attr_mean[out_a]) * (
                                    in_node_attrs[node_map(y)] - in_attr_mean[in_a])
                        pearson_corr[i][j] = corr

                # Normalize on stds
                for i, out_a in enumerate(attrs):
                    for j, in_a in enumerate(attrs):
                        denom = out_attr_denom[out_a] * in_attr_denom[in_a]
                        pc = pearson_corr[i][j] / denom if denom != 0 else 1
                        pearson_corr[i][j] = min(1, max(-1, pc))

                return {'attributes': attrs, 'correlations': pearson_corr.tolist()}

        raise NotImplementedError()

    def _compute_dataset_data(
            self
    ) -> None:
        """ Get DatasetData for debug graph
        Structure according to https://docs.google.com/spreadsheets/d/1fNI3sneeGoOFyIZP_spEjjD-7JX2jNl_P8CQrA4HZiI/edit#gid=1096434224
        """
        # TODO misha - can we use ptg dataset? Problem is that it is not built at this stage.
        # super()._compute_dataset_data()

        self.dataset_data = {
            "edges": [],
        }

        # Read edges and attributes
        if self.is_multi():
            # FIXME misha format
            count = self.info.count
            node_maps = []  # list of node_maps

            # Read edges
            with open(self.edge_index_path, 'r') as f:
                edge_index = json.load(f)

            with open(self.edges_path, 'r') as f:
                g_ix = 0
                node_index = 0
                self.edge_index = []
                edges = []
                ptg_edge_index = [[], []]  # Over each graph
                node_map = {}
                node_maps.append(node_map)
                for l, line in enumerate(f.readlines()):
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
                    # TODO misha can we reuse one of them?
                    edges.append([i, j])
                    ptg_edge_index[0].append(i)
                    ptg_edge_index[1].append(j)
                    if not self.info.directed:
                        ptg_edge_index[0].append(j)
                        ptg_edge_index[1].append(i)

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
                        self.dataset_data['edges'].append(edges)
                        self.edge_index.append(torch.tensor(np.asarray(ptg_edge_index)))
                        g_ix += 1
                        if g_ix == count:
                            break
                        node_index = 0
                        edges = []
                        ptg_edge_index = [[], []]
                        node_map = {}
                        node_maps.append(node_map)

            if self.info.remap:
                # Original ids in the order of appearance
                self.node_map = []
                for node_map in node_maps:
                    self.node_map.append(list(node_map.keys()))
                self.info.node_info = {"id": self.node_map}

            assert sum(len(_) for _ in node_maps) == sum(self.info.nodes)
            assert len(self.dataset_data['edges']) == self.info.count

            # Read attributes
            self.dataset_data["node_attributes"] = {}
            for a in self.info.node_attributes["names"]:
                with open(self.node_attributes_dir / a, 'r') as f:
                    self.dataset_data["node_attributes"][a] = json.load(f)

        else:
            node_map = {}
            edges = []
            ptg_edge_index = [[], []]
            node_index = 0
            with open(self.edges_path, 'r') as f:
                for line in f.readlines():
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
                    # TODO misha can we reuse one of them?
                    edges.append([i, j])
                    ptg_edge_index[0].append(i)
                    ptg_edge_index[1].append(j)
                    if not self.info.directed:
                        ptg_edge_index[0].append(j)
                        ptg_edge_index[1].append(i)

                self.dataset_data['edges'].append(edges)
                self.edge_index = [torch.tensor(np.asarray(ptg_edge_index))]
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
                    self.info.node_info = {"id": self.node_map}

            assert node_index == self.info.nodes[0]
            assert len(self.dataset_data['edges']) == self.info.count

            # Read attributes
            self.dataset_data["node_attributes"] = {}
            if self.node_attributes_dir.exists():
                for a in os.listdir(self.node_attributes_dir):
                    with open(self.node_attributes_dir / a, 'r') as f:
                        self.dataset_data["node_attributes"][a] = [{
                            node_map[int(n)]: v for n, v in json.load(f).items()
                            if int(n) in node_map}]

        # Check for obligate parameters
        assert len(self.dataset_data["edges"]) > 0
        # assert len(info["labelings"]) > 0  # for VK we generate based on files

        # self.dataset_data['info'] = self.info.to_dict()
        # if self.info.name == "":
        #     self.dataset_data['info']['name'] = '/'.join(self.dataset_config.full_name())

    def _create_ptg(
            self
    ) -> None:
        """ Create PTG Dataset and save tensors
        """
        if self.edge_index is None:
            # TODO Misha think if it's good
            self._compute_dataset_data()

        data_list = []
        for ix in range(self.info.count):
            node_features = self._feature_tensor(ix)
            labels = self._labeling_tensor(ix)
            x = torch.tensor(node_features, dtype=torch.float)
            y = torch.tensor(labels)
            data = Data(
                x=x, edge_index=self.edge_index[ix], y=y,
                num_classes=self.info.labelings[self.dataset_var_config.labeling]
            )
            data_list.append(data)

        # Build slices and save
        self.results_dir.mkdir(exist_ok=True, parents=True)
        torch.save(InMemoryDataset.collate(data_list), self.results_dir / 'data.pt')

    def _iter_nodes(
            self,
            graph: int = None
    ) -> None:
        """ Iterate over nodes according to mapping. Yields pairs of (node_index, original_id)
        """
        # offset = sum(self.info.nodes[:graph]) if self.is_multi() else 0
        offset = 0
        if self.node_map is not None:
            node_map = self.node_map[graph] if self.is_multi() else self.node_map
            for ix, orig in enumerate(node_map):
                yield offset+ix, str(orig)
        else:
            for n in range(self.info.nodes[graph or 0]):
                yield offset+n, str(n)

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
        features = self.dataset_var_config.features  # dict about attributes construction
        nodes_onehot = "str_g" in features and features["str_g"] == "one_hot"

        # Read attributes
        def one_hot(
                x: int,
                values: list
        ) -> list:
            res = [0] * len(values)
            for ix, v in enumerate(values):
                if x == v:
                    res[ix] = 1
                    return res

        def as_is(
                x
        ) -> list:
            return x if isinstance(x, list) else [x]

        # TODO other encoding types from Kirill

        if self.is_multi():
            nodes = self.info.nodes[g_ix]
        else:  # single
            nodes = self.info.nodes[0]

        node_features = [[] for _ in range(nodes)]  # List of vectors

        # 1-hot encoding of all nodes
        if nodes_onehot:
            for n in range(nodes):
                vec = [0] * nodes
                vec[n] = 1
                node_features[n].extend(vec)

        def assign_feats(feat):
            for n, orig in self._iter_nodes(g_ix):
                value = feat[orig]
                assert value is not None  # FIXME misha what to do?
                node_features[n].extend(vec(value))

        # TODO misha - can optimize? read the whole files for each graph
        node_attributes = self.info.node_attributes
        assert set(features["attr"]).issubset(node_attributes["names"])
        if self.node_attributes_dir.exists():
            for ix, a in enumerate(node_attributes["names"]):
                if a not in features["attr"]: continue
                if node_attributes["types"][ix] == "categorical":
                    vec = lambda x: one_hot(x, node_attributes["values"][ix])
                else:  # "continuous", "other"
                    vec = as_is
                with open(self.node_attributes_dir / a, 'r') as f:
                    feats = json.load(f)
                    if self.is_multi():
                        feats = feats[g_ix]
                    assign_feats(feats)

        if len(node_features[0]) == 0:
            raise RuntimeError("Feature vector size must be > 0")
        return node_features
