import json
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from aux.utils import GRAPHS_DIR
from aux.declaration import Declare
from base.custom_datasets import CustomDataset
from base.datasets_processing import GeneralDataset, DatasetInfo, DatasetManager
from aux.configs import DatasetConfig, DatasetVarConfig, ConfigPattern
from base.ptg_datasets import LocalDataset


class CustomHeteroDataset(
    CustomDataset
):
    """ Hetero-graph version of user-defined dataset in 'ij' format.
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

        self.node_types = set(self.info.nodes[0].keys())
        self.edge_types = set(tuple(x[1:-1] for x in f.name.split(','))
                           for f in self.edges_path().iterdir() if f.is_dir())

    def edges_path(
            self,
            src: str = None,
            type: str = None,
            dst: str = None
    ) -> Path:
        """ Path to file with edge list for (src, type, dst). """
        edges_dir = self.root_dir / 'raw' / (self.name + '.edges')
        if any(_ is None for _ in [src, type, dst]):
            return edges_dir
        return edges_dir / edge_to_str(src, type, dst) / 'ij'

    @property
    def edge_index_path(
            self
    ) -> Path:
        """ Path to file with edge indices, for multiple graphs. """
        raise NotImplementedError("Hetero datasets with multiple graphs are not supported.")

    def check_validity(
            self
    ) -> None:
        """ Check that dataset files (graph and attributes) are valid and consistent with .info.
        """
        # Assuming info is OK

        # Check edge types (got from file names)
        assert all(et[0] in self.node_types and et[2] in self.node_types for et in self.edge_types)

        # Check nodes
        all_nodes = {nt: set() for nt in self.node_types}  # sets of nodes of each type
        for s, t, d in self.edge_types:
            with open(self.edges_path(s, t, d), 'r') as f:
                for line in f.readlines():
                    i, j = map(int, line.split())
                    all_nodes[s].add(i)
                    all_nodes[d].add(j)
                    if not self.info.directed:
                        all_nodes[s].add(j)
                        all_nodes[d].add(i)
        for nt in self.node_types:
            if self.info.remap:
                assert len(all_nodes[nt]) == self.info.nodes[0][nt]
            else:
                assert all_nodes[nt] == set(range(self.info.nodes[0][nt]))

        # Check node attributes
        for nt in self.node_types:
            for ix, attr in enumerate(self.info.node_attributes["names"][nt]):
                with open(self.node_attributes_dir / nt / attr, 'r') as f:
                    attributes = json.load(f)
                assert all_nodes[nt] == set(map(int, attributes.keys()))
                if self.info.node_attributes["types"][nt][ix] == "continuous":
                    v_min, v_max = self.info.node_attributes["values"][nt][ix]
                    assert all(isinstance(v, (int, float, complex)) for v in attributes.values())
                    assert min(attributes.values()) >= v_min
                    assert max(attributes.values()) <= v_max
                elif self.info.node_attributes["types"][nt][ix] == "categorical":
                    assert set(attributes.values()).issubset(
                        set(self.info.node_attributes["values"][nt][ix]))

        # Check edge attributes
        for et in self.edge_types:
            et = edge_to_str(*et)
            for ix, attr in enumerate(self.info.edge_attributes["names"][et]):
                with open(self.edge_attributes_dir / et / attr, 'r') as f:
                    attributes = json.load(f)
                if self.info.edge_attributes["types"][et][ix] == "continuous":
                    v_min, v_max = self.info.edge_attributes["values"][et][ix]
                    assert all(isinstance(v, (int, float, complex)) for v in attributes.values())
                    assert min(attributes.values()) >= v_min
                    assert max(attributes.values()) <= v_max
                elif self.info.edge_attributes["types"][et][ix] == "categorical":
                    assert set(attributes.values()).issubset(
                        set(self.info.edge_attributes["values"][et][ix]))

        # Check labels
        for nt in self.node_types:
            if nt not in self.info.labelings: continue
            for labelling, n_classes in self.info.labelings[nt].items():
                with open(self.labels_dir / nt / labelling, 'r') as f:
                    labels = json.load(f)
                assert all_nodes[nt] == set(map(int, labels.keys()))

    def _compute_stat(
            self,
            stat: str
    ) -> dict:
        """ Compute some additional stats
        """
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
        node_map = {nt: {} for nt in self.node_types}
        edges = {et: [] for et in self.edge_types}
        ptg_edge_index = {et: [[], []] for et in self.edge_types}
        for et in self.edge_types:
            src, _, dst = et
            src_node_index = 0
            dst_node_index = 0
            with open(self.edges_path(*et), 'r') as f:
                for line in f.readlines():
                    i, j = map(int, line.split())
                    if i not in node_map[src]:
                        node_map[src][i] = src_node_index
                        src_node_index += 1
                    if j not in node_map[dst]:
                        node_map[dst][j] = dst_node_index
                        dst_node_index += 1
                    if self.info.remap:
                        i = node_map[src][i]
                        j = node_map[dst][j]
                    # TODO misha can we reuse one of them?
                    edges[et].append([i, j])
                    ptg_edge_index[et][0].append(i)
                    ptg_edge_index[et][1].append(j)
                    if not self.info.directed:
                        ptg_edge_index[et][0].append(j)
                        ptg_edge_index[et][1].append(i)

        self.dataset_data['edges'].append(edges)
        self.edge_index = {k: [torch.tensor(np.asarray(v))] for k, v in ptg_edge_index.items()}

        # Add remaining nodes from labeling file to complete remapping
        if self.info.remap:
            for nt in self.node_types:
                node_index = len(node_map[nt])
                if node_index < self.info.nodes[0][nt]:
                    labeling_path = self.labels_dir / nt / os.listdir(self.labels_dir / nt)[0]
                    with open(labeling_path, 'r') as f:
                        labeling_dict = json.load(f)
                    for node in labeling_dict.keys():
                        node = int(node)
                        if node not in node_map[nt]:
                            node_map[nt][node] = node_index
                            node_index += 1
                    assert node_index == self.info.nodes[0][nt]
            # assert node_index == self.info.nodes[0]
            # Original ids in the order of appearance
            self.node_map = {nt: list(node_map[nt].keys()) for nt in self.node_types}
            self.info.node_info = {"id": self.node_map}

        # Read attributes
        self.dataset_data["node_attributes"] = {nt: {} for nt in self.node_types}
        for nt in self.node_types:
            for ix, attr in enumerate(self.info.node_attributes["names"][nt]):
                with open(self.node_attributes_dir / nt / attr, 'r') as f:
                    attributes = json.load(f)
                self.dataset_data["node_attributes"][nt][attr] = [{
                        node_map[nt][int(n)]: v for n, v in attributes.items()}]

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
        # TODO misha hetero
        raise NotImplementedError
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
        # TODO misha hetero
        raise NotImplementedError
        # offset = sum(self.info.nodes[:graph]) if self.is_multi() else 0
        offset = 0
        if self.node_map is not None:
            node_map = self.node_map[graph] if self.is_multi() else self.node_map
            for ix, orig in enumerate(node_map):
                yield offset + ix, str(orig)
        else:
            for n in range(self.info.nodes[graph or 0]):
                yield offset + n, str(n)

    def _labeling_tensor(
            self,
            g_ix: int = None
    ) -> list:
        """ Returns list of labels (not tensors) """
        # TODO misha hetero
        raise NotImplementedError
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
        # TODO misha hetero
        raise NotImplementedError
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


def edge_to_str(
        src: str,
        type: str,
        dst: str
) -> str:
    """ Hetero graph edge represent as string.
    """
    return f"'{src}','{type}','{dst}'"


if __name__ == '__main__':
    # dataset_config = DatasetConfig("single-graph", "custom", "example")
    dataset_config = DatasetConfig("single-graph", "hetero", "example")
    gen_dataset = DatasetManager.register_custom_hetero(dataset_config)
    gen_dataset._compute_dataset_data()
    print(gen_dataset.info.to_dict())
