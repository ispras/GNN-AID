import json
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData, InMemoryDataset

from aux.utils import GRAPHS_DIR
from aux.declaration import Declare
from base.custom_datasets import CustomDataset
from base.datasets_processing import DatasetManager
from base.gen_dataset import DatasetInfo, VisiblePart, GeneralDataset
from aux.configs import DatasetConfig, DatasetVarConfig, ConfigPattern


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
        # self.edge_types = set(tuple(x[1:-1] for x in et.split(','))
        #                       for et in self.info.edge_attributes.keys())
        self.edge_types = set(edge_type_from_str(f.name)
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
        return edges_dir / edge_type_to_str(src, type, dst) / 'ij'

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
            for ix, attr in enumerate(self.info.node_attributes[nt]["names"]):
                with open(self.node_attributes_dir / nt / attr, 'r') as f:
                    attributes = json.load(f)
                assert all_nodes[nt] == set(map(int, attributes.keys()))
                if self.info.node_attributes[nt]["types"][ix] == "continuous":
                    v_min, v_max = self.info.node_attributes[nt]["values"][ix]
                    assert all(isinstance(v, (int, float, complex)) for v in attributes.values())
                    assert min(attributes.values()) >= v_min
                    assert max(attributes.values()) <= v_max
                elif self.info.node_attributes[nt]["types"][ix] == "categorical":
                    assert set(attributes.values()).issubset(
                        set(self.info.node_attributes[nt]["values"][ix]))

        # Check edge attributes
        for et in self.edge_types:
            et = edge_type_to_str(*et)
            for ix, attr in enumerate(self.info.edge_attributes[et]["names"]):
                with open(self.edge_attributes_dir / et / attr, 'r') as f:
                    attributes = json.load(f)
                if self.info.edge_attributes[et]["types"][ix] == "continuous":
                    v_min, v_max = self.info.edge_attributes[et]["values"][ix]
                    assert all(isinstance(v, (int, float, complex)) for v in attributes.values())
                    assert min(attributes.values()) >= v_min
                    assert max(attributes.values()) <= v_max
                elif self.info.edge_attributes[et]["types"][ix] == "categorical":
                    assert set(attributes.values()).issubset(
                        set(self.info.edge_attributes[et]["values"][ix]))

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
        """
        Structure according to https://docs.google.com/spreadsheets/d/1fNI3sneeGoOFyIZP_spEjjD-7JX2jNl_P8CQrA4HZiI/edit#gid=1096434224
        """
        self.dataset_data = {
            "edges": [],
        }

        # Read edges and attributes
        node_map = {nt: {} for nt in self.node_types}
        edges = {edge_type_to_str(*et): [] for et in self.edge_types}
        ptg_edge_index = {et: [[], []] for et in self.edge_types}
        for et in self.edge_types:
            src, _, dst = et
            node_index = {src: 0, dst: 0}  # Note, it can be src == dst
            with open(self.edges_path(*et), 'r') as f:
                for line in f.readlines():
                    i, j = map(int, line.split())
                    if i not in node_map[src]:
                        node_map[src][i] = node_index[src]
                        node_index[src] += 1
                    if j not in node_map[dst]:
                        node_map[dst][j] = node_index[dst]
                        node_index[dst] += 1
                    if self.info.remap:
                        i = node_map[src][i]
                        j = node_map[dst][j]
                    # TODO misha can we reuse one of them?
                    edges[edge_type_to_str(*et)].append([i, j])
                    ptg_edge_index[et][0].append(i)
                    ptg_edge_index[et][1].append(j)
                    if not self.info.directed:
                        ptg_edge_index[et][0].append(j)
                        ptg_edge_index[et][1].append(i)

        self.dataset_data['edges'].append(edges)
        # FIXME misha do we create ptg data here? It duplicates dataset_data['edges']
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
        self.dataset_data['node_attributes'] = {nt: {} for nt in self.node_types}
        for nt in self.node_types:
            for ix, attr in enumerate(self.info.node_attributes[nt]["names"]):
                with open(self.node_attributes_dir / nt / attr, 'r') as f:
                    attributes = json.load(f)
                self.dataset_data['node_attributes'][nt][attr] = [{
                        node_map[nt][int(n)]: v for n, v in attributes.items()}]

        # Check for obligate parameters
        assert len(self.dataset_data['edges']) > 0
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
            data = HeteroData()
            for nt in self.node_types:
                data[nt].x = torch.tensor(node_features[nt], dtype=torch.float)
                if nt in labels:
                    data[nt].y = torch.tensor(labels[nt])
            for et in self.edge_types:
                s, t, d = et
                data[s, t, d].edge_index = self.edge_index[et]
                # TODO add edge attrs
                # data[s, t, d].edge_attr = ...
            data_list.append(data)

        # Build slices and save
        self.results_dir.mkdir(exist_ok=True, parents=True)
        torch.save(InMemoryDataset.collate(data_list), self.results_dir / 'data.pt')

    def _iter_nodes(
            self,
            graph: int = None,
            node_type: str = None,
    ) -> None:
        """ Iterate over nodes according to mapping. Yields pairs of (node_index, original_id)
        """
        offset = 0
        if self.node_map is not None:
            node_map = self.node_map[node_type]
            for ix, orig in enumerate(node_map):
                yield offset + ix, str(orig)
        else:
            for n in range(self.info.nodes[0][node_type]):
                yield offset + n, str(n)

    def _labeling_tensor(
            self,
            g_ix: int = None
    ) -> dict:
        """ Returns dict with list of labels (not tensors) """
        nt, labeling = next(iter(self.dataset_var_config.labeling.items()))
        y = []
        # Read labels
        labeling_path = self.labels_dir / nt / labeling
        with open(labeling_path, 'r') as f:
            labeling_dict = json.load(f)

        for _, orig in self._iter_nodes(node_type=nt):
            if labeling_dict[orig] is not None:
                y.append(labeling_dict[orig])
            else:
                y.append(-1)

        return {nt: y}

    def _feature_tensor(
            self,
            g_ix: int = None
    ) -> dict:
        """ Returns dict with lists of features (not tensors) for graph g_ix.
        """
        node_features_dict = {}
        for nt in self.node_types:
            features = self.dataset_var_config.features[nt]  # dict about attributes construction

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

            nodes = self.info.nodes[0][nt]

            node_features = [[] for _ in range(nodes)]  # List of vectors

            # 1-hot encoding of all nodes
            if "str_g" in features and features["str_g"] == "one_hot":
                for n in range(nodes):
                    vec = [0] * nodes
                    vec[n] = 1
                    node_features[n].extend(vec)

            # structural features
            if "str_f" in features:
                for key, value in features["str_f"].items():
                    if key == 'constant':
                        for n in range(nodes):
                            node_features[n].extend(value)
                    else:
                        raise NotImplementedError

            if 'attr' in features:

                def assign_feats(feat):
                    for n, orig in self._iter_nodes(g_ix, nt):
                        value = feat[orig]
                        assert value is not None  # FIXME misha what to do?
                        node_features[n].extend(vec(value))

                node_attributes = self.info.node_attributes[nt]
                assert set(features["attr"]).issubset(node_attributes["names"])
                # TODO misha - can optimize? read the whole files for each graph
                if self.node_attributes_dir.exists():
                    for ix, a in enumerate(node_attributes["names"]):
                        if a not in features["attr"]: continue
                        if node_attributes["types"][ix] == "categorical":
                            vec = lambda x: one_hot(x, node_attributes["values"][ix])
                        else:  # "continuous", "vector"
                            vec = as_is
                        with open(self.node_attributes_dir / nt / a, 'r') as f:
                            feats = json.load(f)
                            if self.is_multi():
                                feats = feats[g_ix]
                            assign_feats(feats)

            if len(node_features[0]) == 0:
                raise RuntimeError("Feature vector size must be > 0")
            node_features_dict[nt] = node_features
        return node_features_dict

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

        for nt, ixes in visible_part.ixes().items():
            # TODO IMP misha replace with getting data from tensors instead of keeping the whole data
            features[nt] = {}
            for ix in ixes:
                features[nt][ix] = self.dataset_var_data['node_features'][nt][ix]
            if nt in self.dataset_var_config.labeling:
                labels[nt] = {}
                for ix in ixes:
                    labels[nt][ix] = self.dataset_var_data['labels'][nt][ix]

        return dataset_var_data

    def _compute_dataset_var_data(
            self
    ) -> None:
        """ Prepare dataset_var_data for frontend on demand.
        """
        # FIXME version fail in torch-geom 2.3.1
        # self.dataset.num_classes = int(self.dataset_data["info"]["labelings"][self.dataset_var_config.labeling])

        data = self.dataset.get(0)
        node_features = {nt: data[nt].x.tolist() for nt in self.node_types}
        nt = next(iter(self.dataset_var_config.labeling.keys()))
        labels = {nt: data[nt].y.tolist()}

        self.dataset_var_data = {
            "features": node_features,
            "labels": labels,
        }


def edge_type_to_str(
        src: str,
        type: str,
        dst: str
) -> str:
    """ Hetero graph edge represent as string.
    Example: 's', 't', 'd' -> "'s','t','d'"
    """
    return f"'{src}','{type}','{dst}'"


def edge_type_from_str(
        edge_type: str
) -> tuple:
    """
    Hetero graph edge triple from string representation.
    Example: '"s","t","d"' -> ('s', 't', 'd')
    """
    return tuple(x[1:-1] for x in edge_type.split(','))


if __name__ == '__main__':
    # from torch_geometric.datasets import OGB_MAG
    # dataset_config = DatasetConfig("single-graph", "custom", "example")
    dataset_config = DatasetConfig("single-graph", "hetero", "example")
    # gen_dataset = DatasetManager.register_custom_hetero(dataset_config)
    # dataset_config = DatasetConfig("single-graph", "local", "СКЗИ")
    gen_dataset = DatasetManager.get_by_config(dataset_config)
    # # print(gen_dataset.info.to_dict())
    gen_dataset._compute_dataset_data()
    print(json.dumps(gen_dataset.dataset_data, indent=1))

    # dataset_var_config = DatasetVarConfig(
    #     features={
    #         'author': {'str_f': {'constant': [1] * 10}},
    #         'paper': {'attr': {'year': 'as_is'}},
    #         'institution': {'attr': {'rating': 'as_is'}}
    #     },
    #     labeling={'paper': 'topic'},
    #     dataset_ver_ind=0)
    # gen_dataset.build(dataset_var_config)
    # # print(gen_dataset.dataset.data)
    # gen_dataset.set_visible_part({})
    # gen_dataset.get_dataset_var_data()
    # print(json.dumps(gen_dataset.dataset_var_data, indent=1))

    # # Example of local user PTG dataset
    # class UserLocalDataset(InMemoryDataset):
    #     def __init__(self, root, transform=None):
    #         super().__init__(root, transform)
    #         # NOTE: it is important to define self.slices here, since it is used to calculate len()
    #         self.data, self.slices = torch.load(self.processed_paths[0])
    #
    #     @property
    #     def processed_file_names(self):
    #         return 'data.pt'
    #
    #     # def process(self):
    #     #     torch.save(self.collate(self.data_list), self.processed_paths[0])
    #
    # dataset = UserLocalDataset(GRAPHS_DIR / 'single-graph' / 'hetero' / 'СКЗИ' / '02b68ee2755fb9d25d64afdf8e6e018198193d8bc8b1bbc2aa805c5b97372cc5')
    # DatasetManager.register_torch_geometric_local(dataset, 'СКЗИ')

