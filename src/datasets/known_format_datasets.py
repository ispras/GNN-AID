import json
import os
import shutil
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from aux.declaration import Declare
from aux.utils import tmp_dir
from data_structures.configs import DatasetConfig, DatasetVarConfig, ConfigPattern
from datasets.dataset_converter import DatasetConverter
from datasets.gen_dataset import LocalDataset, GeneralDataset
from datasets.dataset_info import DatasetInfo


class KnownFormatDataset(
    GeneralDataset
):
    """
    Fully custom dataset: with building graph from raw files and custom feature creation.
    """
    def __init__(
            self,
            dataset_config: Union[ConfigPattern, DatasetConfig],
            format: str = 'ij',
            default_node_attr_value: dict = None,
            default_edge_attr_value: dict = None,
    ):
        """
        Args:
            dataset_config: DatasetConfig dict from frontend
            format: one of known formats: ij, xml,
            default_node_attr_value: dict with default node attributes values to apply where
             missing.
            default_edge_attr_value: dict with default edge attributes values to apply where
             missing.
        """
        assert format == 'ij' or format in DatasetConverter.supported_formats
        self._format = format
        self._default_node_attr_value = default_node_attr_value
        self._default_edge_attr_value = default_edge_attr_value
        self.node_map = None  # Optional nodes mapping: node_map[i] = original id of node i

        super(KnownFormatDataset, self).__init__(dataset_config)

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
        """ Path to file with edge indices, for multiple graphs. """
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

        # Look for obligate files: .info, graph(s), a dir with labels
        # info_file = None
        label_dir = None
        graph_files = []
        path = Declare.dataset_root_dir(self.dataset_config)[0] / 'raw'
        for p in path.iterdir():
            # if p.is_file() and p.name == '.info':
            #     info_file = p
            if p.is_file() and p.name.endswith(f'.{self._format}'):
                graph_files.append(p)
            if p.is_dir() and p.name.endswith('.labels'):
                label_dir = p
        # if info_file is None:
        #     raise RuntimeError(f"No .info file was found at {path}")
        if len(graph_files) == 0:
            raise RuntimeError(f"No files with extension '.{self._format}' found at {path}. "
                               f"If your graph is heterograph, use 'register_custom_hetero()'")
        if label_dir is None:
            raise RuntimeError(f"No file with extension '.label' found at {path}")

        # Order of files is important, should be consistent with .info, we suppose they are sorted
        graph_files = sorted(graph_files)

        # Create a temporary dir to store converted data
        with tmp_dir(path) as tmp:
            # Convert the data if necessary, write it to an empty directory
            if self._format != 'ij':
                from datasets.dataset_converter import DatasetConverter
                DatasetConverter.format_to_ij(self.info, graph_files, self._format, tmp,
                                              self._default_node_attr_value, self._default_edge_attr_value)

            # Move or copy original contents to a temporary dir
            merge_directories(path, tmp, True)

            # Rename the newly created dir to the original one
            tmp.rename(path)

        self.check_validity()

    def _compute_dataset_data(
            self
    ) -> None:
        """
        """
        self.info = DatasetInfo.read(self.info_path)
        if self._format != 'ij':
            self._convert_to_ij()

        self.dataset_data = {}

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
                ptg_edge_index = [[], []]  # Over each graph
                node_map = {}
                node_maps.append(node_map)
                self.dataset_data['edges'] = []
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
                        self.dataset_data['edges'].append(torch.tensor(np.asarray(ptg_edge_index)))
                        g_ix += 1
                        if g_ix == count:
                            break
                        node_index = 0
                        # edges = []
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
            assert len(self.dataset_data['edges']) == self.info.count

            # Read attributes
            self.dataset_data['node_attributes'] = {}
            for a in self.info.node_attributes["names"]:
                with open(self.node_attributes_dir / a, 'r') as f:
                    self.dataset_data['node_attributes'][a] = json.load(f)

        else:
            node_map = {}
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
                    ptg_edge_index[0].append(i)
                    ptg_edge_index[1].append(j)
                    if not self.info.directed:
                        ptg_edge_index[0].append(j)
                        ptg_edge_index[1].append(i)

                self.dataset_data['edges'] = [torch.tensor(np.asarray(ptg_edge_index))]
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
            assert len(self.dataset_data['edges']) == self.info.count

            # Read attributes
            self.dataset_data['node_attributes'] = {}
            if self.node_attributes_dir.exists():
                for a in os.listdir(self.node_attributes_dir):
                    with open(self.node_attributes_dir / a, 'r') as f:
                        self.dataset_data['node_attributes'][a] = [{
                            node_map[int(n)]: v for n, v in json.load(f).items()
                            if int(n) in node_map}]

        # Check for obligate parameters
        assert len(self.dataset_data['edges']) > 0
        # assert len(info["labelings"]) > 0  # for VK we generate based on files

    def _compute_dataset_var_data(
            self
    ) -> None:
        """ Build ptg dataset based on dataset_var_config and create DatasetVarData.
        """
        # self.dataset_var_data = None
        # self.stats.update_var_config()  fixme
        # self.dataset_var_config = dataset_var_config
        self.dataset = LocalDataset(None, self.results_dir, process_func=self._create_ptg)

        # Define auxiliary fields
        self.dataset_var_data = {}
        if self.is_hetero():
            raise NotImplementedError

        else:
            self.dataset_var_data['node_features'] = [data.x for data in self.dataset]
            # self.dataset_var_data['edge_features'] = []
            self.dataset_var_data['labels'] = [data.y for data in self.dataset]

        if not self.is_multi():  # fixme do we really want it ?
            self.dataset_var_data['node_features'] = self.dataset_var_data['node_features'][0]
            # self.dataset_var_data['edge_features'] = []
            self.dataset_var_data['labels'] = self.dataset_var_data['labels'][0]

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
                x=x, edge_index=self.dataset_data['edges'][ix], y=y,
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
                    break
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


def merge_directories(
        source_dir: Union[Path, str],
        destination_dir: Union[Path, str],
        remove_source: bool = False
) -> None:
    """
    Merge source directory into destination directory, replacing existing files.

    :param source_dir: Path to the source directory to be merged
    :param destination_dir: Path to the destination directory
    :param remove_source: if True, remove source directory (empty folders)
    """
    for root, _, files in os.walk(source_dir):
        # Calculate relative path
        relative_path = os.path.relpath(root, source_dir)

        # Create destination path
        dest_path = os.path.join(destination_dir, relative_path)
        os.makedirs(dest_path, exist_ok=True)

        # Move files
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_path, file)
            shutil.move(src_file, dest_file)

    if remove_source:
        shutil.rmtree(source_dir)
