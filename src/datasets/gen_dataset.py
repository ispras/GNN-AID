from pathlib import Path
from typing import Union, List, Callable, Dict
import json

import torch
from torch import default_generator, randperm, tensor, cat
# from torch._C import default_generator
# from torch._C._VariableFunctions import randperm
from torch_geometric.data import Dataset, InMemoryDataset, Data

from datasets.dataset_info import DatasetInfo
from datasets.visible_part import VisiblePart
from data_structures.configs import DatasetConfig, DatasetVarConfig, ConfigPattern
from aux.declaration import Declare


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
        # TODO misha - dataset should not know about visible part, but we use it when model sends stats over visible nodes only
        self.visible_part: VisiblePart = None  # index of visible nodes/graphs at frontend

        self.dataset: Dataset = None  # Current PTG dataset

        from datasets.dataset_stats import DatasetStats
        self.stats = DatasetStats(self)  # dict of {stat -> value}  # fixme misha - should keep it in prepared?

        # Data split
        self.percent_test_class = None  # FIXME misha do we need it here? it is in manager_config
        self.percent_train_class = None

        self.train_mask = None
        self.val_mask = None
        self.test_mask = None

        # Build graph structure
        self._compute_dataset_data()

        self._register()

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
        return Declare.dataset_root_dir(self.dataset_config)[0]

    @property
    def results_dir(
            self
    ) -> Path:
        """ Path to folder where tensor data is stored. """
        return Path(Declare.dataset_prepared_dir(self.dataset_config, self.dataset_var_config)[0])

    @property
    def raw_dir(
            self
    ) -> Path:
        """ Path to 'raw/' folder where raw data is stored. """
        return self.root_dir / 'raw'

    @property
    def info_path(
            self
    ) -> Path:
        """ Path to 'metainfo' file. """
        return Declare.dataset_info_path(self.dataset_config)

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

    def node_attributes(
            self,
            attrs: List[str] = None
    ) -> Dict[str, Union[list, torch.Tensor]]:
        """ Get node attributes as a dict {name -> list}"""
        raise RuntimeError("This should be implemented in subclass")

    def edge_attributes(
            self,
            attrs: List[str] = None
    ) -> Dict[str, Union[list, torch.Tensor]]:
        """ Get node attributes as a dict {name -> list}"""
        raise RuntimeError("This should be implemented in subclass")

    @property
    def edges(
            self
    ) -> List[torch.Tensor]:
        """ Edge index as a list of tensors """
        return [data.edge_index for data in self.dataset]

    @property
    def labels(
            self
    ) -> torch.Tensor:
        if self.is_multi():
            return tensor([data.y for data in self.dataset])
        else:
            return self.dataset[0].y

        # # fixme why do we need it? maybe other field make also
        # # NOTE: this is a copy from torch_geometric.data.dataset v=2.3.1
        # from torch_geometric.data.dataset import _get_flattened_data_list
        # data_list = _get_flattened_data_list([data for data in self.dataset])
        # return torch.cat([data.y for data in data_list if 'y' in data], dim=0)

    @property
    def node_features(
            self
    ) -> List[torch.Tensor]:
        if self.dataset is None:
            raise RuntimeError(f"Cannot get node features: dataset {self} is not built."
                               f" Define {DatasetVarConfig.__name__} and call build() method")
        if self.is_multi():
            return [data.x for data in self.dataset]
        else:
            return self.dataset[0].x

    def __len__(
            self
    ) -> int:
        return self.info.count

    def is_directed(
            self
    ) -> bool:
        """ Return whether edges in this dataset are directed. """
        return self.info.directed

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
        # Save configs
        _, [dc, dvc] = Declare.dataset_prepared_dir(self.dataset_config, self.dataset_var_config)
        with open(dc, 'w') as f:
            json.dump(self.dataset_config.to_json(), f, indent=2)
        with open(dvc, 'w') as f:
            json.dump(self.dataset_var_config.to_json(), f, indent=2)

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
        self.visible_part = VisiblePart(self, **part)

    def _register(
            self
    ) -> None:
        """ Add info about class to metainfo and save to file.
        """
        if self.info.class_name is None:
            self.info.class_name = self.__class__.__name__
            self.info.import_from = self.__class__.__module__
            self.info.save(self.info_path)

    def _compute_dataset_var_data(
            self
    ) -> None:
        """ Build graph(s) tensors - features and labels
        """
        raise RuntimeError("This should be implemented in subclass")

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
        raise RuntimeError("This should be implemented in subclass")

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
                        feats = self.node_features
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
            raise RuntimeError("percent_train_class + percent_test_class > 1")
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

    def apply_modification(
            self,
            artifact: 'GraphModificationArtifact'
    ) -> 'GeneralDataset':
        """
        Applies graph structure and feature modifications described in a GraphModificationArtifact
        to the current GeneralDataset object and returns the modified version.

        This includes:
          - Removing and adding edges
          - Adding new nodes and their features
          - Removing nodes and updating the feature matrix with index remapping
          - Modifying individual node features
          - Reindexing nodes to maintain consistent connectivity

        :param artifact: GraphModificationArtifact containing node and edge changes
        :return: self (GeneralDataset)
        """
        data: Data = self.data
        device = data.x.device if hasattr(data, 'x') else 'cpu'
        from data_structures.graph_modification_artifacts import GraphModificationArtifact
        assert isinstance(artifact, GraphModificationArtifact), (
            f"Invalid type: expected GraphModificationArtifact, got {type(artifact).__name__}"
        )
        # === Handle node operations ===
        y = data.y
        edge_index = data.edge_index
        x = data.x
        num_nodes = x.size(0)
        feature_dim = x.size(1)

        # Validate node removals
        removed_node_ids = set(int(n) for n in artifact.nodes['remove'])
        assert all(0 <= n < num_nodes for n in removed_node_ids), (
            f"Invalid node IDs in 'remove': {removed_node_ids} (max index {num_nodes - 1})"
        )

        # Validate node additions
        existing_ids = set(range(num_nodes))
        add_node_items = artifact.nodes['add'].items()
        added_node_ids = set(int(n) for n in artifact.nodes['add'].keys())
        assert not (added_node_ids & existing_ids), (
            f"Cannot add nodes with existing IDs: {added_node_ids & existing_ids}"
        )

        # Validate node feature changes
        change_features = artifact.nodes['change_f']
        for node_id, feats in change_features.items():
            nid = int(node_id)
            assert 0 <= nid < num_nodes or nid in added_node_ids, (
                f"Invalid node id {nid} in change_f: not in current or added nodes"
            )
            for feat_idx in feats:
                fid = int(feat_idx)
                assert 0 <= fid < feature_dim, (
                    f"Invalid feature index {fid} for node {nid}"
                )

        # Build initial mapping and mask for nodes to keep
        if removed_node_ids:
            keep_mask = torch.tensor([
                i not in removed_node_ids for i in range(num_nodes)
            ], dtype=torch.bool, device=device)
            remap = {}
            new_index = 0
            for old_index in range(num_nodes):
                if old_index not in removed_node_ids:
                    remap[old_index] = new_index
                    new_index += 1

            x = data.x[keep_mask]

            if hasattr(data, 'y') and data.y is not None:
                y = data.y[keep_mask]
        else:
            remap = {i: i for i in range(num_nodes)}

        # Add new nodes (after reindexing existing ones)
        if add_node_items:
            new_node_start = len(remap)
            for i, (node_id, _) in enumerate(add_node_items):
                remap[int(node_id)] = new_node_start + i

            new_features = torch.stack([
                feat.to(device) for _, feat in add_node_items
            ])
            x = torch.cat([data.x, new_features], dim=0)

            if hasattr(data, 'y') and data.y is not None:
                new_labels = torch.full(
                    (new_features.size(0),),
                    -1,
                    dtype=data.y.dtype,
                    device=device
                )
                y = torch.cat([data.y, new_labels], dim=0)

        # Modify node features
        for node_id, feature_changes in change_features.items():
            true_node_id = int(node_id)
            if true_node_id in removed_node_ids:
                continue  # Skip modifications to removed nodes

            mapped_id = remap.get(true_node_id, None)
            if mapped_id is None:
                continue

            for feat_idx, new_val in feature_changes.items():
                feat_idx = int(feat_idx)
                assert 0 <= feat_idx < feature_dim, (
                    f"Feature index {feat_idx} out of bounds for node {true_node_id}"
                )
                data.x[mapped_id, feat_idx] = new_val

        # === Edge processing ===
        edge_index_cpu = data.edge_index.cpu()
        edge_attr = getattr(data, 'edge_attr', None)
        has_edge_attr = edge_attr is not None

        edge_list = edge_index_cpu.t().tolist()
        edge_attr_list = (
            edge_attr.cpu().tolist() if has_edge_attr else [None] * len(edge_list)
        )
        current_edge_set = set((u, v) for u, v in edge_list)

        # Validate edge removals
        removed_edges_set = set((int(u), int(v)) for u, v in artifact.edges['remove'])
        assert removed_edges_set <= current_edge_set, (
            f"Some edges to remove do not exist: {removed_edges_set - current_edge_set}"
        )

        # Validate edge additions
        added_edges_set = set((int(u), int(v)) for u, v, _ in artifact.edges['add'])
        assert not (added_edges_set & current_edge_set), (
            f"Some added edges already exist: {added_edges_set & current_edge_set}"
        )

        filtered_edges = []
        filtered_attrs = []

        for idx, (u, v) in enumerate(edge_list):
            if u in removed_node_ids or v in removed_node_ids:
                continue
            if (u, v) in removed_edges_set:
                continue
            filtered_edges.append([remap[u], remap[v]])
            if has_edge_attr:
                filtered_attrs.append(edge_attr_list[idx])

        for edge in artifact.edges['add']:
            u, v, attr = edge
            u = int(u)
            v = int(v)
            if u in removed_node_ids or v in removed_node_ids:
                continue
            filtered_edges.append([remap.get(u, u), remap.get(v, v)])
            if has_edge_attr:
                filtered_attrs.append(attr.tolist() if attr is not None else [0.0])

        edge_index_tensor = torch.tensor(filtered_edges, dtype=torch.long).t().contiguous()
        edge_index = edge_index_tensor.to(device)

        if has_edge_attr:
            edge_attr_tensor = torch.tensor(
                filtered_attrs, dtype=edge_attr.dtype
            ).to(device)
            edge_attr = edge_attr_tensor

        # === Update GeneralDataset properties ===
        self.dataset.data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
        # self.dataset.num_classes = int(data.y.max().item()) + 1 if hasattr(data, 'y') and data.y is not None else 0
        # self.dataset.num_node_features = data.x.size(1)
        self._labels = data.y if hasattr(data, 'y') else None

        return self


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
            save: bool = True
    ):
        """
        :param data_list: optionally, list of ready torch_geometric.data.Data objects
        :param results_dir: directory where tensors are stored
        :param process_func: optionally, custom process() function, which converts raw files into tensors
        :param save: if True (which is default), save tensors
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


class ExampleGenDataset(GeneralDataset):
    """ Example of user defined GeneralDataset
    """
    def __init__(self, dataset_config: Union[DatasetConfig, ConfigPattern]):
        super().__init__(dataset_config)

    def _compute_dataset_data(
            self
    ) -> None:
        """
        """
        import os
        import json
        import numpy as np
        self.info = DatasetInfo.read(self.info_path)
        self.dataset_data = {}

        node_map = {}
        ptg_edge_index = [[], []]
        node_index = 0
        with open(self.root_dir / 'raw' / 'edges.ij', 'r') as f:
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
        self.node_attributes_dir = self.root_dir / 'raw' / (self.name + '.node_attributes')
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
        x = tensor([[0, 0], [1, 0], [1, 0]])
        edge_index = tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        y = tensor([0, 1, 1])

        data_list = [Data(x=x, edge_index=edge_index, y=y)]
        self.dataset = LocalDataset(data_list, self.results_dir)

        # Define auxiliary fields
        self.dataset_var_data = {}

        self.dataset_var_data['node_features'] = [data.x for data in self.dataset]
        # self.dataset_var_data['edge_features'] = []
        self.dataset_var_data['labels'] = [data.y for data in self.dataset]

        self.dataset_var_data['node_features'] = self.dataset_var_data['node_features'][0]
        # self.dataset_var_data['edge_features'] = []
        self.dataset_var_data['labels'] = self.dataset_var_data['labels'][0]


if __name__ == '__main__':
    print("test dataset")
    from datasets.ptg_datasets import LocalPTGDataset, PTGDataset, LibPTGDataset
    from datasets.known_format_datasets import KnownFormatDataset
    from datasets.datasets_manager import DatasetManager

    # dc = DatasetConfig(('example', 'single-graph', 'example'))
    dc = DatasetConfig(('example', 'multiple-graphs', 'example_gml'))
    # dc = DatasetConfig(('ptg-library-graphs', 'single-graph', 'Planetoid', 'Cora'))
    # dc = DatasetConfig(('ptg-library-graphs', 'multiple-graphs', 'TUDataset', 'MUTAG'))
    root = Declare.dataset_root_dir(dc)[0]
    print(root)
    dataset = DatasetManager.get_by_config(
        dc,
        default_edge_attr_value={'type': "mixed", 'weight': 0},
        default_node_attr_value={'b': "alpha"})

    dataset.set_visible_part({'center': 0, 'depth': 0})
    # dataset.set_visible_part({})
    print(dataset.visible_part.get_dataset_data())

    dvc = DatasetVarConfig(features={'attr': {'a': 'as_is'}}, labeling='binary', dataset_ver_ind=0)

    dataset.build(dvc)
    # print(dataset.dataset_var_data)
    print(dataset.visible_part.get_dataset_var_data())
