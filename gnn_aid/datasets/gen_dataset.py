import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Callable, Dict

import torch
from torch import default_generator, randperm, tensor
from torch_geometric.data import Dataset, InMemoryDataset, Data
from torch_geometric.data.collate import collate

from gnn_aid.aux.declaration import Declare
from gnn_aid.aux.utils import root_dir
from gnn_aid.data_structures.configs import DatasetConfig, DatasetVarConfig, ConfigPattern, FeatureConfig, \
    Task
from .dataset_info import DatasetInfo


class GeneralDataset(ABC):
    """
    Abstract class to represent a dataset in framework.
    It is a generalisation of a :class:`torch_geometric.data.Dataset` to a family of such datasets.
    To fully define a GeneralDataset you should provide 2 configs:

    - :class:`~data_structures.configs.DatasetConfig` - specifies structure of the graph;
    - :class:`~data_structures.configs.DatasetVarConfig` - specifies how to build features and
      labels.

    :class:`~datasets.gen_dataset.GeneralDataset` also requires :class:`~datasets.DatasetInfo` which
    describes metainformation about the dataset family.

    See also:

    - :class:`datasets.ptg_datasets.PTGDataset`
    - :class:`datasets.known_format_datasets.KnownFormatDataset`
    """

    def __init__(
            self,
            dataset_config: Union[DatasetConfig, ConfigPattern]
    ):
        """
        Initialize structural part of the dataset.

        Args:
            dataset_config (DatasetConfig | ConfigPattern): config to define where to get raw files
             and initialization parameters. Can be :class:`~data_structures.configs.DatasetConfig`
             object or :class:`~data_structures.configs.ConfigPattern` with dictionary.
        """
        # TODO check that dataset_config is allowed

        # = Configs
        self.dataset_config = dataset_config
        self.dataset_var_config: DatasetVarConfig = None
        self.info: DatasetInfo = None

        # = Variable part
        self.dataset: Dataset = None  # Current PTG dataset
        self._data: Data = None

        # Statistics
        from .dataset_stats import DatasetStats
        self.stats = DatasetStats(self)  # dict of {stat -> value}

        # Feature vector components
        self.node_attr_slices: dict = None  # indices of each attribute in feature vector, dict of {attr -> (start, end)}
        # self.edge_attr_slices: dict = None

        # Data split
        self.percent_test_class = None  # FIXME misha do we need it here? it is in manager_config
        self.percent_train_class = None

        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.edge_label_index = None  # [train, val, test] edges indices concatenated
        self.edge_labels = None  # labels of edges corresponding to edge_label_index

        # Build graph structure
        self._compute_dataset_data()

        self._register()

    @property
    def root_dir(
            self
    ) -> Path:
        """ Dataset root directory with folder 'raw/' and metainfo file. """
        return Declare.dataset_root_dir(self.dataset_config)[0]

    @property
    def raw_dir(
            self
    ) -> Path:
        """ Path to 'raw/' folder where raw data is stored.
        For ptg datasets, dataset.root_dir folder should point here.
        """
        return self.root_dir / 'raw'

    @property
    def prepared_dir(
            self
    ) -> Path:
        """ Path to folder where tensor data is stored. """
        return Path(Declare.dataset_prepared_dir(self.dataset_config, self.dataset_var_config)[0])

    @property
    def metainfo_path(
            self
    ) -> Path:
        """ Path to 'metainfo' file. """
        return Declare.dataset_info_path(self.dataset_config)

    @property
    def data(
            self
    ) -> Data:
        """
        Get tensors of the whole dataset as :class:`torch_geometric.data.data.Data` object.

        NOTE: this will load the whole dataset into memory, be careful if the size is large.
        """
        if self._data is None:
            if self.dataset is None:
                raise RuntimeError(
                    f"PyG dataset is not defined. Didn't you forget to build() the dataset?"
                    f" If so, define {DatasetVarConfig.__name__} and call build() method.")

            if len(self.dataset) > 1:
                if isinstance(self.dataset, InMemoryDataset):
                    self._data = self.dataset._data
                    if self.dataset.transform is not None:
                        self._data = self.dataset.transform(self._data)
                else:
                    from warnings import warn
                    warn("The ptg dataset is not InMemoryDataset. "
                         "Getting its data might consume too much resources if the dataset is "
                         "large. Consider using dataset[i] to get Data for i graph.")

                    # transform is applied within __get_item__
                    data_list = [self.dataset[i] for i in range(self.info.count)]
                    self._data, _, _ = collate(Data, data_list)

            else:
                self._data = self.dataset[0]

        return self._data

    @property
    def edges(
            self
    ) -> List[torch.Tensor]:
        """ Edge index as a list of tensors """
        # NOTE for the first time this makes shallow copy of all data
        return [data.edge_index for data in self.dataset]

    @abstractmethod
    def node_attributes(
            self,
            attrs: List[str] = None
    ) -> Dict[str, Union[list, torch.Tensor]]:
        """ Get node attributes as a dict {name -> list}"""

    @abstractmethod
    def edge_attributes(
            self,
            attrs: List[str] = None
    ) -> Dict[str, Union[list, torch.Tensor]]:
        """ Get node attributes as a dict {name -> list}"""

    @property
    def labels(
            self
    ) -> torch.Tensor:
        task = self.dataset_var_config.task
        if task.is_node_level():
            return self.data.y
        elif task.is_graph_level():
            return tensor([data.y for data in self.dataset])
        elif task.is_edge_level():
            return self.edge_labels
        else:
            raise ValueError(f"Unsupported task type {task}")

    @property
    def num_classes(
            self
    ) -> int:
        # FIXME depends on task
        return self.dataset.num_classes

    @property
    def num_node_features(
            self
    ) -> int:
        # FIXME depends on task?
        return self.dataset.num_node_features

    @property
    def node_features(
            self
    ) -> List[torch.Tensor]:
        """ Get a list of tensors of node features, one for each graph in the dataset.
        """
        if self.dataset is None:
            raise RuntimeError(f"Cannot get node features: dataset {self} is not built."
                               f" Define {DatasetVarConfig.__name__} and call build() method")
        # if self.is_multi():
        #     return [data.x for data in self.dataset]
        # else:
        #     return self.dataset[0].x
        return self.data.x

    @property
    def edge_features(
            self
    ) -> List[torch.Tensor]:
        """ Get a list of tensors of edge features, one for each graph in the dataset.
        """
        if self.dataset is None:
            raise RuntimeError(f"Cannot get edge features: dataset {self} is not built."
                               f" Define {DatasetVarConfig.__name__} and call build() method")

        # if self.is_multi():
        #     return [data.edge_attr for data in self.dataset]
        # else:
        #     return self.dataset[0].edge_attr
        # TODO misha implement
        return self.data.edge_attr

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

        # Recompute var data
        self._compute_dataset_var_data()

        # Save configs
        _, [dc, dvc] = Declare.dataset_prepared_dir(self.dataset_config, self.dataset_var_config)
        with open(dc, 'w') as f:
            json.dump(self.dataset_config.to_json(), f, indent=2)
        with open(dvc, 'w') as f:
            json.dump(self.dataset_var_config.to_json(), f, indent=2)

    @abstractmethod
    def _compute_dataset_data(
            self
    ) -> None:
        """ Build graph(s) structure - edge index
        Structure according to https://docs.google.com/spreadsheets/d/1fNI3sneeGoOFyIZP_spEjjD-7JX2jNl_P8CQrA4HZiI/edit#gid=1096434224
        """

    def _register(
            self
    ) -> None:
        """ Finalize the dataset if not yet. Have effect on the first call.
        Add info about class to metainfo.
        """
        if self.info.class_name is None:
            import inspect
            self.info.class_name = self.__class__.__name__
            file_path = Path(inspect.getfile(self.__class__))
            parts = list(file_path.relative_to(root_dir).parts)
            # remove extension
            parts[-1] = parts[-1].removesuffix(file_path.suffix)
            self.info.import_from = '.'.join(parts)
            print(f"Registered graph with class_name={self.info.class_name}")
            self.info.save(self.metainfo_path)

    @abstractmethod
    def _compute_dataset_var_data(
            self
    ) -> None:
        """ Build graph(s) tensors - features and labels
        """

    def get_stat(
            self,
            stat: str
    ) -> Union[int, float, dict, str]:
        """ Get statistics.
        """
        return self.stats.get(stat)

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

        task_type = self.dataset_var_config.task
        if task_type in [Task.NODE_CLASSIFICATION, Task.NODE_REGRESSION, Task.GRAPH_CLASSIFICATION, Task.GRAPH_REGRESSION]:
            train_mask = torch.BoolTensor([False] * self.labels.size(dim=0))
            val_mask = torch.BoolTensor([False] * self.labels.size(dim=0))
            test_mask = torch.BoolTensor([False] * self.labels.size(dim=0))

            labeled_nodes_numbers = [n for n, y in enumerate(self.labels) if y != -1]
            num_train = int(percent_train_class * len(labeled_nodes_numbers))
            num_test = int(percent_test_class * len(labeled_nodes_numbers))
            num_val = len(labeled_nodes_numbers) - num_train - num_test
            if percent_val_class <= 0 and num_val > 0:
                num_test += num_val
                num_val = 0
            split = randperm(num_train + num_val + num_test, generator=default_generator).tolist()

            for elem in split[:num_train]:
                train_mask[labeled_nodes_numbers[elem]] = True
            for elem in split[num_train: num_train + num_val]:
                val_mask[labeled_nodes_numbers[elem]] = True
            for elem in split[num_train + num_val:]:
                test_mask[labeled_nodes_numbers[elem]] = True

        elif task_type in [Task.EDGE_PREDICTION, Task.EDGE_REGRESSION]:
            # Split all edges to train/val/test
            # No negative sampling - it will be done in LinkLoader while model training
            from torch_geometric.transforms import RandomLinkSplit

            rls = RandomLinkSplit(
                num_val=percent_val_class,
                num_test=percent_test_class,
                is_undirected=not self.info.directed,
                neg_sampling_ratio=0
            )

            train_data, val_data, test_data = rls(self.data)

            full_edge_label_index = torch.cat([
                train_data.edge_label_index,
                val_data.edge_label_index,
                test_data.edge_label_index
            ], dim=1)
            self.edge_label_index = full_edge_label_index

            # TODO if edges have labels, use them. otherwise - all edges are 1s
            self.edge_labels = torch.cat([
                train_data.edge_label,
                val_data.edge_label,
                test_data.edge_label
            ])

            total_edges = full_edge_label_index.size(1)

            train_mask = torch.zeros(total_edges, dtype=torch.bool)
            train_mask[:train_data.edge_label_index.size(1)] = True

            val_mask = torch.zeros(total_edges, dtype=torch.bool)
            val_mask[train_data.edge_label_index.size(1):
                     train_data.edge_label_index.size(1) + val_data.edge_label_index.size(1)] = True

            test_mask = torch.zeros(total_edges, dtype=torch.bool)
            if test_data.edge_label_index.size(1) > 0:
                test_mask[-test_data.edge_label_index.size(1):] = True
        else:
            raise ValueError(f"Unsupported task type {task_type}")

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

    def save_train_test_mask(
            self,
            path: Union[str, Path]
    ) -> None:
        """ Save current train/test mask to a given path (together with the model). """
        path.mkdir(exist_ok=True, parents=True)
        torch.save([self.train_mask, self.val_mask, self.test_mask,
                    (self.percent_train_class, self.percent_test_class)],
                   path / 'train_test_split')

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
        from gnn_aid.data_structures.graph_modification_artifacts import GraphModificationArtifact
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
        self._data = None  # to be recomputed
        # self.dataset.num_classes = int(data.y.max().item()) + 1 if hasattr(data, 'y') and data.y is not None else 0
        # self.dataset.num_node_features = data.x.size(1)
        self._labels = data.y if hasattr(data, 'y') else None

        return self


class LocalDataset(
    InMemoryDataset
):
    """ Locally saved PTG Dataset. Does not write anything to data/ folder
    """

    def __init__(
            self,
            data_list: Union[List[Data], None],
            prepared_dir: Union[str, Path],
            process_func: Union[Callable, None] = None,
            processed_file_names: Union[Callable, None] = None,
            **in_memory_dataset_kwargs
    ):
        """
        :param data_list: optionally, list of ready torch_geometric.data.Data objects
        :param prepared_dir: directory where tensors are stored
        :param process_func: optionally, custom process() function, which converts raw files into tensors
        :param in_memory_dataset_kwargs: optionally, kwargs to InMemoryDataset constructor, e.g. transform
        """
        self.data_list = data_list
        self._prepared_dir = prepared_dir
        if process_func:
            self.process = process_func
        if processed_file_names:
            self._processed_file_names = processed_file_names

        # Init and process
        super().__init__(**in_memory_dataset_kwargs)

        # Load, (even if data_list is given, to correctly define self._data, self._data_list)
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
        try:
            return self._processed_file_names()
        except (NotImplementedError, AttributeError):
            return 'data.pt'

    def process(
            self
    ) -> None:
        torch.save(self.collate(self.data_list), self.processed_paths[0])

    @property
    def processed_dir(
            self
    ) -> str:
        return self._prepared_dir

    @processed_file_names.setter
    def processed_file_names(self, value):
        self._processed_file_names = value
