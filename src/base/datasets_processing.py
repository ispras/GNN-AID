import json
import shutil
import os
from pathlib import Path
from typing import Union

import torch
import torch_geometric
from torch import default_generator, randperm
from torch_geometric.data import Dataset, InMemoryDataset, Data

from aux.configs import DatasetConfig, DatasetVarConfig, ConfigPattern
from aux.custom_decorators import timing_decorator
from aux.declaration import Declare
from aux.utils import TORCH_GEOM_GRAPHS_PATH, tmp_dir


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
        self.nodes: list = None
        self.remap: bool = False
        self.node_attributes: dict = None
        self.edge_attributes: dict = {
            "names": [],
            "types": [],
            "values": []
        }
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
        assert all(key in self.node_attributes for key in ["names", "types", "values"])
        for attributes in [self.node_attributes, self.edge_attributes]:
            for n, t, v in zip(attributes["names"], attributes["types"], attributes["values"]):
                assert isinstance(n, str)
                assert t in {"continuous", "categorical", "other"}
                if t == "continuous":
                    assert isinstance(v, list) and len(v) == 2 and v[0] < v[1]
                elif t == "continuous":
                    assert isinstance(v, list)
                elif t == "other":
                    assert isinstance(v, int) and v > 0
        assert len(self.labelings) > 0
        for k, v in self.labelings.items():
            # TODO Misha - what about regression?
            assert isinstance(k, str)
            assert isinstance(v, int) and v > 1

    def check_consistency(
            self
    ) -> None:
        """ Check existing fields are consistent. """
        assert self.count == len(self.nodes)
        assert len(self.node_attributes["names"]) == len(self.node_attributes["types"]) == len(
            self.node_attributes["values"])
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
        from base.ptg_datasets import is_graph_directed
        assert self.directed == is_graph_directed(dataset.get(0))
        assert self.remap is False
        assert len(self.node_attributes["names"]) == 1
        assert self.node_attributes["types"][0] == "other"
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
            json.dump(not_nones, f, indent=1)

    @staticmethod
    def induce(
            dataset: Dataset
    ) -> object:
        """ Induce metainfo from a given PTG dataset.
        """
        res = DatasetInfo()
        res.count = len(dataset)
        from base.ptg_datasets import is_graph_directed
        res.directed = is_graph_directed(dataset.get(0))
        res.nodes = [len(dataset.get(ix).x) for ix in range(len(dataset))]
        res.node_attributes = {
            "names": ["unknown"],
            "types": ["other"],
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
            a_dict = json.load(f)
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
        node_attr_slices = {}
        if node_attributes:
            start_attr_index = 0
            for i in range(len(node_attributes['names'])):
                if node_attributes['types'][i] == 'other':
                    attr_len = node_attributes['values'][i]
                elif node_attributes['types'][i] == 'categorical':
                    attr_len = len(node_attributes['values'][i])
                else:
                    attr_len = 1
                node_attr_slices[node_attributes['names'][i]] = (
                    start_attr_index, start_attr_index + attr_len)
                start_attr_index = start_attr_index + attr_len

        edge_attr_slices = {}
        if edge_attributes:
            start_attr_index = 0
            for i in range(len(edge_attributes['names'])):
                if edge_attributes['types'][i] == 'other':
                    attr_len = edge_attributes['values'][i]
                elif edge_attributes['types'][i] == 'categorical':
                    attr_len = len(edge_attributes['values'][i])
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
            center: [int, list] = None,
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

        self._ixes = None

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

                nodes = {0: {center}}
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
                self._ixes = list(range(self.nodes))

    def ixes(
            self
    ) -> list:
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
    """ Generalisation of PTG and user-defined datasets: custom, VK, etc.
    """

    def __init__(
            self,
            dataset_config: Union[DatasetConfig, ConfigPattern]
    ):
        """
        Args:
            dataset_config: DatasetConfig dict from frontend
        """
        # Configuration
        self.dataset_config = dataset_config
        self.dataset_data = None  # structure data prepared for frontend
        self.visible_part: VisiblePart = None  # index of visible nodes/graphs at frontend

        self.dataset_var_config: DatasetVarConfig = None
        self.dataset_var_data = None  # features data prepared for frontend

        self.name = self.dataset_config.graph  # Last folder name
        from base.dataset_stats import DatasetStats
        self.stats = DatasetStats(self)  # dict of {stat -> value}
        self.info: DatasetInfo = None

        self.dataset: Dataset = None  # PTG dataset

        # Train/test mask config
        self.percent_test_class = None  # FIXME misha do we need it here? it is in manager_config
        self.percent_train_class = None

        self.train_mask = None
        self.val_mask = None
        self.test_mask = None

        # To be inferred
        self._labels = None

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
        """ Path to '.info' file. """
        return self.root_dir / 'raw' / '.info'

    @property
    def data(
            self
    ) -> Data:
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
        if self._labels is None:
            # NOTE: this is a copy from torch_geometric.data.dataset v=2.3.1
            from torch_geometric.data.dataset import _get_flattened_data_list
            data_list = _get_flattened_data_list([data for data in self.dataset])
            self._labels = torch.cat([data.y for data in data_list if 'y' in data], dim=0)
        return self._labels

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

    def build(
            self,
            dataset_var_config: Union[ConfigPattern, DatasetVarConfig]
    ) -> None:
        """ Create node feature tensors from attributes based on dataset_var_config.
        """
        raise NotImplementedError()

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
        num = len(self.dataset)
        data_list = [self.dataset.get(ix) for ix in range(num)]
        is_directed = self.info.directed
        # node_attributes exist for custom datasets.
        # We can treat them as PTG features but it's not good.
        node_attributes = {}
        # name_type = self.dataset_var_config.features['attr']
        #
        # if self.is_multi():
        #     edges_list = []
        #     self.nodes = []
        #     for data in data_list:
        #         edges_list.append(data.edge_index.T.tolist())
        #         self.nodes.append(len(data.x))
        #
        #     node_attributes = {
        #         list(name_type.keys())[0]: [data.x.tolist() for data in data_list]
        #     }
        #
        # else:
        #     assert len(data_list) == 1
        #     data = data_list[0]
        #
        #     self.nodes = [len(data.x)]
        #     node_attributes = {
        #         list(name_type.keys())[0]: [data.x.tolist()]
        #     }

        edges_list = []
        for data in data_list:
            edges = []
            dataset_edge_index = data.edge_index.tolist()
            for i in range(len(dataset_edge_index[0])):
                if not is_directed:
                    if dataset_edge_index[0][i] <= dataset_edge_index[1][i]:
                        edges.append([dataset_edge_index[0][i], dataset_edge_index[1][i]])
                else:
                    edges.append([dataset_edge_index[0][i], dataset_edge_index[1][i]])

            edges_list.append(edges)

        self.dataset_data = {
            'edges': edges_list,
            'node_attributes': node_attributes,
            # 'info': self.info.to_dict()
        }
        # if self.info.name == "":
        #     self.dataset_data['info']['name'] = '/'.join(self.dataset_config.full_name())

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
            # FIXME replace with getting data from tensors instead of keeping the whole data
            features[ix] = self.dataset_var_data['features'][ix]
            labels[ix] = self.dataset_var_data['labels'][ix]

        return dataset_var_data

    def _compute_dataset_var_data(
            self
    ) -> None:
        """ Prepare dataset_var_data for frontend on demand.
        """
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
                    elif features['attr'][attr] == 'other':
                        # Check honestly each feature vector
                        feats = self.dataset_var_data['features']
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


class DatasetManager:
    """
    Class for working with datasets. Methods: get - loads dataset in torch_geometric format for gnn
    along the path specified in full_name as tuple. Currently also supports automatic loading and
    processing of all datasets from torch_geometric.datasets
    """

    @staticmethod
    def register_torch_geometric_local(
            dataset: InMemoryDataset,
            name: str = None
    ) -> GeneralDataset:
        """
        Save a given PTG dataset locally.
        Dataset is then always available for use by its config.

        :param dataset: torch_geometric.data.Dataset object.
        :param name: user defined dataset name. If not specified, a timestamp will be used.
        :return: GeneralDataset
        """
        gen_dataset = DatasetManager._register_torch_geometric(
            dataset, name=name, group='local', exists_ok=False, copy_data=True)

        return gen_dataset

    # QUE Misha, Kirill - can we use get_by_config always instead of it?
    @staticmethod
    @timing_decorator
    def get_by_config(
            dataset_config: DatasetConfig,
            dataset_var_config: DatasetVarConfig = None
    ) -> GeneralDataset:
        """ Get GeneralDataset by dataset config. Used from the frontend.
        """
        dataset_group = dataset_config.group
        # TODO misha - better make a more hierarchical grouping?
        if dataset_group in ["custom"]:
            from base.custom_datasets import CustomDataset
            gen_dataset = CustomDataset(dataset_config)

        elif dataset_group in ["vk_samples"]:
            # TODO misha - it is a kind of custom?
            from base.vk_datasets import VKDataset
            gen_dataset = VKDataset(dataset_config)

        else:
            from base.ptg_datasets import PTGDataset
            gen_dataset = PTGDataset(dataset_config)

        if dataset_var_config is not None:
            gen_dataset.build(dataset_var_config)

        return gen_dataset

    @staticmethod
    @timing_decorator
    def get_by_full_name(
            full_name=None,
            **kwargs
    ) -> [GeneralDataset, torch_geometric.data.Data, Path]:
        """
        Get PTG dataset by its full name tuple.
        Starts the creation of an object from raw data or takes already saved datasets in prepared
        form.

        Args:
            full_name: full name of graph data
            **kwargs: other arguments required to create datasets (dataset_var_config)

        Returns: GeneralDataset, a list of tensors with data, and
        the path where the dataset is saved.

        """
        from base.ptg_datasets import PTGDataset
        dataset = DatasetManager.get_by_config(DatasetConfig.from_full_name(full_name))
        cfg = PTGDataset.dataset_var_config.to_saveable_dict()
        cfg.update(**kwargs)
        dataset_var_config = DatasetVarConfig(**cfg)
        dataset.build(dataset_var_config=dataset_var_config)
        dataset.train_test_split(percent_train_class=kwargs.get("percent_train_class", 0.8),
                                 percent_test_class=kwargs.get("percent_test_class", 0.2))
        # IMP Kirill suggest to return only dataset, else is its parts
        return dataset, dataset.data, dataset.results_dir

    @staticmethod
    def register_torch_geometric_api(
            dataset: Dataset,
            name: str = None,
            obj_name: str = 'DATASET_TO_EXPORT'
    ) -> GeneralDataset:
        """
        Register a user defined code implementing a PTG dataset.
        This function should be called at each framework run to make the dataset available for use.

        :param dataset: torch_geometric.data.Dataset object.
        :param name: user defined dataset name. If not specified, a timestamp will be used.
        :param obj_name: variable name to locate when import. DATASET_TO_EXPORT is default
        :return: GeneralDataset
        """
        gen_dataset = DatasetManager._register_torch_geometric(
            dataset, name=name, group='api', exists_ok=False, copy_data=False)

        # Save api info
        import inspect
        import_path = Path(inspect.getfile(dataset.__class__))
        api = {
            'import_path': str(import_path),
            'obj_name': obj_name,
        }
        with gen_dataset.api_path.open('w') as f:
            json.dump(api, f, indent=1)

        return gen_dataset

    @staticmethod
    def register_custom(
            dataset_config: DatasetConfig,
            format: str = 'ij',
            default_node_attr_value: dict = None,
            default_edge_attr_value: dict = None,
    ) -> GeneralDataset:
        """
        Create GeneralDataset from user created files in one of the supported formats.
        Attribute files created by user have priority over attributes extracted from the graph file.

        :param dataset_config: config for a new dataset. Files will be searched for in the folder
         defined by this config.
        :param format: one of the supported formats.
        :param default_node_attr_value: dict with default node attributes values to apply where
         missing.
        :param default_edge_attr_value: dict with default edge attributes values to apply where
         missing.
        :return: CustomDataset
        """
        # Create empty CustomDataset
        from base.custom_datasets import CustomDataset
        gen_dataset = CustomDataset(dataset_config)

        # Look for obligate files: .info, graph(s), a dir with labels
        # info_file = None
        label_dir = None
        graph_files = []
        path = gen_dataset.raw_dir
        for p in path.iterdir():
            # if p.is_file() and p.name == '.info':
            #     info_file = p
            if p.is_file() and p.name.endswith(f'.{format}'):
                graph_files.append(p)
            if p.is_dir() and p.name.endswith('.labels'):
                label_dir = p
        # if info_file is None:
        #     raise RuntimeError(f"No .info file was found at {path}")
        if len(graph_files) == 0:
            raise RuntimeError(f"No files with extension '.{format}' found at {path}")
        if label_dir is None:
            raise RuntimeError(f"No file with extension '.label' found at {path}")

        # Order of files is important, should be consistent with .info, we suppose they are sorted
        graph_files = sorted(graph_files)

        # Create a temporary dir to store converted data
        with tmp_dir(path) as tmp:
            # Convert the data if necessary, write it to an empty directory
            if format != 'ij':
                from base.dataset_converter import DatasetConverter
                DatasetConverter.format_to_ij(gen_dataset.info, graph_files, format, tmp,
                                              default_node_attr_value, default_edge_attr_value)

            # Move or copy original contents to a temporary dir
            merge_directories(path, tmp, True)

            # Rename the newly created dir to the original one
            tmp.rename(path)

        # Check that data is valid
        gen_dataset.check_validity()

        return gen_dataset

    @staticmethod
    def _register_torch_geometric(
            dataset: Dataset,
            name: Union[str, None] = None,
            group: str = None,
            exists_ok: bool = False,
            copy_data: bool = False
    ) -> GeneralDataset:
        """
        Create GeneralDataset from an externally specified torch geometric dataset.

        :param dataset: torch_geometric.data.Dataset object.
        :param name: dataset name.
        :param group: group name, preferred options are: 'local', 'exported', etc.
        :param exists_ok: if True raise Exception if graph with same name exists, otherwise the data
         will be overwritten.
        :param copy_data: if True processed data will be copied, otherwise a symbolic link is
         created.
        :return: GeneralDataset
        """
        info = DatasetInfo.induce(dataset)
        if name is None:
            import time
            info.name = 'graph_' + str(time.time())
        else:
            assert isinstance(name, str) and os.sep not in name
            info.name = name

        # Define graph configuration
        dataset_config = DatasetConfig(
            domain="single-graph" if info.count == 1 else "multiple-graphs",
            group=group or 'exported', graph=info.name
        )

        # Check if exists
        root_dir, files_paths = Declare.dataset_root_dir(dataset_config)
        if root_dir.exists():
            if exists_ok:
                shutil.rmtree(root_dir)
            else:
                raise FileExistsError(
                    f"Graph with config {dataset_config.full_name()} already exists!")

        from base.ptg_datasets import PTGDataset
        gen_dataset = PTGDataset(dataset_config=dataset_config)
        info.save(gen_dataset.info_path)

        # Link or copy original contents to our path
        results_dir = gen_dataset.results_dir
        results_dir.parent.mkdir(parents=True, exist_ok=True)
        if copy_data:
            shutil.copytree(os.path.abspath(dataset.processed_dir), results_dir,
                            dirs_exist_ok=True)
        else:  # Create symlink
            # FIXME what will happen if we modify graph and its data.pt ?
            results_dir.symlink_to(os.path.abspath(dataset.processed_dir),
                                   target_is_directory=True)

        gen_dataset.dataset = dataset
        gen_dataset.info = info
        print(f"Registered graph '{info.name}' as {dataset_config.full_name()}")
        return gen_dataset


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


def is_in_torch_geometric_datasets(
        full_name: tuple = None
) -> bool:
    from aux.prefix_storage import PrefixStorage
    with open(TORCH_GEOM_GRAPHS_PATH, 'r') as f:
        return PrefixStorage.from_json(f.read()).check(full_name)
