import json
from collections import OrderedDict
from pathlib import Path
from typing import Union

from torch_geometric.data import Dataset, HeteroData, Data
from torch_geometric.data.data import BaseData


class DatasetInfo:
    """
    Description for a dataset, its metainfo.
    It is created with the dataset and does not change.
    Some fields are obligate, others are not.
    """

    def __init__(
            self
    ):
        self.class_name: str = None
        self.import_from: str = None

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
        # self.node_info: dict = {}
        # self.edge_info: dict = {}
        # self.graph_info: dict = {}

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
        # FIXME misha move to LocalPTGDataset
        res = DatasetInfo()
        res.count = len(dataset)
        # from datasets.ptg_datasets import is_graph_directed
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
    ) -> 'DatasetInfo':
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