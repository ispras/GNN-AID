import json
from collections import OrderedDict
from pathlib import Path
from typing import Union

from torch_geometric.data import Dataset, Data

from data_structures.configs import Task


class DatasetInfo:
    """
    Description for a dataset, its metainfo.
    It is created with the dataset and does not change.
    Some fields are obligate, others are not.

    Class Attributes
    ================

    - ``class_name`` (*str*)
      The name of the class used to initialize the object.

    - ``import_from`` (*str*)
      The module name from which to import the class.

    - ``name`` (*str*, optional)
      The dataset name. Defaults to the dataset path if not specified.

    - ``format`` (*str*, optional)
      The format of the raw data. Default is ``ij``.

    - ``count`` (*int*)
      The number of graphs.

    - ``directed`` (*bool*, optional)
      Flag indicating whether the edges are directed. Default is ``False``.

    - ``hetero`` (*bool*, optional)
      Flag indicating whether the graph is heterogeneous. Default is ``False``.

    - ``nodes`` (*List[int]*)
      List containing the number of nodes in each graph.

    - ``remap`` (*bool*, optional)
      Flag indicating whether to remap node indices. Default is ``False``.

    - ``node_attributes`` (*dict*, optional)
      Information about node attributes. The dictionary keys may include intermediate keys for node
      types (used in heterogeneous graphs).
      Contains the following fields:

      - ``names`` (*List[str]*)
        List of attribute names.
      - ``types`` (*List[str]*)
        List of attribute types, each being one of the following:

        - ``"continuous"`` — continuous numerical values
        - ``"categorical"`` — categorical values
        - ``"vector"`` — continuous vector values (e.g., as in PyTorch)
        - ``"other"`` — other types, such as strings
      - ``values`` (*List*)
        Possible values for the attributes, depending on their type:

        - For ``continuous`` — list containing minimum and maximum values
        - For ``categorical`` — list enumerating possible categories
        - For ``vector`` — list containing minimum and maximum values
        - For ``other`` — an empty list or some instructions on how to process the values

    - ``edge_attributes`` (*dict*, optional)
      Information about edge attributes, structured similarly to ``node_attributes``.

    - ``labelings`` (*dict*)
      Information about labelings. The dictionary keys may include intermediate keys for node types
      (in heterogeneous graphs). Each labeling entry maps a label name to a value, for example:

      - For classification — the number of classes
      - For regression on nodes — minimum and maximum values or 0
      - Other label types as applicable

    """

    def __init__(
            self
    ):
        self.class_name: str = None
        self.import_from: str = None

        self.name: str = None
        self.format: str = None
        self.count: int = None
        self.directed: bool = None
        self.hetero: bool = False
        self.nodes: list = None
        self.remap: bool = False
        self.node_attributes: OrderedDict = None
        self.edge_attributes: OrderedDict = None
        self.labelings: dict = None
        # self.node_info: dict = {}
        # self.edge_info: dict = {}
        # self.graph_info: dict = {}

    def check_validity(
            self
    ) -> None:
        """ Check existing fields have allowed values. """
        if self.format:
            from datasets.dataset_converter import DatasetConverter
            assert self.format == 'ij' or self.format in DatasetConverter.supported_formats
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
                assert isinstance(value, int) and value > 0, "Node feature size must be positive"
            elif type == "other":
                assert isinstance(value, int) or value in ["str", None]
        assert len(self.labelings) > 0
        if self.hetero:
            labelings = []
            for kv in self.labelings.values():
                labelings.extend(list(kv.items()))
        else:
            labelings = list(self.labelings.items())
        for t, lab in labelings:
            assert t in Task
            # assert Task.has_member(t)
            for name, info in lab.items():
                assert isinstance(name, str)
                if t.endswith("regression"):
                    assert isinstance(info, list) and len(info) == 2
                if t.endswith("classification"):
                    assert isinstance(info, int) and info > 1

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
        assert self.directed == is_graph_directed(dataset[0])
        assert self.remap is False
        if self.hetero:
            from torch_geometric.data import HeteroData
            assert isinstance(dataset[0], HeteroData)
            # TODO misha hetero
        else:
            assert len(self.node_attributes["names"]) == 1
            assert self.node_attributes["types"][0] == "vector"
        # TODO check features values range
        # TODO check labels classes and values range

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
    def read(
            path: Union[str, Path]
    ) -> 'DatasetInfo':
        """ Read info from a file. """
        with path.open('r') as f:
            a_dict = json.load(f, object_pairs_hook=OrderedDict)
        res = DatasetInfo()
        for k, v in a_dict.items():
            setattr(res, k, v)
        res.check()
        return res


def is_graph_directed(
        data: Data
) -> bool:
    """
    Detect whether graph is directed or not (for each edge i->j, exists j->i).
    NOTE: torch func data.is_undirected() does not work correctly,
    e.g. for TUDataset/MUTAG it incorrectly says directed.
    """

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
