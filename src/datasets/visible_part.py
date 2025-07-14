import json
from typing import Union, List, Tuple

from torch import tensor, cat
from torch_geometric.data.dataset import _get_flattened_data_list

from aux.custom_decorators import timing_decorator
from aux.utils import short_str


class DatasetData:
    """
    Container for a part of dataset to transfer to frontend.
    Stores jsonable lists and dicts.
    """
    def __init__(
            self
    ):
        self.edges: list = None  # list of edge lists for each graph
        self.nodes: list = None  # list of nodes
        self.graphs: list = None  # list of graph indices
        self.node_attributes: dict = None  # dict {name -> list of attr dict for each graph}
        # self.edge_attributes: dict = None  # dict {name -> list of attr dict for each graph}

    def __str__(
            self
    ):
        res = f"DatasetData[\n"
        res += f" edges: {short_str(self.edges)}\n"
        res += f" nodes: {short_str(self.nodes)}\n"
        res += f" graphs: {short_str(self.graphs)}\n"
        res += f" node_attributes: {short_str(self.node_attributes)}\n"
        # res += f" edge_attributes: {short_str(self.edge_attributes)}\n"
        res += "]"
        return res

    def to_json(
            self,
            **dump_args
    ) -> str:
        """ Return json string. """
        return json.dumps({
            "edges": self.edges,
            "nodes": self.nodes,
            "graphs": self.graphs,
            "node_attributes": self.node_attributes,
            # "edge_attributes": self.edge_attributes,
        }, **dump_args)


class DatasetVarData:
    """
    Container for a part of dataset var to transfer to frontend.
    Stores jsonable lists and dicts.
    """
    def __init__(
            self
    ):
        self.labels: dict = None  # list of labels over graphs or nodes
        self.node_features: dict = None  # dict {name -> list of attr dict for each graph}
        # self.edge_features: dict = None  # dict {name -> list of attr dict for each graph}

    def __str__(
            self
    ):
        res = f"DatasetVarData[\n"
        res += f" labels: {short_str(self.labels)}\n"
        res += f" node_features: {short_str(self.node_features)}\n"
        # res += f" edge_features: {short_str(self.edge_features)}\n"
        res += "]"
        return res

    def to_json(
            self,
            **dump_args
    ) -> str:
        """ Return json string. """
        return json.dumps({
            "labels": self.labels,
            "node_features": self.node_features,
            # "edge_features": self.edge_features,
        }, **dump_args)


class DatasetIndex:
    """
    """
    def __init__(
            self,
            gen_dataset: 'GeneralDataset',
            center: Union[int, List[int], Tuple[int]] = None,
            depth: int = None
    ) -> None:
        """
        Build index on a dataset.
        :param gen_dataset:
        :param center: central node/graph or a list of nodes/graphs
        :param depth: neighborhood depth or number of graphs before and after center to take,
         e.g. center=7, depth=2 will give 5,6,7,8,9 graphs
        :return:
        """
        # nodes   [[n]]     n      [n]
        # graphs    -       -      [g]
        # edges   [[e]]   [[e]]   [[e]]
        self.graphs = None
        self.nodes = None
        self.edges = None

        self.ixes = None  # node or graph ids to include to the result

        if gen_dataset.is_multi():
            if center is not None:  # Get several graphs
                if isinstance(center, list):
                    self.ixes = center
                else:
                    if depth is None:
                        depth = 3
                    self.ixes = range(
                        max(0, center - depth),
                        min(gen_dataset.info.count, center + depth + 1))
            else:  # Get all graphs
                self.ixes = range(gen_dataset.info.count)

            self.graphs = list(self.ixes)
            self.nodes = [gen_dataset.info.nodes[ix] for ix in self.ixes]
            edges = gen_dataset.edges
            self.edges = [edge_index_to_edge_list(
                edges[ix], gen_dataset.is_directed) for ix in self.ixes]

        else:  # single
            assert gen_dataset.info.count == 1

            if center is not None:  # Get neighborhood
                if isinstance(center, list):
                    raise NotImplementedError
                if depth is None:
                    depth = 2

                if gen_dataset.info.hetero:
                    # TODO misha hetero
                    raise NotImplementedError

                    nodes = {0: {center[0]: {center[1]}}}  # {depth: {type: set of ids}}
                    edges = {0: {}}  # incoming edges, {depth: {type: list}}
                    prev_nodes = set()  # Nodes in neighborhood Up to depth=d-1

                    all_edges = gen_dataset.edges[0]
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
                    self.ixes = [n for ns in self.nodes for n in ns]

                else:  # homo
                    nodes = {0: {center}}  # {depth: set of ids}
                    edges = {0: []}  # incoming edges
                    prev_nodes = set()  # Nodes in neighborhood Up to depth=d-1

                    all_edges = edge_index_to_edge_list(
                        gen_dataset.edges[0], gen_dataset.is_directed())
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
                    self.ixes = [n for ns in self.nodes for n in ns]

            else:  # Get whole graph
                self.edges = edge_index_to_edge_list(
                    gen_dataset.edges, gen_dataset.is_directed)
                self.nodes = gen_dataset.info.nodes[0]
                if gen_dataset.info.hetero:
                    self.ixes = {nt: list(range(self.nodes[nt])) for nt in gen_dataset.node_types}
                else:
                    self.ixes = list(range(self.nodes))

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


class VisiblePart:
    def __init__(
            self,
            gen_dataset: 'GeneralDataset',
            center: Union[int, List[int], Tuple[int]] = None,
            depth: int = None
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
        self.gen_dataset: 'GeneralDataset' = gen_dataset

        self.index = DatasetIndex(gen_dataset, center, depth)

    def ixes(
            self
    ) -> list:
        # if isinstance(self._ixes, list):
        #     for i in self._ixes:
        #         yield i
        # elif isinstance(self._ixes, dict):
        #     for nt, ixes in self._ixes.items():
        #         for i in ixes:
        #             yield nt, i
        return self.index.ixes

    def filter(
            self,
            array
    ) -> dict:
        """ Suppose ixes = [2,4]: [a, b, c, d] ->  {2: b, 4: d}
        """
        return {ix: array[ix] for ix in self.ixes()}

    @timing_decorator
    def get_dataset_data(
            self,
            part: Union[dict, None] = None
    ) -> DatasetData:
        """ Get DatasetData for specified graphs or nodes. Tensors are converted to python lists.
        """
        if part is None:
            index = self.index
        else:
            index = DatasetIndex(self.gen_dataset, **part)

        dataset_data = DatasetData()
        # FIXME misha - we want not tensors
        dataset_data.node_attributes = {}

        # Add structure
        dataset_data.nodes = index.nodes
        dataset_data.edges = index.edges
        dataset_data.graphs = index.graphs

        # Extract attributes
        ixes = index.ixes
        node_attributes = self.gen_dataset.node_attributes()
        if self.gen_dataset.is_multi():
            for a, vals_list in node_attributes.items():
                dataset_data.node_attributes[a] = {ix: vals_list[ix] for ix in ixes}

        else:
            if isinstance(index.nodes, list):  # neighborhood
                for a, vals_list in node_attributes.items():
                    dataset_data.node_attributes[a] = [{
                        n: (vals_list[0][n] if n in vals_list[0] else None) for n in ixes}]

            else:  # whole graph
                for a, vals_list in node_attributes.items():
                    dataset_data.node_attributes[a] = [{
                        n: vals_list[0][n] for n in ixes}]

        return dataset_data

    @timing_decorator
    def get_dataset_var_data(
            self,
            part: Union[dict, None] = None
    ) -> DatasetVarData:
        """ Get DatasetVarData for specified graphs or nodes. Tensors are converted to python lists.
        """
        if part is None:
            index = self.index
        else:
            index = DatasetIndex(self.gen_dataset, **part)

        dataset_var_data = DatasetVarData()
        dataset_var_data.node_features = {}
        dataset_var_data.labels = {}

        node_features = self.gen_dataset.node_features
        labels = self.gen_dataset.labels.tolist()

        for ix in index.ixes:
            dataset_var_data.node_features[ix] = node_features[ix].tolist()
            dataset_var_data.labels[ix] = labels[ix]

        return dataset_var_data


def edge_index_to_edge_list(
        edge_index: Union[list, tensor],
        directed: bool = True
) -> list:
    if isinstance(edge_index, list):
        return [edge_index_to_edge_list(x) for x in edge_index]

    assert edge_index.shape[0] == 2
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    if directed:
        return edges
    else:
        # Удалим дубликаты: (i, j) и (j, i) → оставить только (min(i, j), max(i, j))
        edge_set = set()
        for i, j in edges:
            edge_set.add(tuple(sorted((i, j))))
        return list(edge_set)
