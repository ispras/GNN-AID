import json
from copy import copy
from typing import Union, List, Tuple, Dict

from gnn_aid.aux.custom_decorators import timing_decorator
from gnn_aid.aux.utils import short_str, edge_index_to_edge_list
from gnn_aid.data_structures import Task, GraphModificationArtifact
from gnn_aid.datasets import GeneralDataset
from web_interface.back_front import json_dumps


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
        # self.edge_attributes: dict = None
        # self.graph_attributes: dict = None

    def __str__(
            self
    ):
        res = f"DatasetData[\n"
        res += f" edges: {short_str(self.edges)}\n"
        res += f" nodes: {short_str(self.nodes)}\n"
        res += f" graphs: {short_str(self.graphs)}\n"
        res += f" node_attributes: {short_str(self.node_attributes)}\n"
        # res += f" edge_attributes: {short_str(self.edge_attributes)}\n"
        # res += f" graph_attributes: {short_str(self.graph_attributes)}\n"
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
            # "graph_attributes": self.graph_attributes,
        }, **dump_args)


class DatasetVarData:
    """
    Container for a part of dataset var to transfer to frontend.
    Stores jsonable lists and dicts.
    """
    class Element(dict):
        """ Container for var data for node/edge/graph element.
        """
        def __init__(self, **kwargs):
            super().__init__()
            # self.labels: Union[Dict, List] = kwargs.get('labels')
            # self.features: Union[Dict, List] = kwargs.get('features')
            # self.logits: Union[Dict, List] = kwargs.get('logits')
            # self.predictions: Union[Dict, List] = kwargs.get('predictions')
            # self.answers: Union[Dict, List] = kwargs.get('answers')

        def __setattr__(self, key, value):
            super().__setattr__(key, value)
            # Need it to make it jsonable as a common dict
            self.__setitem__(key, value)

        def to_json(
                self,
                **dump_args
        ) -> str:
            """ Return json string. """
            return json.dumps(self.__dict__, **dump_args)

    def __init__(
            self
    ):
        self.node = DatasetVarData.Element()
        self.edge = DatasetVarData.Element()
        self.graph = DatasetVarData.Element()

    def __str__(
            self
    ):
        res = f"DatasetVarData[\n"
        res += f" node: {str(self.node)[:120]}\n"
        res += f" edge: {str(self.edge)[:120]}\n"
        res += f" graph: {str(self.graph)[:120]}\n"
        res += "]"
        return res

    def to_json(
            self,
            **dump_args
    ) -> str:
        """ Return json string. """
        return json.dumps(self.__dict__, **dump_args)


class DatasetDiffData:
    """
    Container for dataset diff (modification artifact) to transfer to frontend.
    """
    def __init__(
            self,
            artifact: GraphModificationArtifact
    ):
        self._artifact = artifact

    def __str__(
            self
    ):
        return f"DatasetDiffData{self._artifact.to_json()}"

    def to_json(
            self,
            **dump_args
    ) -> str:
        """ Return json string. """
        return json.dumps(self._artifact.to_json(), **dump_args)


class ViewPoint:
    """ Описание какая часть датасета просматривается на фронте
    """
    def __init__(
            self,
            center: Union[int, Tuple[int], None] = None,
            depth: Union[int, None] = None
    ) -> None:
        """
        :param center: central node/edge/graph. None means consider all.
        :param depth: neighborhood depth or number of graphs before and after center to take,
         e.g. center=7, depth=2 will give 5,6,7,8,9 graphs. None means consider all
        """
        self.center = center
        self.depth = depth

    def __str__(self):
        return f"center={self.center}, depth={self.depth}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, ViewPoint):
            return False
        if other.center != self.center:
            return False
        if other.depth != self.depth:
            return False
        return True


class DatasetIndex:
    """ Mask of a part of the dataset visible at frontend
    """
    def __init__(
            self,
            view_point: ViewPoint,
            gen_dataset: GeneralDataset,
    ) -> None:
        """
        Build mask on a dataset.

        :param view_point: какая часть датасета просматривается на фронте
        :param gen_dataset: used but not stored
        :return:
        """
        #             neigh   1 graph  multi
        # node_index   [[n]]     -      -
        # edge_index   [[e]]     -      -
        # graph_index    -       -     [g]
        self.view_point = view_point

        self.node_index = None  # nodes to include to the result
        self.edge_index = None  # edges to include to the result
        self.graph_index = None  # graph ids to include to the result

        center = view_point.center
        depth = view_point.depth

        print("Building DatasetIndex", view_point)

        if gen_dataset.is_multi():
            if center is not None:  # Get several graphs
                if isinstance(center, list):
                    self.graph_index = center
                else:
                    if depth is None:
                        depth = 3
                    self.graph_index = list(range(
                        max(0, center - depth),
                        min(gen_dataset.info.count, center + depth + 1)))
            else:  # Get all graphs
                self.graph_index = range(gen_dataset.info.count)

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

                else:  # homo
                    nodes = {0: {center}}  # {depth: set of ids}
                    edges = {0: []}  # incoming edges
                    prev_nodes = set()  # Nodes in neighborhood Up to depth=d-1

                    # We need iterate all edges even for undirected graph
                    all_edges = edge_index_to_edge_list(gen_dataset.edges[0], True)
                    for d in range(1, depth + 1):
                        ns = nodes[d - 1]
                        es_next = set()
                        ns_next = set()
                        for ix, (i, j) in enumerate(all_edges):
                            if not gen_dataset.info.directed and i > j:
                                continue

                            # Get all incoming edges * -> j
                            if j in ns and i not in prev_nodes:
                                es_next.add(ix)
                                # es_next.append((i, j))
                                if i not in ns:
                                    ns_next.add(i)

                            if not gen_dataset.info.directed:
                                # Check also outcoming edge i -> *, excluding already added
                                if i in ns and j not in prev_nodes:
                                    es_next.add(ix)
                                    # es_next.append((i, j))
                                    if j not in ns:
                                        ns_next.add(j)

                        prev_nodes.update(ns)
                        nodes[d] = ns_next
                        edges[d] = list(sorted(es_next))

                    self.node_index = [list(ns) for ns in nodes.values()]
                    self.edge_index = [list(es) for es in edges.values()]

            else:  # Get whole graph
                pass

    def __str__(self):
        return str(self.__dict__)


class VisiblePart:
    """
    Датасет + индекс видимой части. Отвечает за выборку видимой части от датасета для отправки на
    отрисовку.
    """
    def __init__(
            self,
            view_point: ViewPoint,
            gen_dataset: GeneralDataset,
    ):
        """ Compute a part of dataset specified by a central node/edge/graph and a depth.

        :param view_point: какая часть датасета просматривается на фронте
        :param gen_dataset: GeneralDataset, will be stored
        """
        #         neigh   graph   graphs
        # nodes   [[n]]     n      [n]
        # graphs    -       -      [g]
        # edges   [[e]]   [[e]]   [[e]]
        self.gen_dataset: GeneralDataset = gen_dataset

        self.dataset_index = DatasetIndex(view_point, gen_dataset)

    def update_view_point(
            self,
            view_point: ViewPoint
    ):
        """ If view point is different, update DatasetIndex.
        """
        if view_point is not None and view_point != self.dataset_index.view_point:
            self.dataset_index = DatasetIndex(view_point, self.gen_dataset)

    def iterate(self, elem: str, pairs_for_edges=False):
        """ Iterate over indexed elements: nodes/edges/graphs
        """
        assert elem in ['nodes', 'edges', 'graphs']

        if self.gen_dataset.is_multi():
            if elem == 'graphs':
                for g in self.dataset_index.graph_index:
                    yield g
            else:
                raise RuntimeError

        else:  # single
            assert self.gen_dataset.info.count == 1
            if self.dataset_index.view_point.center is not None:  # Get neighborhood
                if elem == 'nodes':
                    for ns in self.dataset_index.node_index:
                        for n in ns:
                            yield n

                elif elem == 'edges':
                    if pairs_for_edges:
                        edges = edge_index_to_edge_list(self.gen_dataset.edges[0], True)
                    for es in self.dataset_index.edge_index:
                        for e in es:
                            if pairs_for_edges:
                                yield edges[e]
                            else:
                                yield e

                else:
                    raise RuntimeError

            else:  # Get whole graph
                if elem == 'nodes':
                    for n in range(self.gen_dataset.num_nodes):
                        yield n

                elif elem == 'edges':
                    for ix, e in enumerate(edge_index_to_edge_list(
                            self.gen_dataset.edges[0], self.gen_dataset.is_directed())):
                        if pairs_for_edges:
                            yield tuple(e)
                        else:
                            yield ix
                else:
                    raise RuntimeError

    def filter(
            self,
            array: Union[list, dict],
            elem: str = None
    ) -> list:
        """ Suppose ixes = [2,4]: [a, b, c, d] ->  {2: b, 4: d}
        """
        if elem is None:
            # FIXME can it differ from dataset's?
            task = self.gen_dataset.dataset_var_config.task
            if task.is_node_level():
                elem = 'nodes'
            elif task.is_edge_level():
                elem = 'edges'
            elif task.is_graph_level():
                elem = 'graphs'
            else:
                raise ValueError(f"Unsupported task {task}")

        if elem == 'edges':
            assert isinstance(array, dict),\
                f"For edge filtering, array must be dict, not {type(array)}"

        return [array[ix] for ix in self.iterate(elem, pairs_for_edges=True)]

    # @timing_decorator
    def get_dataset_data(
            self,
            view_point: ViewPoint = None
    ) -> DatasetData:
        """ Get DatasetData for specified graphs or nodes. Tensors are converted to python lists.
        """
        self.update_view_point(view_point)

        print("Building DatasetData", self.dataset_index.view_point)
        dataset_data = DatasetData()

        # Add structure
        graphs = None
        center = self.dataset_index.view_point.center
        if self.gen_dataset.is_multi():
            nodes = [self.gen_dataset.num_nodes[ix] for ix in self.dataset_index.graph_index]
            ptg_edges = self.gen_dataset.edges
            edges = [edge_index_to_edge_list(
                ptg_edges[ix], self.gen_dataset.is_directed()) for ix in self.dataset_index.graph_index]
            graphs = list(self.dataset_index.graph_index)

        else:  # single
            assert self.gen_dataset.info.count == 1
            if center is not None:  # Get neighborhood
                nodes = copy(self.dataset_index.node_index)
                # Edge index is over full ptg edge_index, both directions
                all_edges = edge_index_to_edge_list(self.gen_dataset.edges[0], True)
                edges = [
                    [all_edges[i] for i in es] for es in self.dataset_index.edge_index
                ]
                # edges = copy(self.dataset_index.edge_index)

            else:  # Get whole graph
                nodes = self.gen_dataset.num_nodes
                edges = edge_index_to_edge_list(self.gen_dataset.edges[0], self.gen_dataset.is_directed())

        dataset_data.nodes = nodes
        dataset_data.edges = edges
        dataset_data.graphs = graphs

        # Add attributes
        dataset_data.node_attributes = {}

        node_attributes = self.gen_dataset.node_attributes()
        if self.gen_dataset.is_multi():
            for a, vals_list in node_attributes.items():
                dataset_data.node_attributes[a] = {ix: vals_list[ix] for ix in graphs}

        else:
            if center is not None:  # neighborhood
                for a, vals_list in node_attributes.items():
                    dataset_data.node_attributes[a] = [{
                        n: (vals_list[0].get(n, None) if n in vals_list[0] else None) for n in self.iterate('nodes')}]

            else:  # whole graph
                for a, vals_list in node_attributes.items():
                    dataset_data.node_attributes[a] = [{
                        n: vals_list[0].get(n, None) for n in range(nodes)}]

        return dataset_data

    # @timing_decorator
    def get_dataset_var_data(
            self,
            view_point: ViewPoint = None,
            satellites: Union[list, None] = None
    ) -> DatasetVarData:
        """
        Get DatasetVarData for specified graphs or nodes. Tensors are converted to python lists.
        """
        self.update_view_point(view_point)

        dataset_var_data = DatasetVarData()
        print("Computing dataset_var_data for", view_point)

        # TODO use satellites
        if satellites:
            raise NotImplementedError

        # Node level
        dataset_var_data.node.features = {}
        node_features = self.gen_dataset.node_features
        if self.gen_dataset.is_multi():
            dataset_var_data.node.features = [
                node_features[g].tolist() for g in self.iterate('graphs')
            ]
        else:
            dataset_var_data.node.features = [
                node_features[n].tolist() for n in self.iterate('nodes')
            ]

        # Edge level
        edge_features = self.gen_dataset.edge_features
        if edge_features is not None and edge_features[0] is not None:
            dataset_var_data.edge.features = {}
            if self.gen_dataset.is_multi():
                dataset_var_data.edge.features = [
                    edge_features[g].tolist() for g in self.iterate('graphs')
                ]
            else:
                dataset_var_data.edge.features = [
                    edge_features[e].tolist() for e in self.iterate('edges')
                ]

        # Labels according to task
        if self.gen_dataset.labels is not None:
            task = self.gen_dataset.dataset_var_config.task
            if task.is_node_level():
                elem = dataset_var_data.node
                iterated = list(self.iterate('nodes'))
                labels = self.gen_dataset.labels.tolist()
            elif task.is_edge_level():
                elem = dataset_var_data.edge
                iterated = list(self.iterate('edges', pairs_for_edges=True))
                labels = dict(zip(zip(*self.gen_dataset.edge_label_index.tolist()),
                                  self.gen_dataset.labels.tolist()))
            elif task.is_graph_level():
                elem = dataset_var_data.graph
                iterated = list(self.iterate('graphs'))
                labels = self.gen_dataset.labels.tolist()
            else:
                raise ValueError(f"Unsupported task type {task}")

            elem.labels = [labels[ix] for ix in iterated]

        return dataset_var_data

    def get_train_test_mask(
            self,
    ) -> DatasetVarData:
        """ Get train/val/test mask for the dataset and send to frontend.
        """
        dataset_var_data = DatasetVarData()

        task = self.gen_dataset.dataset_var_config.task
        if task.is_node_level():
            elem = dataset_var_data.node
            iterated = list(self.iterate('nodes'))
        elif task.is_edge_level():
            elem = dataset_var_data.edge
            iterated = list(self.iterate('edges', pairs_for_edges=True))
        elif task.is_graph_level():
            elem = dataset_var_data.graph
            iterated = list(self.iterate('graphs'))
        else:
            raise ValueError(f"Unsupported task type {task}")

        if task.is_edge_level():
            # Build sets of edges for each mask
            mask_edges = {}
            for m, mask in zip([1, 3, 2],
                            [self.gen_dataset.train_mask, self.gen_dataset.val_mask, self.gen_dataset.test_mask]):
                mask_edges[m] = set(zip(*self.gen_dataset.edge_label_index[:, mask].tolist()))

            res = [0] * len(iterated)
            for ix, e in enumerate(iterated):
                for m, edges in mask_edges.items():
                    if e in edges:
                        res[ix] = m
                        continue

        else:
            # Encode mask as train=1, val=2, test=3
            res = []
            for n in iterated:
                if self.gen_dataset.train_mask[n]:
                    res.append(1)
                elif self.gen_dataset.test_mask[n]:
                    res.append(2)
                elif self.gen_dataset.val_mask[n]:
                    res.append(3)

        elem["train-test-mask"] = res
        # add_into_dvd(self.gen_dataset, {"train": res}, dataset_var_data)

        return dataset_var_data


def add_into_dvd(
        gen_dataset: GeneralDataset,
        a_dict: dict,
        dvd: DatasetVarData = None
) -> DatasetVarData:
    dvd = dvd or DatasetVarData()

    task = gen_dataset.dataset_var_config.task
    if task.is_node_level():
        elem = dvd.node
    elif task.is_edge_level():
        elem = dvd.edge
    elif task.is_graph_level():
        elem = dvd.graph
    else:
        raise RuntimeError

    elem.update(**a_dict)
    return dvd


if __name__ == '__main__':
    from gnn_aid.data_structures.configs import DatasetConfig, DatasetVarConfig, FeatureConfig, \
        Task
    from gnn_aid.datasets.datasets_manager import DatasetManager
    from gnn_aid.datasets.ptg_datasets import LibPTGDataset

    # dc = DatasetConfig(('example', 'example'))
    # dvc = DatasetVarConfig(features=FeatureConfig(node_attr=['a'], edge_attr=['weight']),
    #                        task=Task.EDGE_PREDICTION, dataset_ver_ind=0)

    # dc = DatasetConfig(('example', 'example3'))
    # dvc = DatasetVarConfig(features=FeatureConfig(node_attr=['type'], edge_attr=[]),
    #                        task=Task.GRAPH_CLASSIFICATION, labeling='binary', dataset_ver_ind=0)

    # dc = DatasetConfig(('example', 'custom', 'ba_random'))
    # dvc = DatasetVarConfig(features=FeatureConfig(node_struct=[FeatureConfig.ten_ones]),
    #                        task=Task.EDGE_PREDICTION, dataset_ver_ind=0)

    dc = DatasetConfig((LibPTGDataset.data_folder, 'Homogeneous', 'Planetoid', 'Cora'))
    dvc = LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.NODE_CLASSIFICATION})
    # dvc = LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.EDGE_PREDICTION})

    # dc = DatasetConfig(('example', 'example'))
    # dvc = DatasetVarConfig(task=Task.EDGE_REGRESSION, labeling="regression",
    #                        features=FeatureConfig(node_attr=['a'], edge_attr=['weight']), dataset_ver_ind=0)

    gen_dataset = DatasetManager.get_by_config(dc, dvc)
    # gen_dataset.set_visible_part({})
    # visible_part = VisiblePart(ViewPoint(), gen_dataset)
    visible_part = VisiblePart(ViewPoint(**{'center': 2, 'depth': 2}), gen_dataset)
    dd = visible_part.get_dataset_data()
    # print(visible_part.dataset_index)
    # print(dd)
    print(json_dumps(dd, indent=1))
    gen_dataset.train_test_split(percent_train_class=0.6, percent_test_class=0.4)

    dvd = visible_part.get_dataset_var_data()
    # print(dvd.to_json(indent=1))
    # print(json_dumps({"node": dvd.node}, indent=1))

    # labels = dict(zip(zip(*gen_dataset.edge_label_index.tolist()),
    #                   gen_dataset.labels.tolist()))
    print(visible_part.get_train_test_mask())
