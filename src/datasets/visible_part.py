from typing import Union


class VisiblePart:
    def __init__(
            self,
            gen_dataset: 'GeneralDataset',
            center: [int, list, tuple] = None,
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

        self._ixes = None  # node or graph ids to include to the result

        self.gen_dataset = gen_dataset
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

                if gen_dataset.info.hetero:
                    # TODO misha hetero
                    raise NotImplementedError

                    nodes = {0: {center[0]: {center[1]}}}  # {depth: {type: set of ids}}
                    edges = {0: {}}  # incoming edges, {depth: {type: list}}
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

                else:  # homo
                    nodes = {0: {center}}  # {depth: set of ids}
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
                if gen_dataset.info.hetero:
                    self._ixes = {nt: list(range(self.nodes[nt])) for nt in gen_dataset.node_types}
                else:
                    self._ixes = list(range(self.nodes))

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

    # def get_dataset_data(
    #         self,
    #         part: Union[dict, None] = None
    # ) -> dict:
    #     """ Get DatasetData for specified graphs or nodes
    #     """
    #     edges_list = []
    #     node_attributes = {}
    #     res = {
    #         'edges': edges_list,
    #         'node_attributes': node_attributes,
    #     }
    #
    #     visible_part = self.visible_part if part is None else VisiblePart(self, **part)
    #
    #     res.update(visible_part.as_dict())
    #
    #     # Get needed part of self.dataset_data
    #     ixes = visible_part.ixes()
    #     if self..is_multi():
    #         for a, vals_list in self.dataset_data['node_attributes'].items():
    #             node_attributes[a] = {ix: vals_list[ix] for ix in ixes}
    #
    #     else:
    #         if isinstance(visible_part.nodes, list):  # neighborhood
    #             for a, vals_list in self.dataset_data['node_attributes'].items():
    #                 node_attributes[a] = [{
    #                     n: (vals_list[0][n] if n in vals_list[0] else None) for n in ixes}]
    #
    #         else:  # whole graph
    #             res['node_attributes'] = self.dataset_data['node_attributes']
    #
    #     return res
    #
    # def get_dataset_var_data(
    #         self,
    #         part: Union[dict, None] = None
    # ) -> dict:
    #     """ Get DatasetVarData for specified graphs or nodes
    #     """
    #     if self.dataset_var_data is None:
    #         self._compute_dataset_var_data()
    #
    #     # Get needed part of self.dataset_var_data
    #     features = {}
    #     labels = {}
    #     dataset_var_data = {
    #         "features": features,
    #         "labels": labels,
    #     }
    #
    #     visible_part = self.visible_part if part is None else VisiblePart(self, **part)
    #
    #     for ix in visible_part.ixes():
    #         # TODO IMP misha replace with getting data from tensors instead of keeping the whole data
    #         features[ix] = self.dataset_var_data['node_features'][ix]
    #         labels[ix] = self.dataset_var_data['labels'][ix]
    #
    #     return dataset_var_data
    #
