import json
from pathlib import Path
from typing import Union, Any


class Explanation:
    """ General class to represent GNN explanation.
    """

    def __init__(
            self,
            local: bool,
            type: str,
            data: str = None,
            meta=None
    ):
        """
        :param local: True if local, False if global
        :param type: "subgraph", "prototype", etc
        :param data: explanation contents
        :param meta: additional info about explanation
        """
        self.dictionary = {'info': {}, 'data': {}}
        self.dictionary['info']['local'] = local
        self.dictionary['info']['type'] = type
        if data is not None:
            self.dictionary['data'] = data
        if meta is not None:
            self.dictionary['info']['meta'] = meta

    def save(
            self,
            path: Union[str, Path]
    ) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.dictionary, f, ensure_ascii=False, indent=4)


class AttributionExplanation(
    Explanation
):
    """
    Attribution explanation as important subgraph.
    Importance scores (binary or continual) can be assigned to nodes, edges, and features.
    """

    def __init__(
            self,
            local: bool = True,
            directed: bool = False,
            nodes: str = "binary",
            edges: bool = False,
            features: bool = False
    ):
        """
        :param local: True if local, False if global
        :param nodes: "binary", "continuous", or None/False
        :param edges: "binary", "continuous", or None/False
        :param features: "binary", "continuous", or None/False
        """
        meta = {
            "nodes": nodes or "none", "edges": edges or "none", "features": features or "none"}
        super(AttributionExplanation, self).__init__(local=local, type="subgraph", meta=meta)
        self.dictionary['info']['directed'] = directed

    def add_edges(
            self,
            edge_data: dict
    ) -> None:
        self.dictionary['data']['edges'] = edge_data

    def add_features(
            self,
            feature_data: dict
    ) -> None:
        self.dictionary['data']['features'] = feature_data

    def add_nodes(
            self,
            node_data: dict
    ) -> None:
        self.dictionary['data']['nodes'] = node_data


class ConceptExplanationGlobal(
    Explanation
):
    def __init__(
            self,
            raw_neurons: list,
            n_neurons: Any
    ):
        Explanation.__init__(self, False, 'string')
        self.dictionary['data']['neurons'] = {}
        for n in range(n_neurons):
            if n not in raw_neurons[0].keys():
                pass
            else:
                self.dictionary['data']['neurons'][n] = {'rule': raw_neurons[0][n][1][2],
                                                         'score': raw_neurons[0][n][1][0],
                                                         'importances': raw_neurons[1][n]}
