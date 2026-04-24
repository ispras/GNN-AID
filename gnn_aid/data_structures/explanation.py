import json
from pathlib import Path
from typing import Union, Any


class Explanation:
    """
    General class to represent a GNN explanation.
    """

    def __init__(
            self,
            local: bool,
            type: str,
            data: str = None,
            meta=None
    ):
        """
        Args:
            local (bool): True if local explanation, False if global.
            type (str): Explanation type, e.g. ``"subgraph"``, ``"prototype"``.
            data (str): Explanation contents. Default value: `None`.
            meta: Additional info about the explanation. Default value: `None`.
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
    Attribution explanation as an important subgraph.

    Importance scores (binary or continuous) can be assigned to nodes, edges, and features.
    """

    def __init__(
            self,
            local: bool = True,
            directed: bool = False,
            nodes: Union[str, bool, None] = "binary",
            edges: Union[str, bool] = False,
            features: Union[str, bool] = False
    ):
        """
        Args:
            local (bool): True if local explanation, False if global. Default value: `True`.
            directed (bool): Whether the explanation graph is directed. Default value: `False`.
            nodes (Union[str, bool, None]): Node importance mode: ``"binary"``, ``"continuous"``,
                or ``None``/``False`` to disable. Default value: `"binary"`.
            edges (Union[str, bool]): Edge importance mode: ``"binary"``, ``"continuous"``,
                or ``False`` to disable. Default value: `False`.
            features (Union[str, bool]): Feature importance mode: ``"binary"``, ``"continuous"``,
                or ``False`` to disable. Default value: `False`.
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
    """
    Global concept-based explanation that maps neurons to their associated rules and importance scores.
    """

    def __init__(
            self,
            raw_neurons: list,
            n_neurons: int
    ):
        """
        Args:
            raw_neurons (list): Raw neuron data; raw_neurons[0] maps neuron index to
                (rule_info, ...) and raw_neurons[1] maps neuron index to importance scores.
            n_neurons (int): Total number of neurons to process.
        """
        Explanation.__init__(self, False, 'string')
        self.dictionary['data']['neurons'] = {}
        for n in range(n_neurons):
            if n not in raw_neurons[0].keys():
                pass
            else:
                self.dictionary['data']['neurons'][n] = {'rule': raw_neurons[0][n][1][2],
                                                         'score': raw_neurons[0][n][1][0],
                                                         'importances': raw_neurons[1][n]}
