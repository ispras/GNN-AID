import json
import torch
from pathlib import Path
from typing import Optional, Union, Dict, List


class GraphModificationArtifact:
    """
    Stores and manages graph modifications due to poisoning and evasion attacks,
    as well as corresponding defenses. Supports serialization to and from JSON format.
    """

    def __init__(self) -> None:
        """ Initializes internal structures for node and edge modifications.
        """
        self.nodes: Dict = {
            "remove": [],
            "add": {},
            "change_f": {}
        }
        self.edges: Dict = {
            "remove": [],
            "add": []
        }

    @staticmethod
    def to_scalar_str(
            idx: Union[str, int, torch.Tensor]
    ) -> str:
        """
        Converts an index to string. Handles int, str, and scalar torch.Tensor.

        Args:
            idx: Index to convert.
        Returns:
            String representation of the index.
        """
        if isinstance(idx, torch.Tensor):
            assert idx.numel() == 1, "Only scalar tensors allowed as indices"
            return str(idx.item())
        return str(idx)

    def add_node(
            self,
            node_id: Union[str, int, torch.Tensor],
            feature_tensor: torch.Tensor
    ) -> None:
        """
        Adds a new node with associated features.
        Args:
            node_id: Identifier of the node to add.
            feature_tensor: Feature tensor for the node.
        """
        assert isinstance(feature_tensor, torch.Tensor), "feature_tensor must be a torch.Tensor"
        self.nodes["add"][self.to_scalar_str(node_id)] = feature_tensor

    def add_nodes(
            self,
            node_dict: Dict[Union[str, int, torch.Tensor], torch.Tensor]
    ) -> None:
        """ Adds multiple nodes from a dictionary mapping node IDs to feature tensors.
        """
        for node_id, features in node_dict.items():
            self.add_node(node_id, features)

    def remove_node(
            self,
            node_id: Union[str, int, torch.Tensor]
    ) -> None:
        """
        Marks a node for removal.
        Args:
            node_id: Identifier of the node to remove.
        """
        self.nodes["remove"].append(self.to_scalar_str(node_id))

    def remove_nodes(
            self,
            node_ids: List[Union[str, int, torch.Tensor]]
    ) -> None:
        """ Marks multiple nodes for removal.
        """
        for node_id in node_ids:
            self.remove_node(node_id)

    def change_node_feature(
            self,
            node_id: Union[str, int, torch.Tensor],
            feature_index: Union[int, torch.Tensor],
            new_value: float
    ) -> None:
        """
        Registers a new value for a specific feature of a node.
        Args:
            node_id: Node identifier.
            feature_index: Index of the feature to change.
            new_value: New value of the feature.
        """
        node_key = self.to_scalar_str(node_id)
        feature_key = self.to_scalar_str(feature_index)
        if node_key not in self.nodes["change_f"]:
            self.nodes["change_f"][node_key] = {}
        self.nodes["change_f"][node_key][feature_key] = new_value

    def change_node_features(
            self,
            changes: Dict[
                Union[str, int, torch.Tensor],
                Dict[Union[int, torch.Tensor], float]
            ]
    ) -> None:
        """ Registers multiple feature changes across multiple nodes.
        """
        for node_id, feature_changes in changes.items():
            for feat_idx, value in feature_changes.items():
                self.change_node_feature(node_id, feat_idx, value)

    def set_nodes(
            self,
            remove_nodes: List[Union[str, int, torch.Tensor]],
            add_nodes: Dict[Union[str, int, torch.Tensor], torch.Tensor],
            changed_features: Dict[
                Union[str, int, torch.Tensor],
                Dict[Union[int, torch.Tensor], float]
            ]
    ) -> None:
        """ Sets all node modification fields at once.
        Args:
            remove_nodes: List of node IDs to remove.
            add_nodes: Dictionary mapping node IDs to feature tensors.
            changed_features: Mapping of node IDs to {feature_index: new_value}.
        """
        self.remove_nodes(remove_nodes)
        self.add_nodes(add_nodes)
        self.change_node_features(changed_features)

    def add_edge(
            self,
            from_node_id: Union[str, int, torch.Tensor],
            to_node_id: Union[str, int, torch.Tensor],
            edge_attr_tensor: Optional[torch.Tensor] = None
    ) -> None:
        """
        Adds a new edge with optional edge attributes.
        Args:
            from_node_id: Source node of the edge.
            to_node_id: Target node of the edge.
            edge_attr_tensor: Optional tensor of edge attributes.
        """
        if edge_attr_tensor is not None:
            assert isinstance(edge_attr_tensor, torch.Tensor), (
                "edge_attr_tensor must be a torch.Tensor or None"
            )
        self.edges["add"].append([
            self.to_scalar_str(from_node_id),
            self.to_scalar_str(to_node_id),
            edge_attr_tensor
        ])

    def add_edges(
            self,
            edge_list: List[List[Union[str, int, torch.Tensor, Optional[torch.Tensor]]]]
    ) -> None:
        """ Adds multiple edges, each given as [from_node_id, to_node_id, edge_attr or None].
        """
        for edge in edge_list:
            self.add_edge(*edge)

    def remove_edge(
            self,
            from_node_id: Union[str, int, torch.Tensor],
            to_node_id: Union[str, int, torch.Tensor]
    ) -> None:
        """
        Marks an edge for removal.
        Args:
            from_node_id: Source node of the edge.
            to_node_id: Target node of the edge.
        """
        self.edges["remove"].append([
            self.to_scalar_str(from_node_id),
            self.to_scalar_str(to_node_id)
        ])

    def remove_edges(
            self,
            edge_list: List[List[Union[str, int, torch.Tensor]]]
    ) -> None:
        """ Marks multiple edges for removal, each given as [from_node_id, to_node_id].
        """
        for from_id, to_id in edge_list:
            self.remove_edge(from_id, to_id)

    def set_edges(
            self,
            remove_edges: List[List[Union[str, int, torch.Tensor]]],
            add_edges: List[List[Union[str, int, torch.Tensor, torch.Tensor]]]
    ) -> None:
        """ Sets all edge modification fields at once.
        Args:
            remove_edges: List of [from_node_id, to_node_id] pairs to remove.
            add_edges: List of [from_node_id, to_node_id, edge_attr or None] to add.
        """
        self.remove_edges(remove_edges)
        self.add_edges(add_edges)

    @staticmethod
    def _tensor_to_list(
            tensor: Optional[torch.Tensor]
    ) -> Optional[List[float]]:
        """
        Converts a tensor to a list for JSON serialization.
        Args:
            tensor: Tensor to convert.
        Returns:
            List of values, or None if input is None.
        """
        return tensor.tolist() if tensor is not None else None

    @staticmethod
    def _list_to_tensor(
            data: Optional[List[float]]
    ) -> Optional[torch.Tensor]:
        """
        Converts a list to a float tensor.
        Args:
            data: List to convert.
        Returns:
            torch.Tensor, or None if input is None.
        """
        return torch.tensor(data) if data is not None else None

    def to_json(self) -> Dict:
        """ Serializes the artifact to a JSON-compatible dictionary.
        """
        return {
            "nodes": {
                "remove": self.nodes["remove"],
                "add": {
                    k: self._tensor_to_list(v) for k, v in self.nodes["add"].items()
                },
                "change_f": self.nodes["change_f"]
            },
            "edges": {
                "remove": self.edges["remove"],
                "add": [
                    [a, b, self._tensor_to_list(c)] for a, b, c in self.edges["add"]
                ]
            }
        }

    @classmethod
    def from_json(
            cls,
            filepath: Union[str, Path]
    ) -> 'GraphModificationArtifact':
        """
        Loads the artifact from a JSON file.
        Args:
            filepath: Path to the JSON file.
        Returns:
            GraphModificationArtifact: Loaded instance.
        """
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
            instance = cls()
            instance.nodes["remove"] = data["nodes"]["remove"]
            instance.nodes["add"] = {
                k: cls._list_to_tensor(v) for k, v in data["nodes"]["add"].items()
            }
            instance.nodes["change_f"] = data["nodes"]["change_f"]
            instance.edges["remove"] = data["edges"]["remove"]
            instance.edges["add"] = [
                [a, b, cls._list_to_tensor(c)] for a, b, c in data["edges"]["add"]
            ]
            return instance
        except Exception as e:
            raise RuntimeError(f"Failed to load artifact from {filepath}: {str(e)}")

    def clear(self) -> None:
        """ Clears all node and edge modifications.
        """
        self.__init__()

    def summary(self) -> Dict[str, int]:
        """ Returns a summary of all modifications as counts per category.
        """
        return {
            "nodes_removed": len(self.nodes["remove"]),
            "nodes_added": len(self.nodes["add"]),
            "nodes_features_changed": len(self.nodes["change_f"]),
            "edges_removed": len(self.edges["remove"]),
            "edges_added": len(self.edges["add"])
        }


class GlobalNodeIndexer:
    """
    Builds a bidirectional mapping between local per-graph node IDs and
    global node IDs across an entire dataset.
    """

    def __init__(self, dataset) -> None:
        """
        Args:
            dataset: Iterable of graph Data objects, each with a `num_nodes` attribute.
        """
        self.offsets = []
        self.global_to_local = {}
        self.local_to_global = {}

        # Assign contiguous global IDs to each graph's nodes
        offset = 0
        for graph_idx, data in enumerate(dataset):
            self.offsets.append(offset)
            for local_id in range(data.num_nodes):
                global_id = offset + local_id
                self.local_to_global[(graph_idx, local_id)] = global_id
                self.global_to_local[global_id] = (graph_idx, local_id)
            offset += data.num_nodes

    def to_global(
            self,
            graph_idx: int,
            local_id: int
    ) -> int:
        """ Returns the global node ID for a given graph index and local node ID.
        """
        return self.local_to_global[(graph_idx, local_id)]

    def to_local(
            self,
            global_id: int
    ) -> tuple:
        """ Returns (graph_idx, local_id) for a given global node ID.
        """
        return self.global_to_local[global_id]

    def graph_offset(
            self,
            graph_idx: int
    ) -> int:
        """ Returns the global ID offset for the first node in a given graph.
        """
        return self.offsets[graph_idx]