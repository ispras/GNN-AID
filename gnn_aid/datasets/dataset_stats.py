import json
from collections import Counter
from pathlib import Path
from typing import Union

import networkx as nx
import numpy as np
from networkx import NetworkXError, NetworkXNotImplemented
from torch_geometric.data import Dataset

from gnn_aid.aux.declaration import Declare
from gnn_aid.aux.utils import edge_index_to_edge_list
from gnn_aid.data_structures.configs import DatasetVarConfig, FeatureConfig, Task
from .gen_dataset import GeneralDataset
from .known_format_datasets import KnownFormatDataset


class DatasetStats:
    """
    Stores and computes statistics for a dataset.
    """
    stats = [
        'num_nodes',
        'num_edges',
        'avg_degree',
        'degree_distr',
        'degree_assort',

        'clustering_coeff',
        'num_triangles',

        'gcc_size',
        'gcc_rel_size',
        'num_cc',
        'cc_distr',
        'gcc_diam',
        # 'gcc_diam90',

        'attr_corr',
        ]
    var_stats = [
        'label_distr',
        'label_assort',
        # 'feature_distr',
        # 'feature_assort',
    ]
    all_stats = stats + var_stats

    multi_stats = [
        'num_nodes',
        'num_edges',
        'avg_degree',

        'label_distr',
        # 'feature_distr',
    ]

    def __init__(
            self,
            dataset: GeneralDataset
    ):
        self.gen_dataset: GeneralDataset = dataset

        self._store = {}  # keep the computed stats
        self._nx_graph = None  # converted to networkx version

    @property
    def dataset(
            self
    ) -> Dataset:
        return self.gen_dataset.dataset  # PTG Dataset

    @property
    def is_directed(
            self
    ) -> bool:
        return self.gen_dataset.info.directed

    @property
    def is_multi(
            self
    ) -> bool:
        return self.gen_dataset.is_multi()

    def _save_path(
            self,
            stat: str
    ) -> Path:
        """ Save directory for statistics of variable part. """
        directory = None
        if stat in DatasetStats.stats:
            # Can't call prepared_dir.parent since dataset_var_config may be None
            _, [_, dvc_path] = Declare.dataset_prepared_dir(
                self.gen_dataset.dataset_config, DatasetVarConfig(
                    task=None, features=FeatureConfig(), dataset_ver_ind=0)
            )
            directory = dvc_path.parent / 'stats'
        elif stat in DatasetStats.var_stats:
            # We suppose here that dataset_var_config is defined for our gen dataset.
            directory = self.gen_dataset.prepared_dir / 'var_stats'
        else:
            raise KeyError(f"Unknown stat {stat}")
        directory.mkdir(parents=True, exist_ok=True)
        return directory / stat

    def __getitem__(
            self,
            item: str
    ) -> Union[int, float, dict, str]:
        return self.get(stat=item)

    def get(
            self,
            stat: str
    ) -> Union[int, float, dict, str]:
        """ Get the specified statistic, reading from file or computing and saving if needed.
        """
        assert stat in DatasetStats.all_stats
        if stat in self._store:
            return self._store[stat]

        # Try to read from file
        path = self._save_path(stat)
        if path.exists():
            with path.open('r') as f:
                value = json.load(f)
            self._store[stat] = value
            return value

        # Compute
        if self.is_multi:
            self._compute_multi(stat)
        else:
            self._compute(stat)

        return self._store[stat]

    def __setitem__(
            self,
            key: str,
            value: Union[int, float, dict, str]
    ) -> None:
        self.set(key, value)

    def set(
            self,
            stat: str,
            value: Union[int, float, dict, str]
    ) -> None:
        """ Set a statistic to a specified value and save to file.
        """
        assert stat in DatasetStats.all_stats
        self._store[stat] = value
        path = self._save_path(stat)
        with path.open('w') as f:
            json.dump(value, f, ensure_ascii=False, indent=1)

    def _compute(
            self,
            stat: str
    ) -> None:
        """ Compute a statistic for a single graph and store it.

        Result can be a number, a string, a distribution dict, or a dict of values.
        """
        edges = edge_index_to_edge_list(self.gen_dataset.edges[0], self.gen_dataset.is_directed())
        num_nodes = self.gen_dataset.info.nodes[0]

        # Simple stats
        if stat == "num_nodes":
            self.set("num_nodes", num_nodes)
            return
        if stat in ["num_edges", "avg_degree"]:
            num_edges = len(edges)
            avg_deg = len(edges) / num_nodes * (1 if self.is_directed else 2)
            self.set("num_edges", num_edges)
            self.set("avg_degree", avg_deg)
            return

        # Var stats
        if stat == "label_distr":
            labels = self.gen_dataset.labels.tolist()
            self.set("label_distr", list_to_hist(labels))
            return

        # More complex stats - we use networkx
        if self._nx_graph is None:
            # Converting to networkx
            self._nx_graph = nx.DiGraph() if self.is_directed else nx.Graph()
            for i, j in edges:
                self._nx_graph.add_edge(i, j)

        try:
            # TODO misha simplify - some stats can be computed easier

            if stat == "clustering_coeff":
                # NOTE this is average local clustering, not global
                self.set("clustering_coeff", nx.average_clustering(self._nx_graph))

            elif stat == "num_triangles":
                self.set("num_triangles", int(sum(nx.triangles(self._nx_graph).values()) / 3))

            elif stat in ['gcc_size', 'gcc_rel_size', 'num_cc', 'cc_distr', 'gcc_diam']:
                if self.is_directed:
                    wcc = list(nx.weakly_connected_components(self._nx_graph))
                    scc = list(nx.strongly_connected_components(self._nx_graph))
                    self.set("num_cc", {"WCC": len(wcc), "SCC": len(scc)})
                    self.set("gcc_size", {"WCC": len(wcc[0]), "SCC": len(scc[0])})
                    self.set("gcc_rel_size", {"WCC": len(wcc[0]) / num_nodes,
                                              "SCC": len(scc[0]) / num_nodes})
                    self.set("cc_distr", {"WCC": list_to_hist([len(c) for c in wcc]),
                                          "SCC": list_to_hist([len(c) for c in scc])})
                    self.set("gcc_diam", nx.diameter(self._nx_graph.subgraph(scc[0])))
                    # self.set("gcc_diam", {"WCC": nx.diameter(self.nx_graph.to_undirected().subgraph(wcc[0])),
                    #                       "SCC": nx.diameter(self.nx_graph.subgraph(scc[0]))})
                else:
                    cc = list(nx.connected_components(self._nx_graph))
                    self.set("num_cc", len(cc))
                    self.set("gcc_size", len(cc[0]))
                    self.set("gcc_rel_size", len(cc[0]) / num_nodes)
                    self.set("cc_distr", list_to_hist([len(c) for c in cc]))
                    self.set("gcc_diam", nx.diameter(self._nx_graph.subgraph(cc[0])))

            elif stat == "degree_assort":
                if self.is_directed:
                    degree_assort = {
                        "in-in": nx.degree_assortativity_coefficient(self._nx_graph, "in", "in"),
                        "in-out": nx.degree_assortativity_coefficient(self._nx_graph, "in", "out"),
                        "out-in": nx.degree_assortativity_coefficient(self._nx_graph, "out", "in"),
                        "out-out": nx.degree_assortativity_coefficient(self._nx_graph, "out", "out"),
                    }
                    self.set("degree_assort", degree_assort)
                else:
                    self.set("degree_assort", nx.degree_assortativity_coefficient(self._nx_graph))

            elif stat == "degree_distr":
                if self.is_directed:
                    self.set("degree_distr", {
                        "in": list_to_hist([d for _, d in self._nx_graph.in_degree()]),
                        "out": list_to_hist([d for _, d in self._nx_graph.out_degree()])
                    })
                else:
                    self.set("degree_distr", {i: d for i, d in enumerate(nx.degree_histogram(self._nx_graph))})

            elif stat == "label_assort":
                labels = self.gen_dataset.labels
                nx.set_node_attributes(self._nx_graph, dict(list(enumerate(labels))), 'label')
                self.set("label_assort", nx.attribute_assortativity_coefficient(self._nx_graph, 'label'))

            elif stat == "attr_corr":
                if not isinstance(self.gen_dataset, KnownFormatDataset):
                    raise NotImplementedError(f"Not implemented for dataset class"
                                              f" {self.gen_dataset.__class__.__name__}")

                attrs = []
                # Pick suitable attributes - continuous type
                for attr, _type in zip(
                        self.gen_dataset.info.node_attributes['names'],
                        self.gen_dataset.info.node_attributes['types']):
                    if _type in ["continuous"]:
                        attrs.append(attr)

                attr_node_attrs = self.gen_dataset.node_attributes(attrs)

                # Compute mean and std over edges
                in_attr_mean = {}
                in_attr_denom = {}
                out_attr_mean = {}
                out_attr_denom = {}
                for a, node_attrs in attr_node_attrs.items():
                    ins = []
                    outs = []
                    node_attrs = node_attrs[0]
                    for i, j in edges:
                        outs.append(node_attrs[i])
                        ins.append(node_attrs[j])
                    in_attr_mean[a] = np.mean(ins)
                    in_attr_denom[a] = (np.sum(np.array(ins) ** 2) - len(edges) * in_attr_mean[
                        a] ** 2) ** 0.5
                    out_attr_mean[a] = np.mean(outs)
                    out_attr_denom[a] = (np.sum(np.array(outs) ** 2) - len(edges) *
                                         out_attr_mean[
                                             a] ** 2) ** 0.5

                # Compute corr
                attrs = list(attr_node_attrs.keys())
                # Matrix of corr numerators
                pearson_corr = np.zeros((len(attrs), len(attrs)), dtype=float)
                for i, out_a in enumerate(attrs):
                    out_node_attrs = attr_node_attrs[out_a][0]
                    for j, in_a in enumerate(attrs):
                        in_node_attrs = attr_node_attrs[in_a][0]
                        corr = 0
                        for x, y in edges:
                            corr += (out_node_attrs[x] - out_attr_mean[out_a]) * (
                                    in_node_attrs[y] - in_attr_mean[in_a])
                        pearson_corr[i][j] = corr

                # Normalize on stds
                for i, out_a in enumerate(attrs):
                    for j, in_a in enumerate(attrs):
                        denom = out_attr_denom[out_a] * in_attr_denom[in_a]
                        pc = pearson_corr[i][j] / denom if denom != 0 else 1
                        pearson_corr[i][j] = min(1, max(-1, pc))

                self.set(stat, {'attributes': attrs, 'correlations': pearson_corr.tolist()})
                return

            else:
                raise NotImplementedError()

        except (NetworkXError, NetworkXNotImplemented, NotImplementedError) as e:
            self.set(stat, str(e))

    def _compute_multi(
            self,
            stat: str
    ) -> None:
        """
        Compute a statistic for a multiple-graphs dataset and store it.
        Result can be a number, a string, a distribution dict, or a dict of values.
        """
        edges = self.gen_dataset.edges

        # Var stats
        if stat == "label_distr":
            labels = self.gen_dataset.labels
            self.set("label_distr", list_to_hist([int(y) for y in labels]))
            return

        # Simple stats
        if stat in ["num_nodes", "num_edges", "avg_degree"]:
            num_nodes = list(self.gen_dataset.info.nodes)
            num_edges = [len(e) for e in edges]
            avg_degree = [e / n * (1 if self.is_directed else 2) for n, e in zip(num_nodes, num_edges)]

            self.set("num_nodes", list_to_hist(num_nodes))
            self.set("num_edges", list_to_hist(num_edges))
            self.set("avg_degree", list_to_hist(avg_degree))
            return

        else:
            raise NotImplementedError


def list_to_hist(
        a_list: list
) -> dict:
    """ Convert a list of values to a frequency histogram dict sorted by count.
    """
    return {k: v for k, v in Counter(a_list).most_common()}
