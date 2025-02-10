import os
import json
from collections import Counter
from pathlib import Path
from typing import Union

import networkx as nx
from networkx import NetworkXError, NetworkXNotImplemented
from torch_geometric.data import Dataset

from base.datasets_processing import GeneralDataset


class DatasetStats:
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
        'feature_distr',
        'feature_assort',
    ]
    all_stats = stats + var_stats

    def __init__(
            self,
            dataset: GeneralDataset
    ):
        self.gen_dataset: GeneralDataset = dataset

        self.stats = {}  # keep the computed stats
        self.nx_graph = None  # converted to networkx version

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
            directory = self.gen_dataset.root_dir / '.stats'
        elif stat in DatasetStats.var_stats:
            # We suppose here that dataset_var_config is defined for our gen dataset.
            directory = self.gen_dataset.results_dir / '.stats'
        else:
            raise NotImplementedError
        directory.mkdir(exist_ok=True)
        return directory / stat

    def get(
            self,
            stat: str
    ) -> Union[int, float, dict, str]:
        """ Get the specified statistics.
        It will be read from file or computed and saved.
        """
        assert stat in DatasetStats.all_stats
        if stat in self.stats:
            return self.stats[stat]

        # Try to read from file
        path = self._save_path(stat)
        if path.exists():
            with path.open('r') as f:
                value = json.load(f)
            self.stats[stat] = value
            return value

        # Compute
        method = {
            False: self._compute,
            True: self._compute_multi,
        }[self.is_multi]
        method(stat)
        value = self.stats[stat]
        if value is None:
            value = f"Statistics '{stat}' is not implemented."

        return value

    def set(
            self,
            stat: str,
            value: Union[int, float, dict, str]
    ) -> None:
        """ Set statistics to a specified value and save to file.
        """
        assert stat in DatasetStats.all_stats
        self.stats[stat] = value
        path = self._save_path(stat)
        with path.open('w') as f:
            json.dump(value, f, ensure_ascii=False, indent=1)

    def remove(
            self,
            stat: str
    ) -> None:
        """ Remove statistics from dict and file.
        """
        if stat in self.stats:
            del self.stats[stat]
        try:
            os.remove(self._save_path(stat))
        except FileNotFoundError: pass

    def clear_all_stats(
            self
    ) -> None:
        """ Remove all stats. E.g. the graph has changed.
        """
        for s in DatasetStats.all_stats:
            self.remove(s)

    def update_var_config(
            self
    ) -> None:
        """ Remove var stats from dict since dataset config has changed.
        """
        for s in DatasetStats.var_stats:
            if s in self.stats:
                del self.stats[s]

    def _compute(
            self,
            stat: str
    ) -> None:
        """ Compute statistics for a single graph.
        Result could be: a number, a string, a distribution, a dict of ones.
        """
        # assert self.info.count == 1
        # data: Data = self.dataset.get(0)
        edges = self.gen_dataset.dataset_data["edges"][0]
        num_nodes = self.gen_dataset.info.nodes[0]

        # Simple stats
        if stat in ["num_edges", "avg_degree"]:
            num_edges = len(edges)
            avg_deg = len(edges) / num_nodes * (1 if self.is_directed else 2)
            self.set("num_edges", num_edges)
            self.set("avg_degree", avg_deg)
            return

        # Var stats
        if stat == "label_distr":
            labels = self.gen_dataset.dataset_var_data["labels"]
            self.set("label_distr", list_to_hist(labels))
            return

        # More complex stats - we use networkx
        if self.nx_graph is None:
            # Converting to networkx
            self.nx_graph = nx.DiGraph() if self.is_directed else nx.Graph()
            for i, j in edges:
                self.nx_graph.add_edge(i, j)

        try:
            # TODO misha simplify - some stats can be computed easier

            if stat == "clustering_coeff":
                # NOTE this is average local clustering, not global
                self.set("clustering_coeff", nx.average_clustering(self.nx_graph))

            elif stat == "num_triangles":
                self.set("num_triangles", int(sum(nx.triangles(self.nx_graph).values()) / 3))

            elif stat in ['gcc_size', 'gcc_rel_size', 'num_cc', 'cc_distr', 'gcc_diam']:
                if self.is_directed:
                    wcc = list(nx.weakly_connected_components(self.nx_graph))
                    scc = list(nx.strongly_connected_components(self.nx_graph))
                    self.set("num_cc", {"WCC": len(wcc), "SCC": len(scc)})
                    self.set("gcc_size", {"WCC": len(wcc[0]), "SCC": len(scc[0])})
                    self.set("gcc_rel_size", {"WCC": len(wcc[0]) / num_nodes,
                                              "SCC": len(scc[0]) / num_nodes})
                    self.set("cc_distr", {"WCC": list_to_hist([len(c) for c in wcc]),
                                          "SCC": list_to_hist([len(c) for c in scc])})
                    self.set("gcc_diam", nx.diameter(self.nx_graph.subgraph(scc[0])))
                    # self.set("gcc_diam", {"WCC": nx.diameter(self.nx_graph.to_undirected().subgraph(wcc[0])),
                    #                       "SCC": nx.diameter(self.nx_graph.subgraph(scc[0]))})
                else:
                    cc = list(nx.connected_components(self.nx_graph))
                    self.set("num_cc", len(cc))
                    self.set("gcc_size", len(cc[0]))
                    self.set("gcc_rel_size", len(cc[0]) / num_nodes)
                    self.set("cc_distr", list_to_hist([len(c) for c in cc]))
                    self.set("gcc_diam", nx.diameter(self.nx_graph.subgraph(cc[0])))

            elif stat == "degree_assort":
                if self.is_directed:
                    degree_assort = {
                        "in-in": nx.degree_assortativity_coefficient(self.nx_graph, "in", "in"),
                        "in-out": nx.degree_assortativity_coefficient(self.nx_graph, "in", "out"),
                        "out-in": nx.degree_assortativity_coefficient(self.nx_graph, "out", "in"),
                        "out-out": nx.degree_assortativity_coefficient(self.nx_graph, "out", "out"),
                    }
                    self.set("degree_assort", degree_assort)
                else:
                    self.set("degree_assort", nx.degree_assortativity_coefficient(self.nx_graph))

            elif stat == "degree_distr":
                if self.is_directed:
                    self.set("degree_distr", {
                        "in": list_to_hist([d for _, d in self.nx_graph.in_degree()]),
                        "out": list_to_hist([d for _, d in self.nx_graph.out_degree()])
                    })
                else:
                    self.set("degree_distr", {i: d for i, d in enumerate(nx.degree_histogram(self.nx_graph))})

            elif stat == "label_assort":
                labels = self.gen_dataset.dataset_var_data["labels"]
                nx.set_node_attributes(self.nx_graph, dict(list(enumerate(labels))), 'label')
                self.set("label_assort", nx.attribute_assortativity_coefficient(self.nx_graph, 'label'))
                return

            else:
                value = self.gen_dataset._compute_stat(stat)
                self.set(stat, value)
        except (NetworkXError, NetworkXNotImplemented) as e:
            self.set(stat, str(e))

    def _compute_multi(
            self,
            stat: str
    ) -> None:
        """ Compute statistics for a multiple-graphs dataset.
        Result could be: a number, a string, a distribution, a dict of ones.
        """
        edges = self.gen_dataset.dataset_data["edges"]

        # Var stats
        if stat == "label_distr":
            labels = self.gen_dataset.dataset_var_data["labels"]
            self.set("label_distr", list_to_hist([x for xs in labels for x in xs]))
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
            value = 'Unknown stats'
        # except (NetworkXError, NetworkXNotImplemented) as e:
        #     value = str(e)


def list_to_hist(
        a_list: list
) -> dict:
    """ Convert a list of integers/floats to a frequency histogram, return it as a dict
    """
    return {k: v for k, v in Counter(a_list).most_common()}
