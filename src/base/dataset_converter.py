import json
from pathlib import Path
from typing import Union

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from base.datasets_processing import DatasetInfo


class DatasetConverter:
    """
    Converts graph data from one format to another.
    """
    supported_formats = ["adjlist", "edgelist", "gml", "g6", "s6"]

    @staticmethod
    def format_to_ij(
            info: DatasetInfo,
            graph_files: [Path],
            format: str,
            output_dir: Path,
            default_node_attr_value: dict = None,
            default_edge_attr_value: dict = None,
    ) -> None:
        """
        Convert graph data to our 'ij' format.
        """
        assert format in DatasetConverter.supported_formats
        if format in ['.g6', '.s6']:
            if info.remap:
                raise RuntimeError(f"Graphs in '{format}' format don't support nodes remapping,"
                                   f" nodes must be enumerated from 0 to N-1.")

        # Read to networkx
        create_using = nx.DiGraph if info.directed else nx.Graph
        graphs = []
        for path in graph_files:
            graph = read_nx_graph(format, path, create_using=create_using)
            graphs.append(graph)

        assert len(graphs) == info.count

        # Extract attributes
        all_node_attributes = []
        all_edge_attributes = []
        for graph in graphs:
            node_attributes, edge_attributes = extract_attributes(
                graph, default_node_attr_value, default_edge_attr_value)
            all_node_attributes.append(node_attributes)
            all_edge_attributes.append(edge_attributes)

        # Write graphs and attributes to output dir
        with open(output_dir / f'{info.name}.ij', 'w') as f:
            for graph in graphs:
                for i, j in graph.edges:
                    f.write(f'{i} {j}\n')
        if len(graphs) > 1:
            edge_index = []
            edges = 0
            for graph in graphs:
                edges += graph.number_of_edges()
                edge_index.append(edges)
            with open(output_dir / f'{info.name}.edge_index', 'w') as f:
                json.dump(edge_index, f)

        node_attr_dir = output_dir / f'{info.name}.node_attributes'
        node_attr_dir.mkdir()
        edge_attr_dir = output_dir / f'{info.name}.edge_attributes'
        edge_attr_dir.mkdir()
        if len(graphs) == 1:
            all_node_attributes = all_node_attributes[0]
            all_edge_attributes = all_edge_attributes[0]
            for attr, data in all_node_attributes.items():
                with open(node_attr_dir / attr, 'w') as f:
                    json.dump(data, f)
            for attr, data in all_edge_attributes.items():
                with open(edge_attr_dir / attr, 'w') as f:
                    json.dump(data, f)
        else:
            keys = set.union(*[set(a.keys()) for a in all_node_attributes])
            for attr in keys:
                with open(node_attr_dir / attr, 'w') as f:
                    json.dump([a[attr] for a in all_node_attributes], f)
            keys = set.union(*[set(a.keys()) for a in all_edge_attributes])
            for attr in keys:
                with open(edge_attr_dir / attr, 'w') as f:
                    json.dump([a[attr] for a in all_edge_attributes], f)

        # We do not add the extracted attributes to DatasetInfo, since it regulates whether
        # attributes should be used.

    @staticmethod
    def networkx_to_format(
            graph: nx.Graph,
            format: str,
            output_dir: Path,
            default_node_attr_value: dict = None,
            default_edge_attr_value: dict = None,
            name: str = 'networkx_graph'
    ) -> None:
        """
        Write a networkx graph to files according to a specified format.
        Attribute files will be created if necessary.
        """
        node_attributes, edge_attributes = extract_attributes(
            graph, default_node_attr_value, default_edge_attr_value)

        graph_file = output_dir / f'{name}.{format}'
        node_attrs_ok = False
        edge_attrs_ok = False
        if format == "adjlist":
            nx.write_adjlist(graph, graph_file)

        elif format == "edgelist":
            nx.write_edgelist(graph, graph_file)

        elif format == "gml":
            nx.write_gml(graph, graph_file)
            node_attrs_ok = True
            edge_attrs_ok = True

        elif format == "g6":
            nx.write_graph6(graph, str(graph_file))

        elif format == "s6":
            nx.write_sparse6(graph, graph_file)

        # FORMATS THAT ARE NOT SUPPORTED:
        # gexf, multiline_adjlist, weighted_edgelist
        # # GRAPHML DOESN'T WORK WITH from_networkx()
        # elif data_format == "graphml":
        # # LEDA format is not supported as it stores edge attributes as strings
        # elif data_format == "leda":
        # # PAJEK format is not supported as it stores node attributes as strings
        # elif data_format == "pajek": # Only works with graphs that have node labels

        else:
            raise NotImplementedError

        node_attributes_dir = output_dir / f'{name}.node_attributes'
        edge_attributes_dir = output_dir / f'{name}.edge_attributes'

        if not node_attrs_ok:
            node_attributes_dir.mkdir()
            for attr, data in node_attributes.items():
                with open(node_attributes_dir / str(attr), 'w') as f:
                    json.dump(node_attributes[attr], f)
        if not edge_attrs_ok:
            edge_attributes_dir.mkdir()
            for attr, data in edge_attributes.items():
                with open(edge_attributes_dir / str(attr), 'w') as f:
                    json.dump(edge_attributes[attr], f)


def extract_attributes(
        graph: nx.Graph,
        default_node_attr_value: dict = None,
        default_edge_attr_value: dict = None,
) -> (dict, dict):
    """
    Extract nodes and edges attributes from a networkx graph.
    """
    all_node_attributes_names = set()
    all_edge_attributes_names = set()
    for node in graph.nodes(data=True):
        all_node_attributes_names.update(node[1].keys())
    for edge in graph.edges(data=True):
        all_edge_attributes_names.update(edge[2].keys())
    node_attributes = {attr: {} for attr in all_node_attributes_names}
    edge_attributes = {attr: {} for attr in all_edge_attributes_names}
    # for attr in all_node_attributes_names:
    #     node_attributes[attr] = nx.get_node_attributes(graph, attr, default_node_attr_value[attr])
    # for attr in all_edge_attributes_names:
    #     edge_attributes[attr] = nx.get_edge_attributes(graph, attr, default_edge_attr_value[attr])
    for n, data in graph.nodes(data=True):
        for attr in all_node_attributes_names:
            node_attributes[attr][n] = data[attr] if attr in data else default_node_attr_value[attr]
    for i, j, data in graph.edges(data=True):
        for attr in all_edge_attributes_names:
            edge_attributes[attr][f"{i},{j}"] = data[attr] if attr in data else default_edge_attr_value[attr]
            # edge_attributes[attr][f"{i},{j}"] = data.get(attr, default_edge_attr_value[attr])
    return node_attributes, edge_attributes


def read_nx_graph(
        data_format: str,
        path: Union[Path, str],
        **kwargs
) -> nx.Graph:
    # FORMATS THAT ARE NOT SUPPORTED:
    # gexf, multiline_adjlist, weighted_edgelist
    if data_format == "adjlist":  # This format does not store graph or node attributes.
        return nx.read_adjlist(path, **kwargs)
    elif data_format == "edgelist":
        return nx.read_edgelist(path, **kwargs)
    elif data_format == "gml":  # Only works with graphs that have node, edge attributes
        return nx.read_gml(path)
    # # GRAPHML DOESN'T WORK WITH from_networkx()
    # elif data_format == "graphml":
    #     return nx.read_graphml(path, **kwargs)
    # # LEDA format is not supported as it stores edge attributes as strings
    # elif data_format == "leda":
    #     return nx.read_leda(path, **kwargs)
    elif data_format == "g6":
        return nx.read_graph6(path)
    elif data_format == "s6":
        return nx.read_sparse6(path)
    # # PAJEK format is not supported as it stores node attributes as strings
    # elif data_format == "pajek": # Only works with graphs that have node labels
    #     return nx.read_pajek(path, **kwargs)
    else:
        raise RuntimeError("the READING format is NOT SUPPORTED!!!")


def networkx_to_ptg(
        nx_graph: nx.Graph
) -> Data:
    """
    Convert networkx graph to a PTG Data.
    Nodes and edges attributes, that numeric, are concatenated.
    """
    node_attribute_names = set()
    edge_attribute_names = set()

    # Iterating through the nodes and collect unique attribute names
    for node, data in nx_graph.nodes(data=True):
        for attribute_name in data:
            node_attribute_names.add(attribute_name)

    # Iterating through the edges and collect unique attribute names
    for u, v, data in nx_graph.edges(data=True):
        for attribute_name in data:
            edge_attribute_names.add(attribute_name)

    node_attribute_names_list = []
    edge_attribute_names_list = []
    # Get only attributes that have numeric types
    for attr in sorted(node_attribute_names):
        if all(isinstance(v, (int, float, complex, list))
               for v in nx.get_node_attributes(nx_graph, attr).values()):
            node_attribute_names_list.append(attr)

    for attr in sorted(edge_attribute_names):
        if all(isinstance(v, (int, float, complex, list))
               for v in nx.get_edge_attributes(nx_graph, attr).values()):
            edge_attribute_names_list.append(attr)

    if len(node_attribute_names_list) < 1:
        node_attribute_names_list = None
    if len(edge_attribute_names_list) < 1:
        edge_attribute_names_list = None

    ptg_graph = from_networkx(nx_graph, group_node_attrs=node_attribute_names_list,
                              group_edge_attrs=edge_attribute_names_list)
    return ptg_graph


def example_single():
    g = nx.Graph()
    g.add_node(11, a=0.4, b=100)
    g.add_node(12, a=0.3, b=50)
    g.add_node(13, a=0.3, b=200)
    g.add_node(14, a=0.2, b=75)
    g.add_node(15, a=0.4, b=25)
    g.add_node(16, a=0.2, b=150)
    g.add_node(17, a=0.5, b=80)
    g.add_node(18, a=0.1, b=40)
    g.add_edge(11, 12, weight=5, type='big')
    g.add_edge(11, 13, weight=3, type='medium')
    g.add_edge(11, 14, weight=4, type='small')
    g.add_edge(12, 15, weight=2, type='big')
    g.add_edge(12, 16, weight=6, type='big')
    g.add_edge(13, 14, weight=3, type='medium')
    g.add_edge(13, 17, weight=5, type='small')
    g.add_edge(14, 18, weight=4, type='big')
    g.add_edge(15, 16, weight=1, type='small')
    g.add_edge(16, 17, weight=5, type='small')
    g.add_edge(17, 18, weight=3, type='medium')

    from aux.configs import DatasetConfig
    from base.datasets_processing import DatasetManager
    from aux.declaration import Declare

    name = 'example_gml'
    dc = DatasetConfig('single-graph', 'custom', name)

    # Create directory
    root, files_paths = Declare.dataset_root_dir(dc)
    raw = root / 'raw'
    raw.mkdir(parents=True)

    # Write info and labels
    nx.write_gml(g, raw / 'graph.gml')
    with open(raw / '.info', 'w') as f:
        json.dump({
            "name": name,
            "count": 1,
            "directed": True,
            "nodes": [g.number_of_nodes()],
            "remap": True,
            "node_attributes": {
                "names": ["a", "b"],
                "types": ["continuous", "continuous"],
                "values": [[0, 1], [0, 200]]
            },
            "edge_attributes": {
                "names": ["weight", "type"],
                "types": ["continuous", "categorical"],
                "values": [[1, 6], ['small', 'medium', 'big']]
            },
            "labelings": {"binary": 2}
        }, f)

    (raw / f'{name}.labels').mkdir()
    with open(raw / f'{name}.labels' / 'binary', 'w') as f:
        json.dump({"11": 1, "12": 0, "13": 0, "14": 0, "15": 0, "16": 0, "17": 0, "18": 0}, f)

    custom_dataset = DatasetManager.register_custom(
        dc, 'gml',
        default_node_attr_value={'a': -1, 'b': -1},
        default_edge_attr_value={'weight': -1, 'type': -1})
    custom_dataset.check_validity()


def example_multi():
    # Multi
    g1 = nx.Graph()
    g1.add_node(1, a=10, b='alpha')
    g1.add_node(2, a=20, b='beta')
    g1.add_node(3, a=30, b='gamma')
    g1.add_node(4, a=40, b='delta')
    g1.add_edge(1, 2, weight=1.5, type='mixed')
    g1.add_edge(2, 3, weight=2.7)
    g1.add_edge(3, 4, type='complex')
    g1.add_edge(1, 4, weight=0.9, type='hybrid')

    g2 = nx.Graph()
    g2.add_node(1, a=15, b='alpha')
    g2.add_node(2, a=25, b='beta')
    g2.add_node(3, a=35, b='gamma')
    g2.add_node(4, a=45, b='delta')
    g2.add_edge(1, 2, weight=1.2, type='mixed')
    g2.add_edge(2, 3, weight=2.3, type='complex')
    g2.add_edge(3, 4, weight=3.4)
    g2.add_edge(1, 4, weight=4.5, type='hybrid')

    g3 = nx.Graph()
    g3.add_node(1, a=20, b='alpha')
    g3.add_node(2, a=30, b='beta')
    g3.add_node(3, a=40, b='gamma')
    g3.add_node(4, a=50, b='delta')
    g3.add_node(5, a=60)
    g3.add_edge(1, 2, weight=1.8, type='mixed')
    g3.add_edge(2, 3)
    g3.add_edge(3, 4, weight=2.5, type='complex')
    g3.add_edge(4, 5, type='hybrid')
    g3.add_edge(1, 5, weight=3.2)

    from aux.configs import DatasetConfig
    from base.datasets_processing import DatasetManager
    from aux.declaration import Declare

    name = 'example_gml'
    dc = DatasetConfig('multiple-graphs', 'custom', name)

    # Create directory
    root, files_paths = Declare.dataset_root_dir(dc)
    raw = root / 'raw'
    raw.mkdir(parents=True)

    # Write info and labels
    nx.write_gml(g1, raw / 'graph1.gml')
    nx.write_gml(g2, raw / 'graph2.gml')
    nx.write_gml(g3, raw / 'graph3.gml')
    with open(raw / '.info', 'w') as f:
        json.dump({
            "name": name,
            "count": 3,
            "directed": False,
            "nodes": [g.number_of_nodes() for g in [g1,g2,g3]],
            "remap": True,
            "node_attributes": {
                "names": ["a", "b"],
                "types": ["continuous", "categorical"],
                "values": [[0, 100], ['alpha', 'beta', 'gamma', 'delta']]
            },
            "edge_attributes": {
                "names": ["weight", "type"],
                "types": ["continuous", "categorical"],
                "values": [[0, 5], ['mixed', 'complex', 'hybrid']]
            },
            "labelings": {"binary": 2}
        }, f)

    (raw / f'{name}.labels').mkdir()
    with open(raw / f'{name}.labels' / 'binary', 'w') as f:
        json.dump({"0":1,"1":0,"2":0}, f)

    custom_dataset = DatasetManager.register_custom(
        dc, 'gml',
        default_node_attr_value={'a': 0, 'b': 'alpha'},
        default_edge_attr_value={'weight': 1, 'type': 'mixed'})
    custom_dataset.check_validity()


if __name__ == '__main__':
    # example_single()
    example_multi()
