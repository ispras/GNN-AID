import collections.abc
collections.Callable = collections.abc.Callable

import json
import unittest
import shutil
import torch
from torch import tensor
from torch_geometric.data import InMemoryDataset, Data, Dataset

# Monkey patch main dirs - before other imports
from aux.utils import monkey_patch_directories
monkey_patch_directories(include_graphs_dir=True)

from aux.declaration import Declare
from datasets_block.dataset_converter import networkx_to_ptg
from data_structures.configs import DatasetConfig, DatasetVarConfig, FeatureConfig
from datasets_block.datasets_manager import DatasetManager
from datasets_block.known_format_datasets import KnownFormatDataset
from datasets_block.ptg_datasets import LocalPTGDataset, LibPTGDataset


def _create_single_ij(dc: DatasetConfig):
    """ Single graph in ij format """
    root, files_paths = Declare.dataset_root_dir(dc)
    raw = root / 'raw'
    raw.mkdir(parents=True)
    with open(raw / 'edges.ij', 'w') as f:
        f.write("10 11\n")
        f.write("10 12\n")
    with open(root / 'metainfo', 'w') as f:
        json.dump({
            "count": 1,
            "directed": False,
            "nodes": [3],
            "remap": True,
            "node_attributes": {
                "names": ["a", "b", "c"],
                "types": ["continuous", "categorical", "vector"],
                "values": [[0, 1], ["A", "B", "C"], 2]
            },
            "edge_attributes": {
                "names": ["weight"],
                "types": ["continuous"],
                "values": [[0, 1]]
            },
            "labelings": {
                "binary": 2,
                "threeClasses": 3
            }
        }, f)
    (raw / 'labels').mkdir()
    with open(raw / 'labels' / 'binary', 'w') as f:
        json.dump({"10": 0, "11": 1, "12": 1}, f)
    with open(raw / 'labels' / 'threeClasses', 'w') as f:
        json.dump({"10": 0, "11": 1, "12": 2}, f)

    (raw / 'node_attributes').mkdir()
    with open(raw / 'node_attributes' / 'a', 'w') as f:
        json.dump({"10": 0.0, "11": 0.1, "12": 0.2}, f)
    with open(raw / 'node_attributes' / 'b', 'w') as f:
        json.dump({"10": "A", "11": "B", "12": "C"}, f)
    with open(raw / 'node_attributes' / 'c', 'w') as f:
        json.dump({"10": [0.3, -0.2], "11": [0, 0], "12": [1e5, 0]}, f)

    # (raw / 'edge_attributes').mkdir()
    # with open(raw / 'edge_attributes' / 'weight', 'w') as f:
    #     json.dump({"10": 0.0, "11": 0.1, "12": 0.2}, f)


def _create_single2_ij(dc: DatasetConfig):
    """ Single graph in ij format, equals single-graph/example """
    root, files_paths = Declare.dataset_root_dir(dc)
    raw = root / 'raw'
    raw.mkdir(parents=True)
    with open(raw / 'edges.ij', 'w') as f:
        f.write("10 11\n")
        f.write("11 12\n")
        f.write("11 13\n")
        f.write("11 15\n")
        f.write("12 13\n")
        f.write("12 17\n")
        f.write("15 14\n")
        f.write("15 16\n")
    with open(root / 'metainfo', 'w') as f:
        json.dump({
            "class_name": "KnownFormatDataset",
            "import_from": "datasets_block.known_format_datasets",
            "name": "example",
            "count": 1,
            "directed": False,
            "nodes": [8],
            "remap": True,
            "node_attributes": {
                "names": ["a", "b"],
                "types": ["continuous", "categorical"],
                "values": [[0, 1], ["A", "B", "C"]]
            },
            "edge_attributes": {
                "names": ["weight"],
                "types": ["continuous"],
                "values": [[0, 1]]
            },
            "labelings": {
                "binary": 2,
                "threeClasses": 3
            }
        }, f)
    (raw / 'labels').mkdir()
    with open(raw / 'labels' / 'binary', 'w') as f:
        json.dump({"10": 1, "11": 1, "12": 1, "13": 1, "14": 0, "15": 0, "16": 0, "17": 0}, f)

    (raw / 'node_attributes').mkdir()
    with open(raw / 'node_attributes' / 'a', 'w') as f:
        json.dump({"10": 1, "11": 1, "12": 0.6, "13": 0.7, "14": 0.5, "15": 0.5, "16": 0.5, "17": 0.7},
                  f)
    with open(raw / 'node_attributes' / 'b', 'w') as f:
        json.dump(
            {"10": "A", "11": "A", "12": "B", "13": "C", "14": "B", "15": "A", "16": "A", "17": "C"}, f)


def _create_multi_ij(dc: DatasetConfig):
    """ Multi graph in ij format, equals multiple-graphs/custom/example """
    root, files_paths = Declare.dataset_root_dir(dc)
    raw = root / 'raw'
    raw.mkdir(parents=True)
    with open(raw / 'edges.ij', 'w') as f:
        f.write("0 1\n")
        f.write("1 0\n")
        f.write("1 2\n")
        f.write("0 1\n")
        f.write("1 2\n")
        f.write("2 3\n")
        f.write("3 0\n")
        f.write("0 1\n")
        f.write("0 2\n")
        f.write("0 3\n")
        f.write("0 4\n")
    with open(raw / 'edge_index', 'w') as f:
        f.write("[3, 7, 11]")
    with open(root / 'metainfo', 'w') as f:
        json.dump({
            "class_name": "KnownFormatDataset",
            "import_from": "datasets_block.known_format_datasets",
            "count": 3,
            "directed": False,
            "nodes": [3, 4, 5],
            "remap": True,
            "node_attributes": {
                "names": ["type"],
                "types": ["categorical"],
                "values": [["alpha", "beta", "gamma"]]
            },
            "edge_attributes": {
                "names": ["weight"],
                "types": ["continuous"],
                "values": [[0, 1]]
            },
            "labelings": {
                "binary": 2,
                "threeClasses": 3
            }
        }, f)
    (raw / 'labels').mkdir()
    with open(raw / 'labels' / 'binary', 'w') as f:
        json.dump({"0": 1, "1": 0, "2": 0}, f)
    with open(raw / 'labels' / 'threeClasses', 'w') as f:
        json.dump({"0": 0, "1": 1, "2": 2}, f)

    (raw / 'node_attributes').mkdir()
    with open(raw / 'node_attributes' / 'type', 'w') as f:
        json.dump([
            {"0": "alpha", "1": "beta", "2": "alpha"},
            {"0": "gamma", "1": "beta", "2": "gamma", "3": "gamma"},
            {"0": "beta", "1": "gamma", "2": "gamma", "3": "alpha", "4": "beta"}], f)

    # (raw / 'edge_attributes').mkdir()
    # with open(raw / 'edge_attributes' / 'weight', 'w') as f:
    #     json.dump([[0.1,0.1,0.1,0.2,0.2,0.2,],[0.1,0.1,0.1,0.1],[0.2,0.2,0.2,0.2]], f)


class DatasetsTest(unittest.TestCase):
    class UserApiDataset(Dataset):
        """ Generates 3 graphs with random features on the fly.
        """

        def __init__(self, root):
            super().__init__(root)

        @property
        def processed_file_names(self):
            return ''

        def process(self):
            pass

        def len(self) -> int:
            return 3

        def get(self, idx):
            x = torch.rand((3, 2))
            edge_index = tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
            y = torch.tensor([[0, 1, 1][idx]])

            return Data(x=x, edge_index=edge_index, y=y)

    def setUp(self) -> None:
        # Example of local user PTG dataset
        class UserLocalDataset(InMemoryDataset):
            def __init__(self, root, data_list, transform=None):
                self.data_list = data_list
                super().__init__(root, transform)
                # NOTE: it is important to define self.slices here, since it is used to calculate len()
                self.data, self.slices = torch.load(self.processed_paths[0])

            @property
            def processed_file_names(self):
                return 'data.pt'

            def process(self):
                torch.save(self.collate(self.data_list), self.processed_paths[0])

        self.UserLocalDataset = UserLocalDataset

        # DatasetsTest.UserApiDataset = UserApiDataset

    # def test_converted_local_ptg(self):
    #     """ """
    #     from datasets_block.datasets_manager import DatasetManager
    #     from dgl.data import BA2MotifDataset
    #     from torch_geometric.data import Data
    #
    #     def from_dgl(g, label):
    #         """ Converter from DGL graph by Misha S.
    #         """
    #         x = g.nodes[0].data['feat']
    #         for i in range(1, g.nodes().size(0)):
    #             x_i = g.nodes[i].data['feat']
    #             x = torch.cat((x, x_i), 0)
    #
    #         edge_index_tup = g.edges()
    #         t1 = edge_index_tup[0].unsqueeze(0)
    #         t2 = edge_index_tup[1].unsqueeze(0)
    #         edge_index = torch.cat((t1, t2), 0)
    #
    #         y = torch.argmax(label).unsqueeze(0)
    #
    #         return Data(x=x, edge_index=edge_index, y=y)
    #
    #     dgl_dataset = BA2MotifDataset()
    #     data_list = []
    #     for ix in range(len(dgl_dataset)):
    #         dgl_g, label = dgl_dataset[ix]
    #         ptg_data = from_dgl(dgl_g, label)
    #         data_list.append(ptg_data)
    #     ptg = self.UserLocalDataset(tmp_dir / 'test_dataset_converted_dgl', data_list)
    #
    #     gen_dataset = DatasetManager.register_torch_geometric_local(ptg, name='dgl_dataset')
    #     self.assertEqual(len(gen_dataset), len(dgl_dataset))
    #
    #     # Load
    #     gen_dataset = DatasetManager.get_by_config(gen_dataset.dataset_config)
    #     self.assertEqual(len(gen_dataset), len(dgl_dataset))

    def test_local_ptg(self):
        """ """
        x = tensor([[0, 0], [1, 0], [1, 0]])
        edge_index = tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        y = tensor([0, 1, 1])

        # Single
        data_list = [Data(x=x, edge_index=edge_index, y=y)]
        gen_dataset_s = LocalPTGDataset(data_list)
        self.assertEqual(len(gen_dataset_s), 1)

        # Multi
        data_list = [Data(x=x, edge_index=edge_index, y=tensor([0])),
                     Data(x=x, edge_index=edge_index, y=tensor([1]))]
        gen_dataset_m = LocalPTGDataset(data_list)
        self.assertEqual(len(gen_dataset_m), 2)

        # Load
        gen_dataset_s = DatasetManager.get_by_config(gen_dataset_s.dataset_config)
        self.assertEqual(len(gen_dataset_s.data), 3)

        gen_dataset_m = DatasetManager.get_by_config(gen_dataset_m.dataset_config)
        self.assertEqual(len(gen_dataset_m), 2)

    def test_custom_ij_single(self):
        """ """
        dc = DatasetConfig(('single', 'custom', 'test'))
        _create_single_ij(dc)

        # Load
        gen_dataset = KnownFormatDataset(dc)
        self.assertTrue(len(gen_dataset), 1)

        # Build
        dvc1 = DatasetVarConfig(
            features=FeatureConfig(node_attr=['a', 'b', 'c']),
            labeling='binary',
            dataset_ver_ind=0)
        gen_dataset.build(dvc1)
        self.assertTrue(gen_dataset.num_classes, 2)
        self.assertTrue(gen_dataset.num_node_features, 1 + 3 + 2)

        # Load built
        gen_dataset.build(dvc1)
        self.assertTrue(1)

        # Build another way
        dvc2 = DatasetVarConfig(
            features=FeatureConfig(node_struct=[FeatureConfig.one_hot],
                                   node_attr=['a', 'b', 'c']),
            labeling='threeClasses',
            dataset_ver_ind=0)
        gen_dataset.build(dvc2)
        self.assertTrue(gen_dataset.num_classes, 3)
        self.assertTrue(gen_dataset.num_node_features, 3 + 1)

    def test_custom_ij_multi(self):
        """ """
        dc = DatasetConfig(('multi', 'custom', 'test'))
        _create_multi_ij(dc)

        # Load
        gen_dataset = KnownFormatDataset(dc)
        self.assertTrue(len(gen_dataset), 3)

        # Build
        dvc1 = DatasetVarConfig(
            features=FeatureConfig(node_attr=['type']),
            labeling='binary',
            dataset_ver_ind=0)
        gen_dataset.build(dvc1)
        self.assertTrue(gen_dataset.num_classes, 2)
        self.assertTrue(gen_dataset.num_node_features, 3)

        # Load built
        gen_dataset.build(dvc1)
        self.assertTrue(gen_dataset.num_classes, 2)
        self.assertTrue(gen_dataset.num_node_features, 3)

    def test_custom_other_single(self):
        """ """
        from datasets_block.dataset_converter import DatasetConverter
        import networkx as nx

        g = nx.Graph()
        g.add_node(0, a=0.4, b=100)
        g.add_node(1, a=0.4, b=100)
        g.add_node(2, a=0.3, b=50)
        g.add_node(3, a=0.3, b=200)
        g.add_node(4, a=0.2, b=75)
        g.add_node(5, a=0.4, b=25)
        g.add_node(6, a=0.2, b=150)
        g.add_node(7, a=0.5, b=80)
        g.add_node(8, a=0.1, b=40)
        g.add_edge(0, 1, weight=5, type='big')
        g.add_edge(1, 2, weight=5, type='big')
        g.add_edge(1, 3, weight=3, type='medium')
        g.add_edge(1, 4, weight=4, type='small')
        g.add_edge(2, 5, weight=2, type='big')
        g.add_edge(2, 6, weight=6, type='big')
        g.add_edge(3, 4, weight=3, type='medium')
        g.add_edge(3, 7, weight=5, type='small')
        g.add_edge(4, 8, weight=4, type='big')
        g.add_edge(5, 6, weight=1, type='small')
        g.add_edge(6, 7, weight=5, type='small')
        g.add_edge(7, 8, weight=3, type='medium')

        node_labels = {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 1, 7: 0, 8: 0}

        true_ptg_data = networkx_to_ptg(g)

        # for format in ['.g6']:
        for format in DatasetConverter.supported_formats:
            print(f"Checking format {format}")

            name = f'test_{format}'
            dc = DatasetConfig(('single-graph', 'custom', name))

            # Write graph and attributes files
            root, files_paths = Declare.dataset_root_dir(dc)
            raw = root / 'raw'
            raw.mkdir(parents=True)
            DatasetConverter.networkx_to_format(g, format, raw, name=name,
                                                default_node_attr_value={'a': -1, 'b': -1},
                                                default_edge_attr_value={'weight': -1, 'type': -1})

            # Write info and labels
            with open(root / 'metainfo', 'w') as f:
                json.dump({
                    "name": name,
                    "format": format,
                    "count": 1,
                    "directed": False,
                    "nodes": [g.number_of_nodes()],
                    "remap": False,
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

            (raw / 'labels').mkdir()
            with open(raw / 'labels' / 'binary', 'w') as f:
                json.dump(node_labels, f)

            # Convert from the format
            gen_dataset = KnownFormatDataset(dc)

            dataset_var_config = DatasetVarConfig(
                features=FeatureConfig(node_attr=['a', 'b']),
                labeling='binary', dataset_ver_ind=0)
            gen_dataset.build(dataset_var_config)
            ptg_data = gen_dataset.data

            # Check features and edges coincide
            self.assertTrue(torch.equal(true_ptg_data.x.sort(dim=0)[0], ptg_data.x.sort(dim=0)[0]))
            sorted_edges1 = torch.sort(true_ptg_data.edge_index, dim=1)[0]
            sorted_edges2 = torch.sort(ptg_data.edge_index, dim=1)[0]
            self.assertTrue(torch.equal(sorted_edges1, sorted_edges2))
            # FIXME add it later when edge features are ready
            # self.assertTrue(torch.equal(true_ptg_data.edge_attr, ptg_data.edge_attr))

    def test_visible_part(self):
        # Create files
        dc = DatasetConfig(('single-graph', 'example'))
        dvc = DatasetVarConfig(
            features=FeatureConfig(node_attr=['a']), labeling='binary', dataset_ver_ind=0)
        _create_single2_ij(dc)
        single = DatasetManager.get_by_config(dc)
        single.build(dvc)

        dc = DatasetConfig(('multi-graph', 'test'))
        dvc = DatasetVarConfig(
            features=FeatureConfig(node_attr=['type']), labeling='binary', dataset_ver_ind=0)
        _create_multi_ij(dc)
        multi = DatasetManager.get_by_config(dc)
        multi.build(dvc)

        # TODO misha add hetero

        # Test that getting functions work
        for dataset in [single, multi]:
            dataset.set_visible_part({})
            dataset.set_visible_part({'center': 0})
            dataset.set_visible_part({'center': 0, 'depth': 2})
            dataset.visible_part.get_dataset_data()
            dataset.visible_part.get_dataset_var_data()

        # Test correctness
        single.set_visible_part({'center': 1, 'depth': 2})
        dd = single.visible_part.get_dataset_data()
        self.assertEqual(dd.edges, [
            [],
            [(0, 1), (2, 1), (4, 1), (3, 1)],
            [(6, 4), (2, 3), (3, 2), (5, 2), (7, 4)]])
        self.assertEqual(dd.nodes, [[1], [0, 2, 3, 4], [5, 6, 7]])
        self.assertEqual(dd.graphs, None)

        dvd = single.visible_part.get_dataset_var_data()
        self.assertEqual(dvd.labels, {1: 1, 0: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0})
        self.assertEqual(set(dvd.node_features.keys()), set(range(8)))

    def test_stats(self):
        """ Statistics
        """
        from datasets_block.dataset_stats import DatasetStats

        dc = DatasetConfig(('single-graph', 'test_stats'))
        dvc = DatasetVarConfig(
            features=FeatureConfig(node_attr=['a']), labeling='binary', dataset_ver_ind=0)
        _create_single2_ij(dc)
        single = DatasetManager.get_by_config(dc)
        single.build(dvc)

        dc = DatasetConfig(('multi-graph', 'test_stats'))
        dvc = DatasetVarConfig(
            features=FeatureConfig(node_attr=['type']), labeling='binary', dataset_ver_ind=0)
        _create_multi_ij(dc)
        multi = DatasetManager.get_by_config(dc)
        multi.build(dvc)

        single = DatasetStats(single)
        for stat in DatasetStats.all_stats:
            res = single.get(stat)
            print(stat, res)

        multi = DatasetStats(multi)
        for stat in DatasetStats.multi_stats:
            res = multi.get(stat)
            print(stat, res)

    def test_ptg_lib(self):
        """ NOTE: takes a lot of time
        """
        from data_structures.prefix_storage import TuplePrefixStorage
        from aux.utils import TORCH_GEOM_GRAPHS_PATH
        import traceback
        with open(TORCH_GEOM_GRAPHS_PATH, 'r') as f:
            ps = TuplePrefixStorage.from_json(f.read(), )

        errors = []
        for ix, (full_name, default_init_kwargs) in enumerate(ps):
            print(f"Checking {full_name} ({ix+1} of {len(ps)})")

            try:
                try:
                    # Downloads and processes for the first time
                    dc = DatasetConfig(tuple([LibPTGDataset.data_folder] + full_name))
                    default_init_kwargs = default_init_kwargs or {}
                    dataset = LibPTGDataset(dc, **default_init_kwargs)
                    self.assertTrue(1)
                except Exception as e:
                    print('\n\n')
                    print(f"ERROR at {full_name}:")
                    print(traceback.print_exc())
                    errors.append(dc)
                    continue

                try:
                    # Ensure metainfo and raw exist
                    path = Declare.dataset_root_dir(dc)[0]
                    self.assertTrue((path / 'raw').exists())
                    self.assertTrue((path / 'metainfo').exists())

                    # Read from file for second time
                    if default_init_kwargs:
                        dataset = LibPTGDataset(dc, **default_init_kwargs)
                    else:
                        DatasetManager.get_by_config(dc)
                    self.assertTrue(1)
                except Exception as e:
                    print('\n\n')
                    print(f"ERROR at load {dc}:")
                    print(traceback.print_exc())
                    errors.append(dc)

            # Remove
            finally:
                root_dir, files_paths = Declare.dataset_root_dir(dc)
                if root_dir.exists():
                    shutil.rmtree(root_dir)

        if len(errors) > 0:
            self.assertFalse(1)
            print(f"{len(errors)} Errors", '\n'.join(errors))

    def test_ptg_lib_with_params(self):
        """
        """
        import torch_geometric.transforms as T
        dataset_config_list = [
            [
                DatasetConfig(
                    (LibPTGDataset.data_folder, 'Homogeneous', 'InfectionDataset',
                     'BAGraph(num_nodes=30,num_edges=10),num_infected_nodes=10,max_path_length=3')),
                {
                    "graph_generator": "BAGraph",
                    "num_infected_nodes": 10,
                    "max_path_length": 3,
                    "graph_generator_kwargs": {"num_nodes": 30, "num_edges": 10}
                }
            ],
            [
                DatasetConfig(
                    (LibPTGDataset.data_folder, 'Homogeneous', 'FakeDataset',
                     'version1')),
                    {
                        "num_graphs": 2,
                        "avg_num_nodes": 200,
                        "avg_degree": 15,
                        "num_channels": 64,
                        "edge_dim": 0,
                        "num_classes": 10,
                        "task": "auto",
                    }
            ],
            [
                DatasetConfig(
                    (LibPTGDataset.data_folder, 'Homogeneous', 'MixHopSyntheticDataset',
                     '04')),
                    {
                        "homophily": 0.4,
                    }
            ],
            [
                # Requires transform to be applied
                DatasetConfig(
                    (LibPTGDataset.data_folder, 'Homogeneous', 'ExplainerDataset',
                     'BAGraph(num_nodes=300,num_edges=10),HouseMotif(),num_motifs=3)')),
                {
                    "graph_generator": "BAGraph",
                    "graph_generator_kwargs": {"num_nodes": 300, "num_edges": 10},
                    "motif_generator": "HouseMotif",
                    "motif_generator_kwargs": {},
                    "num_motifs": 3,
                    "transform": T.Constant()
                }
            ]
        ]

        import traceback
        errors = []
        for dc, params in dataset_config_list:
            print(f"Checking {dc}")

            try:
                try:
                    # Downloads and processes for the first time
                    dataset = LibPTGDataset(dc, **params)
                    self.assertTrue(1)
                except Exception as e:
                    print('\n\n')
                    print(f"ERROR at {dc}:")
                    print(traceback.print_exc())
                    errors.append(dc)
                    continue

                try:
                    # Read from file for second time
                    dataset = LibPTGDataset(dc, **params)
                    self.assertTrue(1)
                except Exception as e:
                    print('\n\n')
                    print(f"ERROR at load {dc}:")
                    print(traceback.print_exc())
                    errors.append(dc)

            # Remove
            finally:
                from aux.declaration import Declare
                root_dir, files_paths = Declare.dataset_root_dir(dc)
                if root_dir.exists():
                    shutil.rmtree(root_dir)

        if len(errors) > 0:
            self.assertFalse(1)
            print(f"{len(errors)} Errors", '\n'.join(errors))


if __name__ == '__main__':
    unittest.main()
