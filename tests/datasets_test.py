import collections.abc

collections.Callable = collections.abc.Callable

import signal
import unittest
import shutil
import torch
from torch import tensor
from torch_geometric.data import InMemoryDataset, Data, Dataset

# Monkey patch GRAPHS_DIR - before other imports
from aux import utils
if not str(utils.GRAPHS_DIR).endswith("__DatasetsTest_tmp"):
    tmp_dir = utils.GRAPHS_DIR.parent / (utils.GRAPHS_DIR.name + "__DatasetsTest_tmp")
    utils.GRAPHS_DIR = tmp_dir
else:
    tmp_dir = utils.GRAPHS_DIR

from base.dataset_converter import networkx_to_ptg


def my_ctrlc_handler(signal, frame):
    print('my_ctrlc_handler', tmp_dir, tmp_dir.exists())
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, my_ctrlc_handler)


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

    @classmethod
    def tearDownClass(cls) -> None:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    def test_converted_local_ptg(self):
        """ """
        from base.datasets_processing import DatasetManager
        from dgl.data import BA2MotifDataset
        from torch_geometric.data import Data

        def from_dgl(g, label):
            """ Converter from DGL graph by Misha S.
            """
            x = g.nodes[0].data['feat']
            for i in range(1, g.nodes().size(0)):
                x_i = g.nodes[i].data['feat']
                x = torch.cat((x, x_i), 0)

            edge_index_tup = g.edges()
            t1 = edge_index_tup[0].unsqueeze(0)
            t2 = edge_index_tup[1].unsqueeze(0)
            edge_index = torch.cat((t1, t2), 0)

            y = torch.argmax(label).unsqueeze(0)

            return Data(x=x, edge_index=edge_index, y=y)

        dgl_dataset = BA2MotifDataset()
        data_list = []
        for ix in range(len(dgl_dataset)):
            dgl_g, label = dgl_dataset[ix]
            ptg_data = from_dgl(dgl_g, label)
            data_list.append(ptg_data)
        ptg = self.UserLocalDataset(tmp_dir / 'test_dataset_converted_dgl', data_list)

        gen_dataset = DatasetManager.register_torch_geometric_local(ptg, name='dgl_dataset')
        self.assertEqual(len(gen_dataset), len(dgl_dataset))

        # Load
        gen_dataset = DatasetManager.get_by_config(gen_dataset.dataset_config)
        self.assertEqual(len(gen_dataset), len(dgl_dataset))

    def test_local_ptg(self):
        """ """
        from base.datasets_processing import DatasetManager
        x = tensor([[0, 0], [1, 0], [1, 0]])
        edge_index = tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        y = tensor([0, 1, 1])

        # Single
        data_list = [Data(x=x, edge_index=edge_index, y=y)]
        dataset = self.UserLocalDataset(tmp_dir / 'test_dataset_single', data_list)
        gen_dataset_s = DatasetManager.register_torch_geometric_local(dataset)
        self.assertEqual(len(gen_dataset_s), 1)

        # Multi
        data_list = [Data(x=x, edge_index=edge_index, y=tensor([0])),
                     Data(x=x, edge_index=edge_index, y=tensor([1]))]
        dataset = self.UserLocalDataset(tmp_dir / 'test_dataset_multi', data_list)
        gen_dataset_m = DatasetManager.register_torch_geometric_local(dataset)
        self.assertEqual(len(gen_dataset_m), 2)

        # Load
        gen_dataset_s = DatasetManager.get_by_config(gen_dataset_s.dataset_config)
        self.assertEqual(len(gen_dataset_s), 1)

        gen_dataset_m = DatasetManager.get_by_config(gen_dataset_m.dataset_config)
        self.assertEqual(len(gen_dataset_m), 2)

    def test_api_ptg(self):
        """ """
        from base.datasets_processing import DatasetManager

        # Should be globally visible to be visible for import
        gen_dataset = DatasetManager.register_torch_geometric_api(
            DATASET_TO_EXPORT, name='api_random_features',
            obj_name='DATASET_TO_EXPORT')
        self.assertEqual(len(gen_dataset), 3)

        # Load
        gen_dataset = DatasetManager.get_by_config(gen_dataset.dataset_config)
        self.assertEqual(len(gen_dataset), 3)

    def test_custom_ij_single(self):
        """ """
        from aux.configs import DatasetVarConfig
        from aux.configs import DatasetConfig
        from aux.declaration import Declare
        from base.custom_datasets import CustomDataset
        import json
        dc = DatasetConfig('single', 'custom', 'test')

        # Create files
        root, files_paths = Declare.dataset_root_dir(dc)
        raw = root / 'raw'
        raw.mkdir(parents=True)
        with open(raw / 'test.ij', 'w') as f:
            f.write("10 11\n")
            f.write("10 12\n")
        with open(raw / '.info', 'w') as f:
            json.dump({
                "count": 1,
                "directed": False,
                "nodes": [3],
                "remap": True,
                "node_attributes": {
                    "names": ["a", "b", "c"],
                    "types": ["continuous", "categorical", "other"],
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
        (raw / 'test.labels').mkdir()
        with open(raw / 'test.labels' / 'binary', 'w') as f:
            json.dump({"10": 0, "11": 1, "12": 1}, f)
        with open(raw / 'test.labels' / 'threeClasses', 'w') as f:
            json.dump({"10": 0, "11": 1, "12": 2}, f)

        (raw / 'test.node_attributes').mkdir()
        with open(raw / 'test.node_attributes' / 'a', 'w') as f:
            json.dump({"10": 0.0, "11": 0.1, "12": 0.2}, f)
        with open(raw / 'test.node_attributes' / 'b', 'w') as f:
            json.dump({"10": "A", "11": "B", "12": "C"}, f)
        with open(raw / 'test.node_attributes' / 'c', 'w') as f:
            json.dump({"10": [0.3, -0.2], "11": [0, 0], "12": [1e5, 0]}, f)

        # (raw / 'test.edge_attributes').mkdir()
        # with open(raw / 'test.edge_attributes' / 'weight', 'w') as f:
        #     json.dump({"10": 0.0, "11": 0.1, "12": 0.2}, f)

        # Load
        gen_dataset = CustomDataset(dc)
        self.assertTrue(len(gen_dataset), 1)

        # Build
        dvc1 = DatasetVarConfig(
            features={'attr': {'a': 'as_is', 'b': 'one_hot', 'c': 'as_is'}},
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
            features={'str_g': 'one_hot', 'attr': {'a': 'as_is'}},
            labeling='threeClasses',
            dataset_ver_ind=0)
        gen_dataset.build(dvc2)
        self.assertTrue(gen_dataset.num_classes, 3)
        self.assertTrue(gen_dataset.num_node_features, 3 + 1)

    def test_custom_ij_multi(self):
        """ """
        from aux.configs import DatasetVarConfig
        from aux.configs import DatasetConfig
        from aux.declaration import Declare
        from base.custom_datasets import CustomDataset
        import json
        dc = DatasetConfig('multi', 'custom', 'test')

        # Create files
        root, files_paths = Declare.dataset_root_dir(dc)
        raw = root / 'raw'
        raw.mkdir(parents=True)
        with open(raw / 'test.ij', 'w') as f:
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
        with open(raw / 'test.edge_index', 'w') as f:
            f.write("[3, 7, 11]")
        with open(raw / '.info', 'w') as f:
            json.dump({
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
        (raw / 'test.labels').mkdir()
        with open(raw / 'test.labels' / 'binary', 'w') as f:
            json.dump({"0": 1, "1": 0, "2": 0}, f)
        with open(raw / 'test.labels' / 'threeClasses', 'w') as f:
            json.dump({"0": 0, "1": 1, "2": 2}, f)

        (raw / 'test.node_attributes').mkdir()
        with open(raw / 'test.node_attributes' / 'type', 'w') as f:
            json.dump([
                {"0": "alpha", "1": "beta", "2": "alpha"},
                {"0": "gamma", "1": "beta", "2": "gamma", "3": "gamma"},
                {"0": "beta", "1": "gamma", "2": "gamma", "3": "alpha", "4": "beta"}], f)

        # (raw / 'test.edge_attributes').mkdir()
        # with open(raw / 'test.edge_attributes' / 'weight', 'w') as f:
        #     json.dump([[0.1,0.1,0.1,0.2,0.2,0.2,],[0.1,0.1,0.1,0.1],[0.2,0.2,0.2,0.2]], f)

        # Load
        gen_dataset = CustomDataset(dc)
        self.assertTrue(len(gen_dataset), 3)

        # Build
        dvc1 = DatasetVarConfig(
            features={'attr': {'type': 'as_is'}},
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
        from aux.configs import DatasetVarConfig
        from aux.configs import DatasetConfig
        from aux.declaration import Declare
        from base.custom_datasets import CustomDataset
        from base.dataset_converter import DatasetConverter
        from base.datasets_processing import DatasetManager
        import json
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
            dc = DatasetConfig('single-graph', 'custom', name)

            # Write graph and attributes files
            root, files_paths = Declare.dataset_root_dir(dc)
            raw = root / 'raw'
            raw.mkdir(parents=True)
            DatasetConverter.networkx_to_format(g, format, raw, name=name,
                                                default_node_attr_value={'a': -1, 'b': -1},
                                                default_edge_attr_value={'weight': -1, 'type': -1})

            # Write info and labels
            with open(raw / '.info', 'w') as f:
                json.dump({
                    "name": name,
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

            (raw / f'{name}.labels').mkdir()
            with open(raw / f'{name}.labels' / 'binary', 'w') as f:
                json.dump(node_labels, f)

            # Convert from the format
            gen_dataset = DatasetManager.register_custom(dc, format)

            dataset_var_config = DatasetVarConfig(
                features={'attr': {'a': 'as_is', 'b': 'as_is'}}, labeling='binary', dataset_ver_ind=0)
            gen_dataset.build(dataset_var_config)
            ptg_data = gen_dataset.data

            # Check features and edges coincide
            self.assertTrue(torch.equal(true_ptg_data.x.sort(dim=0)[0], ptg_data.x.sort(dim=0)[0]))
            sorted_edges1 = torch.sort(true_ptg_data.edge_index, dim=1)[0]
            sorted_edges2 = torch.sort(ptg_data.edge_index, dim=1)[0]
            self.assertTrue(torch.equal(sorted_edges1, sorted_edges2))
            # FIXME add it later when edge features are ready
            # self.assertTrue(torch.equal(true_ptg_data.edge_attr, ptg_data.edge_attr))

    def test_ptg_lib(self):
        """ NOTE: takes a lot of time
        """
        from aux.prefix_storage import PrefixStorage
        from aux.utils import TORCH_GEOM_GRAPHS_PATH
        import traceback
        from base.datasets_processing import DatasetManager
        from aux.configs import DatasetConfig
        with open(TORCH_GEOM_GRAPHS_PATH, 'r') as f:
            ps = PrefixStorage.from_json(f.read())

        for full_name in ps:
            dc = DatasetConfig.from_full_name(full_name)
            print(f"Checking {dc}")

            try:
                try:
                    # Downloads and processes for the first time
                    DatasetManager.get_by_config(dc)
                    self.assertTrue(1)
                except Exception as e:
                    print('\n\n')
                    print(f"ERROR at download {dc}:")
                    print(traceback.print_exc())
                    continue

                try:
                    # Read from file for second time
                    DatasetManager.get_by_config(dc)
                    self.assertTrue(1)
                except Exception as e:
                    print('\n\n')
                    print(f"ERROR at load {dc}:")
                    print(traceback.print_exc())

            # Remove
            finally:
                from aux.declaration import Declare
                root_dir, files_paths = Declare.dataset_root_dir(dc)
                if root_dir.exists():
                    shutil.rmtree(root_dir)


DATASET_TO_EXPORT = DatasetsTest.UserApiDataset(tmp_dir / 'test_dataset_api')

if __name__ == '__main__':
    unittest.main()
