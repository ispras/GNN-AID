import torch
from torch import tensor
from torch_geometric.data import InMemoryDataset, Data, Dataset

from base.datasets_processing import DatasetManager


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


def local():
    x = tensor([[0, 0], [1, 0], [1, 0]])
    edge_index = tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    y = tensor([0, 1, 1])

    # Single
    data_list = [Data(x=x, edge_index=edge_index, y=y)]
    dataset = UserLocalDataset('test_dataset_single', data_list)
    gen_dataset = DatasetManager.register_torch_geometric_local(dataset)
    print("len =", len(gen_dataset))

    # Multi
    data_list = [Data(x=x, edge_index=edge_index, y=tensor([0])),
                 Data(x=x, edge_index=edge_index, y=tensor([1]))]
    dataset = UserLocalDataset('test_dataset_multi', data_list)
    gen_dataset = DatasetManager.register_torch_geometric_local(dataset)
    print("len =", len(gen_dataset))


def converted_local():
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
    ptg = UserLocalDataset('test_dataset_converted_dgl', data_list)

    gen_dataset = DatasetManager.register_torch_geometric_local(ptg, name='dgl_dataset')
    print("len =", len(gen_dataset))


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


# Should be global to be visible for import
DATASET_TO_EXPORT = UserApiDataset('test_dataset_api')


def api():
    gen_dataset = DatasetManager.register_torch_geometric_api(
        DATASET_TO_EXPORT, name='api_random_features', obj_name='DATASET_TO_EXPORT')
    print("len =", len(gen_dataset))


# This is to register user-defined dataset classes
def simgnn():
    # from src.aux.utils import root_dir
    # from external.simgnn_for_mdr.simgnn.data import random_data, src_dst_data
    from simgnn.data import random_data, src_dst_data

    gen_dataset = DatasetManager.register_torch_geometric_api(
        random_data, name='simgnn_random_data', obj_name='random_data')
    print("len =", len(gen_dataset))
    gen_dataset = DatasetManager.register_torch_geometric_api(
        src_dst_data, name='simgnn_src_dst_data', obj_name='src_dst_data')
    print("len =", len(gen_dataset))


def nx_to_ptg_converter():
    from aux.utils import GRAPHS_DIR
    from base.dataset_converter import networkx_to_ptg
    from base.datasets_processing import DatasetManager
    import networkx as nx

    nx_path = GRAPHS_DIR / 'networkx-graphs' / 'input' / 'reply_graph.edgelist'
    nx_graph = nx.read_edgelist(nx_path)
    nx_graph = nx.to_undirected(nx_graph)
    ptg_graph = networkx_to_ptg(nx_graph)
    if ptg_graph.x is None:
        ptg_graph.x = torch.zeros((ptg_graph.num_nodes, 1))
    if ptg_graph.y is None:
        ptg_graph.y = torch.zeros(ptg_graph.num_nodes)
        ptg_graph.y[0] = 1
    ptg_dataset = UserLocalDataset('test_dataset_single', [ptg_graph])
    gen_dataset = DatasetManager.register_torch_geometric_local(ptg_dataset)
    print(len(gen_dataset))


if __name__ == '__main__':

    # local()
    # converted_local()
    # api()
    # simgnn()

    nx_to_ptg_converter()
