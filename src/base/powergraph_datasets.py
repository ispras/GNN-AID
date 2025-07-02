import os.path as osp
from typing import Union, List, Tuple

import torch
import os
import mat73
import numpy as np
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from base.datasets_processing import GeneralDataset
from data_structures.configs import DatasetConfig

device = 'cpu'  # TODO remove it


def index_edgeorder(edge_order):
    return torch.tensor(edge_order["bList"] - 1)


class ConfigPatter:  # TODO for what? Ask Kirill
    pass


class PowerGrid(GeneralDataset, InMemoryDataset):
    def __init__(
            self,
            dataset_config: Union[ConfigPatter, DatasetConfig],
            datatype='binary'
    ):
        """
        Args:
            dataset_config: DatasetConfig dict from frontend
        """
        #  TODO Georgii temporary only for AAAI experiments
        self.datatype = datatype

        GeneralDataset.__init__(self, dataset_config=dataset_config)
        InMemoryDataset.__init__(self, root=self.root_dir)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ['Bf.mat',
                'blist.mat',
                'Ef.mat',
                "exp.mat",
                'of_bi.mat',
                'of_mc.mat',
                'of_reg.mat']

    @property
    def processed_file_names(
            self
    ) -> str:
        return 'data.pt'

    def build(
            self,
            dataset_var_config: dict = None
    ) -> None:
        if dataset_var_config == self.dataset_var_config:
            # PTG is cached
            return

        self.dataset_var_data = None
        self.stats.update_var_config()
        self.dataset_var_config = dataset_var_config

    def process(self):

        # function that deletes row
        def th_delete(tensor, indices):
            mask = torch.ones(tensor.size(), dtype=torch.bool)
            mask[indices] = False
            return tensor[mask]

        # load branch list also called edge order or edge index
        path = os.path.join(self.raw_dir, 'blist.mat')
        edge_order = mat73.loadmat(path)
        edge_order = torch.tensor(edge_order["bList"] - 1)
        # load output binary classification labels
        path = os.path.join(self.raw_dir, 'of_bi.mat')
        of_bi = mat73.loadmat(path)
        # load output binary regression labels
        path = os.path.join(self.raw_dir, 'of_reg.mat')
        of_reg = mat73.loadmat(path)
        # load output mc labels
        path = os.path.join(self.raw_dir, 'of_mc.mat')
        of_mc = mat73.loadmat(path)
        # load output node feature matrix
        path = os.path.join(self.raw_dir, 'Bf.mat')
        node_f = mat73.loadmat(path)
        # load output edge feature matrix
        path = os.path.join(self.raw_dir, 'Ef.mat')
        edge_f = mat73.loadmat(path)
        # load explanations
        path = os.path.join(self.raw_dir, "exp.mat")
        exp = mat73.loadmat(path)

        node_f = node_f['B_f_tot']
        edge_f = edge_f['E_f_post']
        of_bi = of_bi['output_features']
        of_mc = of_mc['category']
        of_reg = of_reg['dns_MW']
        exp_mask = exp["explainations"]

        data_list = []
        index = 0
        # MAIN data processing loop
        for i in range(len(node_f)):
            # node feat
            x = torch.tensor(node_f[i][0], dtype=torch.float32).reshape([-1, 3]).to(device)
            # edge feat
            f = torch.tensor(edge_f[i][0], dtype=torch.float32)
            e_mask = torch.zeros(len(edge_f[i][0]), 1)
            if exp_mask[i][0] is None:  # .all() == 0:
                e_mask = e_mask
            else:
                e_mask[exp_mask[i][0].astype('int')-1] = 1
            # contigency lists, finds where do we have contigencies from the .mat edge feature matrices
            # ( if a line is part of the contigency list all egde features are set 0)
            cont = [j for j in range(len(f)) if np.all(np.array(f[j])) == 0]
            e_mask_post = th_delete(e_mask, cont)
            e_mask_post = torch.cat((e_mask_post, e_mask_post), 0).to(device)
            # remove edge features of the associated line
            f_tot = th_delete(f, cont).reshape([-1, 4]).type(torch.float32)
            # concat the post-contigency edge feature matrix to take into account the reversed edges
            f_totw = torch.cat((f_tot, f_tot), 0).to(device)
            # remove failed lines from branch list
            edge_iw = th_delete(edge_order, cont).reshape(-1, 2).type(torch.long)
            # flip branch list
            edge_iwr = torch.fliplr(edge_iw)
            #  and concat the non flipped and flipped branch list
            edge_iw = torch.cat((edge_iw, edge_iwr), 0)
            edge_iw = edge_iw.t().contiguous().to(device)

            if self.datatype.lower() == 'binary':
                ydata = torch.tensor(of_bi[i][0], dtype=torch.float, device=device).view(1, -1)
            if self.datatype.lower() == 'regression':
                ydata = torch.tensor(of_reg[i], dtype=torch.float, device=device).view(1, -1)
            if self.datatype.lower() == 'multiclass':
                #do argmax
                ydata = torch.tensor(np.argmax(of_mc[i][0]), dtype=torch.float, device=device).view(1, -1)
                # ydata = torch.tensor(of_mc[i][0], dtype=torch.int, device=device).view(1, -1)

            # Fill Data object, 1 Data object -> 1 graph

            data = Data(x=x, edge_index=edge_iw, edge_attr=f_totw, y=ydata.to(torch.float), edge_mask=e_mask_post, idx=index)
            data_list.append(data)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

        torch.save(self.collate(data_list), self.processed_paths[0])

