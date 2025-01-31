import torch
import numpy as np

from defense.evasion_defense import EvasionDefender

from defense.poison_defense import PoisonDefender

# class JaccardDefender(EvasionDefender):
#     name = 'JaccardDefender'
#
#     def __init__(self, threshold, **kwargs):
#         super().__init__()
#         self.thrsh = threshold
#
#     def pre_batch(self, model_manager, batch, **kwargs):
#         # TODO need to check whether features binary or not. Consistency required - Cora has 'unknown' features e.g.
#         #self.drop_edges(batch)
#         edge_index = batch.edge_index.tolist()
#         new_edge_mask = torch.zeros_like(batch.edge_index)
#         for i in range(len(edge_index)):
#             if self.jaccard_index(batch.x, edge_index[0][i], edge_index[1][i]) < self.thrsh:
#                 new_edge_mask[0,i] = True
#                 new_edge_mask[1,i] = True
#         batch.edge_index = batch.edge_index[new_edge_mask]
#
#     def jaccard_index(self, x, u, v):
#         im1 = x[u,:].numpy().astype(bool)
#         im2 = x[v,:].numpy().astype(bool)
#         intersection = np.logical_and(im1, im2)
#         union = np.logical_or(im1, im2)
#         return intersection.sum() / float(union.sum())
#
#     def post_batch(self, **kwargs):
#         pass

    # def drop_edges(self, batch):
    #     print("KEK")


class JaccardDefender(PoisonDefender):
    name = 'JaccardDefender'

    def __init__(self, threshold, **kwargs):
        super().__init__()
        self.thrsh = threshold

    def defense(self, gen_dataset, **kwargs):
        # TODO need to check whether features binary or not. Consistency required - Cora has 'unknown' features e.g.
        #self.drop_edges(batch)
        edge_index = gen_dataset.dataset.data.edge_index.tolist()
        #new_edge_mask = torch.zeros_like(gen_dataset.dataset.data.edge_index).bool()
        new_edge_index = [[],[]]
        for i in range(len(edge_index[0])):
            if self.jaccard_index(gen_dataset.dataset.data.x, edge_index[0][i], edge_index[1][i]) > self.thrsh:
                # new_edge_mask[0,i] = True
                # new_edge_mask[1,i] = True
                new_edge_index[0].append(edge_index[0][i])
                new_edge_index[1].append(edge_index[1][i])
        # gen_dataset.dataset.data.edge_index *= new_edge_mask.float()
        gen_dataset.dataset.data.edge_index = torch.tensor(new_edge_index).long()
        return gen_dataset

    def jaccard_index(self, x, u, v):
        im1 = x[u,:].numpy().astype(bool)
        im2 = x[v,:].numpy().astype(bool)
        intersection = np.logical_and(im1, im2)
        union = np.logical_or(im1, im2)
        return intersection.sum() / float(union.sum())