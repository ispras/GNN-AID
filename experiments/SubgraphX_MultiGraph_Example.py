import os
import os.path as osp

import torch
from torch_geometric.data import download_url, extract_zip

from dig.xgraph.dataset import SynGraphDataset
from torch_geometric.datasets import TUDataset
from dig.xgraph.models import *
from dig.xgraph.utils.compatibility import compatible_state_dict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = TUDataset(root='./data/torch-geometric', name='MUTAG')

graph_idx = 5

data = dataset[graph_idx]

data.x = data.x.float()

dim_node = dataset.num_node_features

#dim_edge = dataset.num_edge_features
num_classes = dataset.num_classes

#model = GCN_2l(model_level='node', dim_node=dim_node, dim_hidden=300, num_classes=num_classes)
model = GIN_2l(model_level='graph', dim_node=dim_node, dim_hidden=128, num_classes=num_classes)

model.to(device)

from dig.xgraph.method import SubgraphX

explainer = SubgraphX(model, num_classes=True, device=device,
                      explain_graph=True, reward_method='mc_l_shapley')


from dig.xgraph.method.subgraphx import PlotUtils
# from dig.xgraph.method.subgraphx import find_closest_node_result

# Visualization
max_nodes = 5
print(f'explain graph {graph_idx}')
data.to(device)
logits = model(data.x, data.edge_index)
prediction = logits.argmax(-1).item()

_, explanation_results, related_preds = explainer(data.x, data.edge_index, max_nodes=max_nodes) #we shouldn't pass here node_idx!

#explanation_results = explanation_results[prediction]
explanation_results = explainer.read_from_MCTSInfo_list(explanation_results)
explanation_results = explanation_results[0]

plotutils = PlotUtils(dataset_name='mutag', is_show=True)
explainer.visualization(explanation_results,
                        max_nodes=max_nodes,
                        plot_utils=plotutils,
                        y=data.y)

max_nodes = 5

