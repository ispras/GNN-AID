import warnings

import torch
from torch import device
from torch.cuda import is_available

from datasets.datasets_manager import DatasetManager
from models_builder.gnn_models import FrameworkGNNModelManager
from models_builder.gnn_constructor import GNNStructure
from explainers.gradcam.GradCAM import GradCAM, GradCAMOut
from aux.declaration import Declare
import matplotlib.pyplot as plt


def GradCAM_test():
    my_device = device('cuda' if is_available() else 'cpu')

    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=(LibPTGDataset.data_folder, "single-graph", "Planetoid", "Cora"),
        dataset_ver_ind=0)

    # print(dataset)
    # print(data)
    # print(data.y)
    # print(data.train_mask)
    # print(results_path)

    gcn2 = GNNStructure(
        # conv_classes=('SGConv', 'SGConv'),
        conv_classes=('GCNConv', 'GCNConv'),
        layers_sizes=(dataset.num_node_features, 16, dataset.num_classes),
        activations=('torch.relu', 'torch.nn.functional.log_softmax'),
        conv_kwargs={}
    )

    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gcn2,
        dataset_path=results_dataset_path,
        epochs=50,
        batch=10000,
        model_ver_ind=1,
        # train_mask_flag=True,
        # train_mask_flag=False,
        mask_features=[],
    )

    data.x = data.x.float()
    data = data.to(my_device)

    warnings.warn("Start training")
    try:
        gnn_model_manager.load_model()
    except FileNotFoundError:
        dataset.train_test_split()

        train_test_split_path = gnn_model_manager.train_model(gen_dataset=dataset)

        # train_test_split_path = gnn_model_manager.train_test_split_path()
        dataset.save_train_test_mask(train_test_split_path)
        train_mask, val_mask, test_mask = torch.load(train_test_split_path / 'train_test_split')[:]
        data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    # except FileNotFoundError:
    #     pass
    warnings.warn("Training was  successful")

    explainer = GradCAM(gnn_model_manager.gnn)

    node_idx = torch.tensor(1709)
    preds = gnn_model_manager.gnn(data.x, data.edge_index)
    pred = int(preds[node_idx].argmax())
    print("predicted class", pred)
    print("real class", int(data.y[node_idx]))

    explainer_result_file_path = Declare.explanation_file_path(

        models_path=gnn_model_manager.model_path_info(),
        explainer_name='GradCAM',
        explain_node=int(node_idx))

    data.x.requires_grad = True
    edge_mask, hard_edge_mask, related_preds = explainer(data.x, data.edge_index, node_idx=node_idx,
                                                         num_classes=dataset.num_classes)

    out = GradCAMOut(edge_mask[pred], edge_mask[pred], None, data)
    out.save(explainer_result_file_path)

    # x, G = ExplainerBase(model).visualize_graph(2377, data.edge_index, edge_mask[pred],
    # data.y, nolabel=False)
    # plt.show()

    from visualization.visualizer import visualize_subgraph
    ax, G = visualize_subgraph(model=gnn_model_manager.gnn, node_idx=int(node_idx),
                               path=explainer_result_file_path, edge_index=data.edge_index,
                               y=data.y, edge_threshold=0.5, node_threshold=0.3433)
    plt.show()


if __name__ == '__main__':
    GradCAM_test()
