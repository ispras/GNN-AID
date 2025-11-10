import warnings

import torch
from torch import device
from torch.cuda import is_available

from data_structures.configs import ConfigPattern, ModelManagerConfig, ModelModificationConfig
from aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH
from explainers.explainers_manager import FrameworkExplainersManager
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from datasets.datasets_manager import DatasetManager
from models_builder.models_zoo import model_configs_zoo


# from pytorch_model_summary import summary


# from visualization.plotutils import draw_vk, draw_cora, draw


def test_Zorro(save_nan=True):
    my_device = device('cuda' if is_available() else 'cpu')
    my_device = device('cpu')

    dataset_ifo = {}

    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=(LibPTGDataset.data_folder, "Homogeneous", "Planetoid", "Cora"),
        dataset_ver_ind=0)

    gcn2 = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')

    gnn_model_manager_config = ModelManagerConfig(**{
        "mask_features": [],
        "optimizer": {
            "_class_name": "Adam",
            "_config_kwargs": {},
        }
    })

    steps_epochs = 200
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gcn2,
        dataset_path=results_dataset_path,
        manager_config=gnn_model_manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )

    data.x = data.x.float()
    gnn_model_manager.gnn.to(my_device)
    data = data.to(my_device)

    # save_model_flag = False
    save_model_flag = True

    warnings.warn("Start training")
    try:
        gnn_model_manager.load_model_executor()
    except FileNotFoundError:
        gnn_model_manager.epochs = gnn_model_manager.modification.epochs = 0
        train_test_split_path = gnn_model_manager.train_model(gen_dataset=dataset, steps=steps_epochs,
                                                              save_model_flag=save_model_flag,
                                                              metrics=[Metric("F1", mask='train', average=None)])

        if train_test_split_path is not None:
            dataset.save_train_test_mask(train_test_split_path)
            train_mask, val_mask, test_mask, train_test_sizes = torch.load(train_test_split_path / 'train_test_split')[
                                                                :]
            dataset.train_mask, dataset.val_mask, dataset.test_mask = train_mask, val_mask, test_mask
            data.percent_train_class, data.percent_test_class = train_test_sizes

    warnings.warn("Training was successful")

    metric_loc = gnn_model_manager.evaluate_model(
        gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro')])
    print(metric_loc)

    # Explain node 10
    node = 10

    warnings.warn("Start Zorro")
    explainer_init_config = ConfigPattern(
        _class_name="Zorro",
        _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
        _config_class="ExplainerInitConfig",
        _config_kwargs={
        }
    )
    explainer_run_config = ConfigPattern(
        _config_class="ExplainerRunConfig",
        _config_kwargs={
            "mode": "local",
            "kwargs": {
                "_class_name": "Zorro",
                "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                "_config_class": "Config",
                "_config_kwargs": {

                },
            }
        }
    )
    explainer_Zorro = FrameworkExplainersManager(
        init_config=explainer_init_config,
        dataset=dataset, gnn_manager=gnn_model_manager,
        explainer_name='Zorro',
    )
    explainer_Zorro.conduct_experiment(explainer_run_config)


# def test_Vk():
#     my_device = device('cuda' if is_available() else 'cpu')
#
#     dataset, data, results_path = DatasetManager.get_by_full_name(
#         ("vk_samples", "vk2-ff20-N10000-A.1611943634",))
#
#     gcn2 = GNNStructure(
#         # conv_classes=('SGConv', 'SGConv'),
#         conv_classes=('GCNConv', 'GCNConv'),
#         layers_sizes=(dataset.num_node_features, 16, dataset.num_classes),
#         activations=('torch.relu', 'torch.nn.functional.log_softmax'),
#         conv_kwargs={}
#     )
#
#     gnn_model_manager = FrameworkGNNModelManager(
#         gnn=gcn2,
#         dataset_path=results_path,
#         epochs=50,
#         batch=10000,
#         model_ver_ind=1,
#         # train_mask_flag=True,
#         # train_mask_flag=False,
#         mask_features=[],
#
#     )
#
#     data.x = data.x.float()
#     gnn_model_manager.gnn.to(my_device)
#     data = data.to(my_device)
#
#     warnings.warn("Start training")
#     try:
#         gnn_model_manager.load_model()
#     except FileNotFoundError:
#         gnn_model_manager.train_model(gen_dataset=dataset)
#     warnings.warn("Training was  successful")
#
#     explainer_loc = Zorro(gnn_model_manager.gnn, my_device)
#
#     # Same as the Zorro \tau=0.98 in the paper
#     tau = .03
#     # Explain node 10
#     node = 1358
#     # only retrieve 1 explanation
#     recursion_depth = 1
#
#     logits = gnn_model_manager.gnn(data.x, data.edge_index)
#     prediction = logits[node].argmax(-1).item()
#     print("Prediction: ", prediction)
#
#     explanation = explainer_loc.explain_node(node, data.x, data.edge_index,
#                                              tau=tau,
#                                              recursion_depth=recursion_depth)
#
#     selected_nodes, selected_features, executed_selections = explanation[0]
#
#     print(explanation)
#
#     print(selected_nodes)
#
#     print(selected_features.shape)
#
#     print(selected_features.sum())
#
#     print(executed_selections)
#
#     # draw_vk(node, selected_nodes)
#
#     if data.y[node].item() == prediction:
#         print("CORRECT PREDICTION")
#     else:
#         print("INCORRECT PREDICTION")


if __name__ == '__main__':
    # t0 = time.clock()
    test_Zorro()

    # print(time.clock() - t0)
    # test_Vk()

    # import numpy as np
    # params = {
    #     "Zorro": [
    #         # name, label, type, def, possible, tip
    #         ["greedy",              "Greedy",               "bool",  True,   None,                            "?"],
    #         ["add_noise",           "Noise",                "bool",  False,  None,                            "?"],
    #         ["samples",             "Samples",              "int",   100,    {"min": 1},                      "?"],
    #         ["tau",                 "tau",                  "float", 0.15,   {"min": 0},                      "?"],
    #         ["recursion_depth",     "Recursion depth",      "int",   np.inf, {"min": 1, "special": [np.inf]}, "?"],
    #         ["save_initial_improve","Save initial improve", "bool",  False,  {"min": 1},                      "?"],
    #     ]
    # }
    # import json
    # print(json.dumps(params))
