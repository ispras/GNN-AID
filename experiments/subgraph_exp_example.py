import torch

import warnings

from torch import device

from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from data_structures.configs import ModelModificationConfig, ConfigPattern
from base.datasets_processing import DatasetManager
from models_builder.models_zoo import model_configs_zoo


def test_SubgraphX():
    # my_device = device('cuda' if is_available() else 'cpu')
    my_device = device('cpu')

    full_name = None

    # full_name = (LibPTGDataset.data_folder, "Homogeneous", "TUDataset", "MUTAG")
    # full_name = ("single-graph", "custom", 'karate')
    full_name = (LibPTGDataset.data_folder, "Homogeneous", "Planetoid", "Cora")
    # full_name = ("Homogeneous", "TUDataset", 'PROTEINS')

    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=full_name,
        dataset_ver_ind=0
    )

    # dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
    #     full_name=("example", "single-graph", "example",),
    #     features=FeatureConfig(node_attr=['a', 'b']),
    #     labeling='threeClasses',
    #     dataset_ver_ind=0
    # )

    # dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
    #     # full_name=("example", "custom", "vk_samples", "vk2-ff40-N100000-A.1612175945",),
    #     full_name=("example", "custom", "vk_samples", "vk2-ff20-N10000-A.1611943634",),
    #     # full_name=("example", "custom", "vk_samples", "vk2-ff20-N1000-U.1612273925",),
    #     # features=('sex',),
    #     features={'attr': {
    #         # "('personal', 'political')": 'one_hot',
    #         # "('occupation', 'type')": 'one_hot', # Don't work now
    #         # "('relation',)": 'one_hot',
    #         # "('age',)": 'one_hot',
    #         "('sex',)": 'one_hot',
    #     }},
    #     # features=FeatureConfig(node_attr=['sex']),
    #     labeling='sex1',
    #     dataset_ver_ind=0
    # )

    # print(data.train_mask)

    gnn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')
    # gnn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn_lin')
    # gnn = model_configs_zoo(dataset=dataset, model_name='test_gnn')
    # gnn = model_configs_zoo(dataset=dataset, model_name='gin_gin_gin_lin_lin')
    # gnn = model_configs_zoo(dataset=dataset, model_name='gin_gin_gin_lin_lin_prot')

    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "batch": 100,
            "mask_features": [],
            "optimizer": {
                # "_config_class": "Config",
                "_class_name": "Adam",
                # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                # "_class_import_info": ["torch.optim"],
                "_config_kwargs": {},
            }
        }
    )
    # manager_config = ModelManagerConfig(**{
    #         "mask_features": [],
    #         "optimizer": {
    #             # "_config_class": "Config",
    #             "_class_name": "Adam",
    #             # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
    #             # "_class_import_info": ["torch.optim"],
    #             "_config_kwargs": {},
    #         }
    #     }
    # )

    # train_test_split = [0.8, 0.2]
    # train_test_split = [0.6, 0.4]
    steps_epochs = 200
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )

    # save_model_flag = False
    save_model_flag = True

    # data.x = data.x.float()
    gnn_model_manager.gnn.to(my_device)
    data = data.to(my_device)

    warnings.warn("Start training")
    dataset.train_test_split()

    try:
        raise FileNotFoundError()
        # gnn_model_manager.load_model_executor()
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
    node = 0

    # Explanation size
    max_nodes = 5

    # warnings.warn("Start SubgraphX")
    # explainer_init_config = ConfigPattern(
    #     _class_name="SubgraphX",
    #     _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
    #     _config_class="ExplainerInitConfig",
    #     _config_kwargs={
    #         # "class_name": "SubgraphX",
    #     }
    # )
    # explainer_run_config = ConfigPattern(
    #     _config_class="ExplainerRunConfig",
    #     _config_kwargs={
    #         "mode": "local",
    #         "kwargs": {
    #             "_class_name": "GNNExplainer(torch-geom)",
    #             "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
    #             "_config_class": "Config",
    #             "_config_kwargs": {
    #                 "element_idx": 0, "max_nodes": 5
    #             },
    #         }
    #     }
    # )
    # explainer_SubgraphX = FrameworkExplainersManager(
    #     init_config=explainer_init_config,
    #     dataset=dataset, gnn_manager=gnn_model_manager,
    #     explainer_name="SubgraphX",
    # )
    # explainer_SubgraphX.conduct_experiment(explainer_run_config)
    #
    #
    # return metric_loc


if __name__ == '__main__':
    test_SubgraphX()

    # metric_val = []
    #
    # for _ in range(10):
    #     metric_val.append(test_SubgraphX())
    #
    # print(metric_val)
    #
    # print(max(metric_val))
    # print(min(metric_val))
    # print(sum(metric_val) / len(metric_val))

    # import numpy as np
    # params = {
    #     "SubgraphX": [
    #         # name, label, type, def, possible, tip
    #         ["num_hops", "Hops", "int", None, {"min": 1, "special": [np.inf]}, "The number of hops to extract neighborhood of target node"],
    #         ["explain_graph", "Explain", "bool", True, None, "Whether to explain graph classification model"],
    #         ["rollout", "Rollout", "int", 20, {"min": 1}, "Number of iteration to get the prediction"],
    #         ["min_atoms", "Min atoms", "int", 5, {"min": 1}, "Number of atoms of the leaf node in search tree"],
    #         ["c_puct", "C-puct", "float", 10, {"min": 0}, "The hyperparameter which encourages the exploration"],
    #         ["expand_atoms", "Expand atoms", "int", 14, {"min": 1}, "The number of atoms to expand when extend the child nodes in the search tree"],
    #         ["high2low", "High to low", "bool", False, None, "Whether to expand children nodes from high degree to low degree when extend the child nodes in the search tree"],
    #         ["local_radius", "Local radius", "int", 4, {"min": 1}, "Number of local radius to calculate"],
    #         ["sample_num", "Samples", "int", 100, {"min": 1}, "Sampling time of monte carlo sampling approximation for mc_shapley"],
    #         ["reward_method", "Reward", "string", "mc_l_shapley", ["gnn_score","l_shapley","mc_shapley","mc_l_shapley","nc_mc_l_shapley"], "The command string to select the subgraph_building_method"],
    #         ["subgraph_building_method", "Subgraph building", "string", "zero_filling", ["zero_filling","split"], "The command string for different subgraph building method,  such as `zero_filling`, `split` (default: `zero_filling`)"],
    #         ["max_nodes", "Max nodes", "int", 5, {"min": 1}, None],
    #         # ["vis"			FALSE			],
    #     ]
    # }
    # import json
    # print(json.dumps(params))
