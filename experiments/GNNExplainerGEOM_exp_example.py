import warnings

import torch
from torch import device
from torch.cuda import is_available

from aux.configs import ModelManagerConfig, ModelModificationConfig, ConfigPattern
from aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH
from explainers.explainers_manager import FrameworkExplainersManager

from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from base.datasets_processing import DatasetManager
from models_builder.models_zoo import model_configs_zoo


def geom_GNNExplainer_test():
    my_device = device('cuda' if is_available() else 'cpu')
    my_device = device('cpu')

    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=("single-graph", "Planetoid", 'Cora'),
        dataset_ver_ind=0)

    gcn2 = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')

    gnn_model_manager_config = ConfigPattern(
            _config_class="ModelManagerConfig",
            _config_kwargs={
                "mask_features": []
            }
        )

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

    node_idx = 666

    explainer_init_config = ConfigPattern(
        _class_name="GNNExplainer(torch-geom)",
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
                "_class_name": "GNNExplainer(torch-geom)",
                "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                "_config_class": "Config",
                "_config_kwargs": {

                },
            }
        }
    )
    explainer_GNNExpl = FrameworkExplainersManager(
        init_config=explainer_init_config,
        dataset=dataset, gnn_manager=gnn_model_manager,
        explainer_name='GNNExplainer(torch-geom)',
    )
    explainer_GNNExpl.conduct_experiment(explainer_run_config)


if __name__ == '__main__':
    geom_GNNExplainer_test()

    # import numpy as np
    # params = {
    #     "GNNExplainer(torch-geom)": [
    #         ["epochs", "Epochs", "int", 100, {"min": 1}, "The learning rate to apply"],
    #         ["lr", "Learn rate", "float", 0.01, {"min": 0}, "The number of epochs to train"],
    #         ["num_hops", "Number of hops", "int", None, {"min": 0, "special": [None]}, "The number of hops the model is aggregating information from. If set to None, will automatically try to detect this information based on the number of torch_geometric.nn.conv.message_passing.MessagePassing layers inside model"],
    #         ["return_type", "Model return", "string", "log_prob", ["log_prob","prob","raw","regression"], "Denotes the type of output from model. Valid inputs are 'log_prob' (the model returns the logarithm of probabilities), 'prob' (the model returns probabilities), 'raw' (the model returns raw scores) and 'regression' (the model returns scalars)"],
    #         ["feat_mask_type", "Feature mask", "string", "feature", ["feature","individual_feature","scalar"], "Denotes the type of feature mask that will be learned. Valid inputs are 'feature' (a single feature-level mask for all nodes), 'individual_feature' (individual feature-level masks for each node), and 'scalar' (scalar mask for each each node)."],
    #         ["allow_edge_mask", "Edge mask", "bool", True, None, "If set to False, the edge mask will not be optimized"],
    #         # [],
    #     ]
    # }
    # import json
    # print(json.dumps(params))
