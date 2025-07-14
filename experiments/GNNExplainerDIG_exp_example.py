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


def dig_GNNExplainer_test():
    my_device = device('cuda' if is_available() else 'cpu')
    my_device = device('cpu')

    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=(LibPTGDataset.data_folder, "single-graph", "Planetoid", "Cora"),
        dataset_ver_ind=0)

    # dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
    #     full_name=("multiple-graphs", "TUDataset", 'PROTEINS'),
    #     dataset_ver_ind=0
    # )

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

    # node_idx = 2377

    explainer_init_config = ConfigPattern(
        _class_name="GNNExplainer(dig)",
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
                "_class_name": "GNNExplainer(dig)",
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
        explainer_name='GNNExplainer(dig)',
    )
    explainer_GNNExpl.conduct_experiment(explainer_run_config)


if __name__ == '__main__':
    dig_GNNExplainer_test()
