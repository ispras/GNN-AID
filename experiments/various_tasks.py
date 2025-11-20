import torch
from torch import device

from gnn_aid.attacks import Attacker
from gnn_aid.aux.utils import FUNCTIONS_PARAMETERS_PATH, all_subclasses
from gnn_aid.data_structures.configs import DatasetConfig, DatasetVarConfig, FeatureConfig, Task, \
    ConfigPattern, ModelModificationConfig
from gnn_aid.datasets.datasets_manager import DatasetManager
from gnn_aid.datasets.ptg_datasets import LibPTGDataset
from gnn_aid.models_builder.models_utils import Metric
from gnn_aid.models_builder.model_managers import FrameworkGNNModelManager
from gnn_aid.models_builder.models_zoo import model_configs_zoo


def node_regression():
    dc = DatasetConfig(('example', 'example'))
    dvc = DatasetVarConfig(task=Task.NODE_REGRESSION, labeling='regression',
                           features=FeatureConfig(node_attr=['a']), dataset_ver_ind=0)

    gen_dataset = DatasetManager.get_by_config(dc, dvc)
    gen_dataset.info.check()

    print(gen_dataset.data)

    # gnn = model_configs_zoo(dataset=gen_dataset, model_name='gcn_gcn')
    # manager_config = ConfigPattern(
    #     _config_class="ModelManagerConfig",
    #     _config_kwargs={
    #         "mask_features": [],
    #         "optimizer": {
    #             "_class_name": "Adam",
    #             "_config_kwargs": {},
    #         }
    #     }
    # )
    #
    # steps_epochs = 10
    # my_device = device('cuda' if torch.cuda.is_available() else 'cpu')
    # gnn_model_manager = FrameworkGNNModelManager(
    #     gnn=gnn,
    #     dataset_path=gen_dataset.prepared_dir,
    #     manager_config=manager_config,
    #     modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    # )
    #
    # gnn_model_manager.gnn.to(my_device)
    # gen_dataset.data.to(my_device)
    #
    # gen_dataset.train_test_split()
    # gnn_model_manager.train_model(
    #     gen_dataset=gen_dataset, steps=steps_epochs,
    #     save_model_flag=False,
    #     metrics=[Metric("F1", mask='train', average=None)]
    # )
    # print("Training was successful")


def graph_regression():
    dc = DatasetConfig(('example', 'example3'))

    labeling_dict = {"0": 0.3, "1": 0.4, "2": 0.2}
    DatasetManager.add_labeling(dc, Task.GRAPH_REGRESSION, "regression", labeling_dict)

    dvc = DatasetVarConfig(task=Task.GRAPH_REGRESSION, labeling='regression',
                           features=FeatureConfig(node_attr=['type']), dataset_ver_ind=0)

    gen_dataset = DatasetManager.get_by_config(dc, dvc)
    gen_dataset.info.check()

    print(gen_dataset.data)


def edge_regression():
    dc = DatasetConfig(('example', 'example'))

    labeling_dict = {
        "10,11": 0.32,
        "11,12": 0.25,
        "11,13": 0.13,
        "11,15": 0.1,
        "12,13": 0.3,
        "12,17": 0.22,
        "15,14": 0.35,
        "15,16": 0.40
    }

    DatasetManager.add_labeling(dc, Task.EDGE_REGRESSION, "regression", labeling_dict)

    dvc = DatasetVarConfig(task=Task.EDGE_REGRESSION, labeling="regression",
                           features=FeatureConfig(node_attr=['a']), dataset_ver_ind=0)

    gen_dataset = DatasetManager.get_by_config(dc, dvc)

    print(gen_dataset.data)


def link_prediction():
    # dc = DatasetConfig(('example', 'example'))
    # dvc = DatasetVarConfig(features=FeatureConfig(node_attr=['a']),
    #                        task=Task.EDGE_PREDICTION, dataset_ver_ind=0)

    dc = DatasetConfig((LibPTGDataset.data_folder, 'Homogeneous', 'Planetoid', 'Cora'))
    dvc = LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.EDGE_PREDICTION})

    gen_dataset = DatasetManager.get_by_config(dc, dvc)
    print(gen_dataset.data)
    gen_dataset.train_test_split()

    gnn = model_configs_zoo(dataset=gen_dataset, model_name='gcn_gcn')
    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "batch": 64,
            "mask_features": [],
            "optimizer": {
                "_class_name": "Adam",
                "_config_kwargs": {},
            },
            "loss_function": {
                "_config_class": "Config",
                "_class_name": "CrossEntropyLoss",
                "_import_path": FUNCTIONS_PARAMETERS_PATH,
                "_class_import_info": ["torch.nn"],
                "_config_kwargs": {},
            },
        }
    )

    steps_epochs = 30
    my_device = device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=gen_dataset.prepared_dir,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )

    gnn_model_manager.gnn.to(my_device)
    gen_dataset.data.to(my_device)

    gnn_model_manager.modification.epochs = 0
    gnn_model_manager.train_model(
        gen_dataset=gen_dataset, steps=steps_epochs,
        save_model_flag=False,
        metrics=[Metric("F1", mask='train', average=None)]
    )
    print("Training was successful")


if __name__ == '__main__':
    # node_regression()
    # graph_regression()

    # edge_regression()
    # link_prediction()

    for c in all_subclasses(Attacker):
        print(c)