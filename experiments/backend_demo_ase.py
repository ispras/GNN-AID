import numpy as np
import torch

import warnings

from torch import device

from gnn_aid.aux.utils import EVASION_ATTACK_PARAMETERS_PATH, EVASION_DEFENSE_PARAMETERS_PATH
from gnn_aid.datasets.ptg_datasets import LibPTGDataset
from gnn_aid.models_builder.model_managers import FrameworkGNNModelManager
from gnn_aid.models_builder.models_utils import Metric
from gnn_aid.data_structures.configs import ModelModificationConfig, DatasetConfig, Task
from gnn_aid.data_structures.gen_config import ConfigPattern
from gnn_aid.datasets.datasets_manager import DatasetManager
from gnn_aid.models_builder.models_zoo import model_configs_zoo


def test_attack_defense_small():
    my_device = device('cuda' if torch.cuda.is_available() else 'cpu')
    full_name = (LibPTGDataset.data_folder, "Homogeneous", "Planetoid", 'Cora')

    gen_dataset = DatasetManager.get_by_config(
        DatasetConfig(full_name),
        LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.NODE_CLASSIFICATION})
    )
    gnn = model_configs_zoo(dataset=gen_dataset, model_name='gcn_gcn')

    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "mask_features": [],
            "optimizer": {
                "_class_name": "Adam",
                "_config_kwargs": {},
            }
        }
    )

    steps_epochs = 200
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=gen_dataset.prepared_dir,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )

    save_model_flag = False

    gnn_model_manager.gnn.to(my_device)
    gen_dataset.data.to(my_device)

    fgsm_evasion_attack_config = ConfigPattern(
        _class_name="FGSM",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
            "epsilon": 0.01,
            "is_feature_attack": True,
        }
    )

    gradientregularization_evasion_defense_config = ConfigPattern(
        _class_name="GradientRegularizationDefender",
        _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
        _config_class="EvasionDefenseConfig",
        _config_kwargs={
            "regularization_strength": 100
        }
    )

    warnings.warn("Start training")
    gen_dataset.train_test_split()

    gnn = model_configs_zoo(dataset=gen_dataset, model_name='gcn_gcn')

    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "mask_features": [],
            "optimizer": {
                "_class_name": "Adam",
                "_config_kwargs": {},
            }
        }
    )

    steps_epochs = 200
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=gen_dataset.prepared_dir,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )
    warnings.warn("Start training")
    gen_dataset.train_test_split()

    gnn_model_manager.epochs = gnn_model_manager.modification.epochs = 0
    gnn_model_manager.train_model(
        gen_dataset=gen_dataset, steps=steps_epochs,
        save_model_flag=save_model_flag,
        metrics=[Metric("F1", mask='train', average=None)]
    )

    warnings.warn("Training was successful")

    metric_loc_clean = gnn_model_manager.evaluate_model(
        gen_dataset=gen_dataset, metrics=[Metric("F1", mask='test', average='macro'),
                                      Metric("Accuracy", mask='test')])

    gnn = model_configs_zoo(dataset=gen_dataset, model_name='gcn_gcn')

    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "mask_features": [],
            "optimizer": {
                "_class_name": "Adam",
                "_config_kwargs": {},
            }
        }
    )

    gnn_model_manager.set_evasion_attacker(evasion_attack_config=fgsm_evasion_attack_config)

    metric_loc_fgsm = gnn_model_manager.evaluate_model(
        gen_dataset=gen_dataset, metrics=[Metric("F1", mask='test', average='macro'),
                                      Metric("Accuracy", mask='test')])

    gnn = model_configs_zoo(dataset=gen_dataset, model_name='gcn_gcn')

    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "mask_features": [],
            "optimizer": {
                "_class_name": "Adam",
                "_config_kwargs": {},
            }
        }
    )

    steps_epochs = 200
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=gen_dataset.prepared_dir,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )

    gnn_model_manager.set_evasion_attacker(evasion_attack_config=fgsm_evasion_attack_config)
    gnn_model_manager.set_evasion_defender(evasion_defense_config=gradientregularization_evasion_defense_config)

    warnings.warn("Start training")
    # dataset.train_test_split()

    gnn_model_manager.epochs = gnn_model_manager.modification.epochs = 0
    gnn_model_manager.train_model(
        gen_dataset=gen_dataset, steps=steps_epochs,
        save_model_flag=save_model_flag,
        metrics=[Metric("F1", mask='train', average=None)]
    )

    warnings.warn("Training was successful")

    metric_loc_grad_reg = gnn_model_manager.evaluate_model(
        gen_dataset=gen_dataset, metrics=[Metric("F1", mask='test', average='macro'),
                                      Metric("Accuracy", mask='test')])

    print(f"Model accuracy without attack and defense: {metric_loc_clean}")
    print(f"Model accuracy after FGSM attack: {metric_loc_fgsm}")
    print(f"Model accuracy after FGSM attack and use Gradient Regularization Defender: {metric_loc_grad_reg}")


if __name__ == '__main__':
    import random

    random.seed(10)
    test_attack_defense_small()
