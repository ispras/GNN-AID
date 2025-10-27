import numpy as np
import torch

import warnings

from torch import device

from src.aux.utils import EVASION_ATTACK_PARAMETERS_PATH, EVASION_DEFENSE_PARAMETERS_PATH
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from data_structures.configs import ModelModificationConfig, ConfigPattern
from base.datasets_processing import DatasetManager
from models_builder.models_zoo import model_configs_zoo


def test_attack_defense_small():
    my_device = device('cuda' if torch.cuda.is_available() else 'cpu')
    full_name = ("single-graph", "Planetoid", 'Cora')

    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=full_name,
        dataset_ver_ind=0
    )
    gnn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')

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
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )

    save_model_flag = False

    gnn_model_manager.gnn.to(my_device)
    dataset.dataset.data.to(my_device)

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
    dataset.train_test_split()

    gnn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')

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
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )
    warnings.warn("Start training")
    dataset.train_test_split()

    gnn_model_manager.epochs = gnn_model_manager.modification.epochs = 0
    gnn_model_manager.train_model(
        gen_dataset=dataset, steps=steps_epochs,
        save_model_flag=save_model_flag,
        metrics=[Metric("F1", mask='train', average=None)]
    )

    warnings.warn("Training was successful")

    metric_loc_clean = gnn_model_manager.evaluate_model(
        gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro'),
                                      Metric("Accuracy", mask='test')])

    gnn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')

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
        gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro'),
                                      Metric("Accuracy", mask='test')])

    gnn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')

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
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )

    gnn_model_manager.set_evasion_attacker(evasion_attack_config=fgsm_evasion_attack_config)
    gnn_model_manager.set_evasion_defender(evasion_defense_config=gradientregularization_evasion_defense_config)

    warnings.warn("Start training")
    # dataset.train_test_split()

    gnn_model_manager.epochs = gnn_model_manager.modification.epochs = 0
    gnn_model_manager.train_model(
        gen_dataset=dataset, steps=steps_epochs,
        save_model_flag=save_model_flag,
        metrics=[Metric("F1", mask='train', average=None)]
    )

    warnings.warn("Training was successful")

    metric_loc_grad_reg = gnn_model_manager.evaluate_model(
        gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro'),
                                      Metric("Accuracy", mask='test')])

    print(f"Model accuracy without attack and defense: {metric_loc_clean}")
    print(f"Model accuracy after FGSM attack: {metric_loc_fgsm}")
    print(f"Model accuracy after FGSM attack and use Gradient Regularization Defender: {metric_loc_grad_reg}")


if __name__ == '__main__':
    import random

    random.seed(10)
    test_attack_defense_small()
