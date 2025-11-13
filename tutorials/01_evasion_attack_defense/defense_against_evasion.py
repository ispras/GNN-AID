import torch
from torch import device

from aux.utils import EVASION_ATTACK_PARAMETERS_PATH, EVASION_DEFENSE_PARAMETERS_PATH
from datasets.ptg_datasets import LibPTGDataset
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from data_structures.configs import ConfigPattern, DatasetConfig, Task
from datasets.datasets_manager import DatasetManager
from models_builder.models_zoo import model_configs_zoo


def test_attack_defense_small():
    my_device = device('cuda' if torch.cuda.is_available() else 'cpu')

    full_name = (LibPTGDataset.data_folder, "Homogeneous", "Planetoid", 'Cora')
    gen_dataset = DatasetManager.get_by_config(
        DatasetConfig((LibPTGDataset.data_folder, "Homogeneous", "Planetoid", "Cora")),
        LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.NODE_CLASSIFICATION})
    )
    gen_dataset.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
    gen_dataset.data.to(my_device)

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

    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=gen_dataset.prepared_dir,
        manager_config=manager_config,
    )

    gnn_model_manager.gnn.to(my_device)

    gnn_model_manager.train_model(gen_dataset=gen_dataset, steps=200)

    metric_before_evasion_attack = gnn_model_manager.evaluate_model(
        gen_dataset=gen_dataset, metrics=[Metric("F1", mask='test', average='macro'),
                                      Metric("Accuracy", mask='test')])

    evasion_attack_config = ConfigPattern(
        _class_name="FGSM",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
            "epsilon": 0.01,
            "is_feature_attack": True,
        }
    )

    gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

    metric_after_evasion_attack = gnn_model_manager.evaluate_model(
        gen_dataset=gen_dataset, metrics=[Metric("F1", mask='test', average='macro'),
                                      Metric("Accuracy", mask='test')])

    # Now we will create evasion defense, and then we will attack the defended model. You can see the available evasion
    # defenses and their default parameters in ./metainfo/evasion_defense_parameters.json
    gradientregularization_evasion_defense_config = ConfigPattern(
        _class_name="GradientRegularizationDefender",
        _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
        _config_class="EvasionDefenseConfig",
        _config_kwargs={
            "regularization_strength": 100
        }
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

    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=gen_dataset.prepared_dir,
        manager_config=manager_config,
    )

    gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

    # Here we pass information to the model manager about the defense configuration, which will be enabled by default
    # at the training stage. If you need to disable it, you can change the defense flag to inactive
    # (gnn_model_manager.evasion_defense_flag=False), then the manager will know about the defense, but will not use it.
    gnn_model_manager.set_evasion_defender(evasion_defense_config=gradientregularization_evasion_defense_config)

    gnn_model_manager.train_model(gen_dataset=gen_dataset, steps=200)

    metric_after_evasion_attack_use_reg_def = gnn_model_manager.evaluate_model(
        gen_dataset=gen_dataset, metrics=[Metric("F1", mask='test', average='macro'),
                                      Metric("Accuracy", mask='test')])

    print(f"Model accuracy without attack and defense: {metric_before_evasion_attack}")
    print(f"Model accuracy after FGSM attack: {metric_after_evasion_attack}")
    print(f"Model accuracy after FGSM attack and use Gradient Regularization Defender: {metric_after_evasion_attack_use_reg_def}")


if __name__ == '__main__':
    import random

    random.seed(10)
    test_attack_defense_small()