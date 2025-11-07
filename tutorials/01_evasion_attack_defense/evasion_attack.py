import torch
from torch import device

from src.aux.utils import EVASION_ATTACK_PARAMETERS_PATH
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from data_structures.configs import ConfigPattern
from datasets.datasets_manager import DatasetManager
from models_builder.models_zoo import model_configs_zoo


def evasion_attack():
    my_device = device('cuda' if torch.cuda.is_available() else 'cpu')

    full_name = ("Homogeneous", "Planetoid", 'Cora')
    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=full_name,
        dataset_ver_ind=0
    )
    dataset.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
    dataset.dataset.data.to(my_device)

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

    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
    )

    gnn_model_manager.gnn.to(my_device)

    gnn_model_manager.train_model(gen_dataset=dataset, steps=200)

    metric_before_evasion_attack = gnn_model_manager.evaluate_model(
        gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro'),
                                      Metric("Accuracy", mask='test')])

    # Let's attack our model by first creating the evasion attack we need, and then measure the model's metrics after
    # the evasion attack. You can see the available evasion attack types and their default parameters in
    # ./metainfo/evasion_attack_parameters.json
    evasion_attack_config = ConfigPattern(
        _class_name="FGSM",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
            "epsilon": 0.01,
            "is_feature_attack": True,
        }
    )
    # Here we pass information to the model manager about the attack configuration, which will be enabled by default
    # at the evaluation stage. If you need to disable it, you can change the attack flag to inactive
    # (gnn_model_manager.evasion_attack_flag=False), then the manager will know about the attack, but will not use it.
    gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

    metric_after_evasion_attack = gnn_model_manager.evaluate_model(
        gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro'),
                                      Metric("Accuracy", mask='test')])

    print(f"Model accuracy without attack: {metric_before_evasion_attack}")
    print(f"Model accuracy after FGSM attack: {metric_after_evasion_attack}")


if __name__ == '__main__':
    import random

    random.seed(10)
    evasion_attack()