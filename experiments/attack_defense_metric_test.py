import copy
import warnings

import torch
from torch import device

from models_builder.attack_defense_manager import FrameworkAttackDefenseManager
from models_builder.attack_defense_metric import AttackMetric, DefenseMetric
from models_builder.models_utils import apply_decorator_to_graph_layers
from src.aux.utils import POISON_ATTACK_PARAMETERS_PATH, POISON_DEFENSE_PARAMETERS_PATH, EVASION_ATTACK_PARAMETERS_PATH, \
    EVASION_DEFENSE_PARAMETERS_PATH
from src.models_builder.gnn_models import FrameworkGNNModelManager, Metric
from src.aux.configs import ModelModificationConfig, ConfigPattern
from src.base.datasets_processing import DatasetManager
from src.models_builder.models_zoo import model_configs_zoo


def attack_defense_metrics():
    my_device = device('cuda' if torch.cuda.is_available() else 'cpu')
    my_device = device('cpu')

    full_name = None

    # full_name = ("multiple-graphs", "TUDataset", 'MUTAG')
    # full_name = ("single-graph", "custom", 'karate')
    full_name = ("single-graph", "Planetoid", 'Cora')
    # full_name = ("single-graph", "Amazon", 'Photo')
    # full_name = ("single-graph", "Planetoid", 'CiteSeer')
    # full_name = ("multiple-graphs", "TUDataset", 'PROTEINS')

    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=full_name,
        dataset_ver_ind=0
    )

    gnn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')
    # gnn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn_lin')
    # gnn = model_configs_zoo(dataset=dataset, model_name='gin_gin_gin_lin_lin')

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

    # save_model_flag = False
    save_model_flag = True

    gnn_model_manager.gnn.to(my_device)

    random_poison_attack_config = ConfigPattern(
        _class_name="RandomPoisonAttack",
        _import_path=POISON_ATTACK_PARAMETERS_PATH,
        _config_class="PoisonAttackConfig",
        _config_kwargs={
            "n_edges_percent": 1.0,
        }
    )

    gnnguard_poison_defense_config = ConfigPattern(
        _class_name="GNNGuard",
        _import_path=POISON_DEFENSE_PARAMETERS_PATH,
        _config_class="PoisonDefenseConfig",
        _config_kwargs={
            "lr": 0.01,
            "train_iters": 100,
            # "model": gnn_model_manager.gnn
        }
    )

    jaccard_poison_defense_config = ConfigPattern(
        _class_name="JaccardDefender",
        _import_path=POISON_DEFENSE_PARAMETERS_PATH,
        _config_class="PoisonDefenseConfig",
        _config_kwargs={
            "threshold": 0.05,
        }
    )

    fgsm_evasion_attack_config = ConfigPattern(
        _class_name="FGSM",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
            "epsilon": 0.001 * 12,
        }
    )

    gradientregularization_evasion_defense_config = ConfigPattern(
        _class_name="GradientRegularizationDefender",
        _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
        _config_class="EvasionDefenseConfig",
        _config_kwargs={
            "regularization_strength": 0.1 * 1000
        }
    )

    gnn_model_manager.set_poison_attacker(poison_attack_config=random_poison_attack_config)
    gnn_model_manager.set_poison_defender(poison_defense_config=jaccard_poison_defense_config)
    gnn_model_manager.set_evasion_attacker(evasion_attack_config=fgsm_evasion_attack_config)
    gnn_model_manager.set_evasion_defender(evasion_defense_config=gradientregularization_evasion_defense_config)

    warnings.warn("Start training")
    dataset.train_test_split()

    # try:
    #     # raise FileNotFoundError()
    #     gnn_model_manager.load_model_executor()
    #     dataset = gnn_model_manager.load_train_test_split(dataset)
    # except FileNotFoundError:
    #     gnn_model_manager.epochs = gnn_model_manager.modification.epochs = 0
    #     train_test_split_path = gnn_model_manager.train_model(gen_dataset=dataset, steps=steps_epochs,
    #                                                           save_model_flag=save_model_flag,
    #                                                           metrics=[Metric("F1", mask='train', average=None)])
    #
    #     if train_test_split_path is not None:
    #         dataset.save_train_test_mask(train_test_split_path)
    #         train_mask, val_mask, test_mask, train_test_sizes = torch.load(train_test_split_path / 'train_test_split')[
    #                                                             :]
    #         dataset.train_mask, dataset.val_mask, dataset.test_mask = train_mask, val_mask, test_mask
    #         data.percent_train_class, data.percent_test_class = train_test_sizes
    #
    # warnings.warn("Training was successful")
    #
    # metric_loc = gnn_model_manager.evaluate_model(
    #     gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro'),
    #                                   Metric("Accuracy", mask='test')])
    # print(metric_loc)

    adm = FrameworkAttackDefenseManager(
        gen_dataset=copy.deepcopy(dataset),
        gnn_manager=gnn_model_manager,
    )
    # adm.evasion_attack_pipeline(
    #     steps=steps_epochs,
    #     save_model_flag=save_model_flag,
    #     metrics_attack=[AttackMetric("ASR")],
    #     mask='test'
    # )
    # adm.poison_attack_pipeline(
    #     steps=steps_epochs,
    #     save_model_flag=save_model_flag,
    #     metrics_attack=[AttackMetric("ASR")],
    #     mask='test'
    # )
    adm.evasion_defense_pipeline(
        steps=steps_epochs,
        save_model_flag=save_model_flag,
        metrics_attack=[AttackMetric("ASR"), AttackMetric("AuccAttackDiff"),],
        metrics_defense=[DefenseMetric("AuccDefenseCleanDiff"), DefenseMetric("AuccDefenseAttackDiff"), ],
        mask='test'
    )


if __name__ == '__main__':
    import random

    random.seed(10)
    attack_defense_metrics()