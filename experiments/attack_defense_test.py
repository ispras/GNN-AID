import torch

import warnings

from torch import device

from src.aux.utils import POISON_ATTACK_PARAMETERS_PATH, POISON_DEFENSE_PARAMETERS_PATH, EVASION_ATTACK_PARAMETERS_PATH, \
    EVASION_DEFENSE_PARAMETERS_PATH
from src.models_builder.gnn_models import FrameworkGNNModelManager, Metric
from src.aux.configs import ModelModificationConfig, ConfigPattern
from src.base.datasets_processing import DatasetManager
from src.models_builder.models_zoo import model_configs_zoo


def test_attack_defense():
    # my_device = device('cuda' if is_available() else 'cpu')
    my_device = device('cpu')

    full_name = None

    # full_name = ("multiple-graphs", "TUDataset", 'MUTAG')
    # full_name = ("single-graph", "custom", 'karate')
    full_name = ("single-graph", "Planetoid", 'Cora')
    # full_name = ("multiple-graphs", "TUDataset", 'PROTEINS')

    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=full_name,
        dataset_ver_ind=0
    )

    # dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
    #     full_name=("single-graph", "custom", "example",),
    #     features={'attr': {'a': 'as_is', 'b': 'as_is'}},
    #     labeling='threeClasses',
    #     dataset_ver_ind=0
    # )

    # dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
    #     # full_name=("single-graph", "vk_samples", "vk2-ff40-N100000-A.1612175945",),
    #     full_name=("single-graph", "vk_samples", "vk2-ff20-N10000-A.1611943634",),
    #     # full_name=("single-graph", "vk_samples", "vk2-ff20-N1000-U.1612273925",),
    #     # features=('sex',),
    #     features={'str_f': tuple(), 'str_g': None, 'attr': {
    #         # "('personal', 'political')": 'one_hot',
    #         # "('occupation', 'type')": 'one_hot', # Don't work now
    #         # "('relation',)": 'one_hot',
    #         # "('age',)": 'one_hot',
    #         "('sex',)": 'one_hot',
    #     }},
    #     # features={'str_f': tuple(), 'str_g': None, 'attr': {'sex': 'one_hot', }},
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

    save_model_flag = False
    # save_model_flag = True

    # data.x = data.x.float()
    gnn_model_manager.gnn.to(my_device)
    data = data.to(my_device)

    poison_attack_config = ConfigPattern(
        _class_name="RandomPoisonAttack",
        _import_path=POISON_ATTACK_PARAMETERS_PATH,
        _config_class="PoisonAttackConfig",
        _config_kwargs={
            "n_edges_percent": 0.1,
        }
    )

    # poison_defense_config = ConfigPattern(
    #     _class_name="BadRandomPoisonDefender",
    #     _import_path=POISON_DEFENSE_PARAMETERS_PATH,
    #     _config_class="PoisonDefenseConfig",
    #     _config_kwargs={
    #         "n_edges_percent": 0.1,
    #     }
    # )
    poison_defense_config = ConfigPattern(
        _class_name="EmptyPoisonDefender",
        _import_path=POISON_DEFENSE_PARAMETERS_PATH,
        _config_class="PoisonDefenseConfig",
        _config_kwargs={
        }
    )

    evasion_attack_config = ConfigPattern(
        _class_name="FGSM",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
            "epsilon": 0.01 * 1,
        }
    )
    evasion_defense_config = ConfigPattern(
        _class_name="GradientRegularizationDefender",
        _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
        _config_class="EvasionDefenseConfig",
        _config_kwargs={
            "regularization_strength": 0.1 * 10
        }
    )

    gnn_model_manager.set_poison_attacker(poison_attack_config=poison_attack_config)
    # gnn_model_manager.set_poison_defender(poison_defense_config=poison_defense_config)
    # gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)
    # gnn_model_manager.set_evasion_defender(evasion_defense_config=evasion_defense_config)

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

def test_meta():
    from attacks.poison_attacks_collection.metattack import meta_gradient_attack
    my_device = device('cpu')
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
                # "_config_class": "Config",
                "_class_name": "Adam",
                # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                # "_class_import_info": ["torch.optim"],
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
    data = data.to(my_device)

    poison_attack_config = ConfigPattern(
        _class_name="MetaAttackApprox",
        _import_path=POISON_ATTACK_PARAMETERS_PATH,
        _config_class="PoisonAttackConfig",
        _config_kwargs={
            "num_nodes": dataset.dataset.x.shape[0]
        }
    )
    gnn_model_manager.set_poison_attacker(poison_attack_config=poison_attack_config)

    warnings.warn("Start training")
    dataset.train_test_split(percent_train_class=0.1)

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
        gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro'),
                                      Metric("Accuracy", mask='test')])
    print(metric_loc)

def test_nettack_evasion():
    my_device = device('cpu')

    # Load dataset
    full_name = ("single-graph", "Planetoid", 'Cora')
    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=full_name,
        dataset_ver_ind=0
    )

    # Train model on original dataset and remember the model metric and node predicted probability
    gcn_gcn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')

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
        gnn=gcn_gcn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=0)
    )

    gnn_model_manager.gnn.to(my_device)

    num_steps = 200
    gnn_model_manager.train_model(gen_dataset=dataset,
                                  steps=num_steps,
                                  save_model_flag=False)

    # Evaluate model
    acc_train = gnn_model_manager.evaluate_model(gen_dataset=dataset,
                                                 metrics=[Metric("Accuracy", mask='train')])['train']['Accuracy']
    acc_test = gnn_model_manager.evaluate_model(gen_dataset=dataset,
                                                metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
    print(f"Accuracy on train: {acc_train}. Accuracy on test: {acc_test}")

    # Node for attack
    node_idx = 0

    # Model prediction on a node before an evasion attack on it
    gnn_model_manager.gnn.eval()
    with torch.no_grad():
        probabilities = torch.exp(gnn_model_manager.gnn(dataset.data.x, dataset.data.edge_index))

    predicted_class = probabilities[node_idx].argmax().item()
    predicted_probability = probabilities[node_idx][predicted_class].item()
    real_class = dataset.data.y[node_idx].item()

    info_before_evasion_attack = {"node_idx": node_idx,
                                  "predicted_class": predicted_class,
                                  "predicted_probability": predicted_probability,
                                  "real_class": real_class}

    # Attack config
    evasion_attack_config = ConfigPattern(
        _class_name="NettackEvasionAttacker",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
            "node_idx": node_idx,
            "n_perturbations": 20,
            "perturb_features": True,
            "perturb_structure": True,
            "direct": True,
            "n_influencers": 0
        }
    )

    gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

    # Attack
    gnn_model_manager.evaluate_model(gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro')])

    # Model prediction on a node after an evasion attack on it
    with torch.no_grad():
        probabilities = torch.exp(gnn_model_manager.gnn(gnn_model_manager.evasion_attacker.attack_diff.data.x,
                                                        gnn_model_manager.evasion_attacker.attack_diff.data.edge_index))

    predicted_class = probabilities[node_idx].argmax().item()
    predicted_probability = probabilities[node_idx][predicted_class].item()
    real_class = dataset.data.y[node_idx].item()

    info_after_evasion_attack = {"node_idx": node_idx,
                                 "predicted_class": predicted_class,
                                 "predicted_probability": predicted_probability,
                                 "real_class": real_class}

    print(f"info_before_evasion_attack: {info_before_evasion_attack}")
    print(f"info_after_evasion_attack: {info_after_evasion_attack}")


def test_pgd():
    my_device = device('cpu')

    # Load dataset
    full_name = ("single-graph", "Planetoid", 'Cora')
    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=full_name,
        dataset_ver_ind=0
    )

    # Train model on original dataset and remember the model metric and node predicted probability
    gcn_gcn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn_no_self_loops')

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
        gnn=gcn_gcn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=0)
    )

    gnn_model_manager.gnn.to(my_device)

    num_steps = 200
    gnn_model_manager.train_model(gen_dataset=dataset,
                                  steps=num_steps,
                                  save_model_flag=False)

    # Evaluate model before attack on it
    acc_test_ba = gnn_model_manager.evaluate_model(gen_dataset=dataset,
                                                metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
    # print(f"Before attack: Accuracy on train: {acc_train}. Accuracy on test: {acc_test}")

    # Attack config
    evasion_attack_config = ConfigPattern(
        _class_name="PGD",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
            "perturb_ratio": 0.5,
            "learning_rate": 0.01,
            "num_iterations": 100,
            "num_rand_trials": 100
        }
    )

    gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

    # Attack
    acc_test_aa = gnn_model_manager.evaluate_model(gen_dataset=dataset,
                                                metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']

    print(f"Before attack: Accuracy on test: {acc_test_ba}")
    print(f"After PGD attack: Accuracy on test: {acc_test_aa}")


if __name__ == '__main__':
    # test_attack_defense()
    # test_nettack_evasion()
    # torch.manual_seed(5000)
    # test_meta()
    test_pgd()
