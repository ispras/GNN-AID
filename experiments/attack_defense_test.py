import torch

import warnings

from torch import device

from models_builder.models_utils import apply_decorator_to_graph_layers
from src.aux.utils import POISON_ATTACK_PARAMETERS_PATH, POISON_DEFENSE_PARAMETERS_PATH, EVASION_ATTACK_PARAMETERS_PATH, \
    EVASION_DEFENSE_PARAMETERS_PATH
from src.models_builder.gnn_models import FrameworkGNNModelManager, Metric
from src.aux.configs import ModelModificationConfig, ConfigPattern
from src.base.datasets_processing import DatasetManager
from src.models_builder.models_zoo import model_configs_zoo
from attacks.QAttack import qattack
from defense.JaccardDefense import jaccard_def
from attacks.metattack import meta_gradient_attack
from defense.GNNGuard import gnnguard


def test_attack_defense():
    my_device = device('cuda' if torch.cuda.is_available() else 'cpu')

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
    # gnn = model_configs_zoo(dataset=dataset, model_name='gat_gcn_sage_gcn_gcn')
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
    dataset.dataset.data.to(my_device)

    # poison_attack_config = ConfigPattern(
    #     _class_name="RandomPoisonAttack",
    #     _import_path=POISON_ATTACK_PARAMETERS_PATH,
    #     _config_class="PoisonAttackConfig",
    #     _config_kwargs={
    #         "n_edges_percent": 0.1,
    #     }
    # )

    metafull_poison_attack_config = ConfigPattern(
        _class_name="MetaAttackFull",
        _import_path=POISON_ATTACK_PARAMETERS_PATH,
        _config_class="PoisonAttackConfig",
        _config_kwargs={
            "num_nodes": dataset.dataset.x.shape[0]
        }
    )

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

    qattack_evasion_attack_config = ConfigPattern(
        _class_name="QAttack",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
            "population_size": 500,
            "individual_size": 100,
            "generations": 100,
            "prob_cross": 0.5,
            "prob_mutate": 0.02
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

    netattack_evasion_attack_config = ConfigPattern(
        _class_name="NettackEvasionAttacker",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
            "node_idx": 0,  # Node for attack
            "n_perturbations": 20,
            "perturb_features": True,
            "perturb_structure": True,
            "direct": True,
            "n_influencers": 3
        }
    )

    netattackgroup_evasion_attack_config = ConfigPattern(
        _class_name="NettackGroupEvasionAttacker",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
            "node_idxs": [random.randint(0, 500) for _ in range(20)],  # Nodes for attack
            "n_perturbations": 50,
            "perturb_features": True,
            "perturb_structure": True,
            "direct": True,
            "n_influencers": 10
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

    quantization_evasion_defense_config = ConfigPattern(
        _class_name="QuantizationDefender",
        _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
        _config_class="EvasionDefenseConfig",
        _config_kwargs={
            "num_levels": 2
        }
    )

    distillation_evasion_defense_config = ConfigPattern(
        _class_name="DistillationDefender",
        _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
        _config_class="EvasionDefenseConfig",
        _config_kwargs={
            "temperature": 0.5 * 20
        }
    )

    autoencoder_evasion_defense_config = ConfigPattern(
        _class_name="AutoEncoderDefender",
        _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
        _config_class="EvasionDefenseConfig",
        _config_kwargs={
            "hidden_dim": 300,
            "bottleneck_dim": 100,
            "reconstruction_loss_weight": 0.1,
        }
    )

    fgsm_evasion_attack_config0 = ConfigPattern(
        _class_name="FGSM",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
            "epsilon": 0.1 * 1,
        }
    )
    at_evasion_defense_config = ConfigPattern(
        _class_name="AdvTraining",
        _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
        _config_class="EvasionDefenseConfig",
        _config_kwargs={
            "attack_name": None,
            "attack_config": fgsm_evasion_attack_config0
        }
    )

    # gnn_model_manager.set_poison_attacker(poison_attack_config=random_poison_attack_config)
    # gnn_model_manager.set_poison_defender(poison_defense_config=gnnguard_poison_defense_config)
    # gnn_model_manager.set_evasion_attacker(evasion_attack_config=fgsm_evasion_attack_config)
    # gnn_model_manager.set_evasion_defender(evasion_defense_config=autoencoder_evasion_defense_config)

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
        gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro'),
                                      Metric("Accuracy", mask='test')])
    print(metric_loc)


def test_meta():
    from attacks.metattack import meta_gradient_attack
    # my_device = device('cpu')
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

    # Node for attack
    node_idx = 1

    # Evaluate model
    mask_loc = Metric.create_mask_by_target_list(y_true=dataset.labels, target_list=[node_idx])
    acc_test_loc = gnn_model_manager.evaluate_model(gen_dataset=dataset,
                                                    metrics=[Metric("Accuracy", mask=mask_loc)])[mask_loc]['Accuracy']

    # acc_train = gnn_model_manager.evaluate_model(gen_dataset=dataset,
    #                                              metrics=[Metric("Accuracy", mask='train')])['train']['Accuracy']
    # acc_test = gnn_model_manager.evaluate_model(gen_dataset=dataset,
    #                                             metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']

    # print(f"Accuracy on train: {acc_train}. Accuracy on test: {acc_test}")
    print(f"Accuracy on test loc: {acc_test_loc}")

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
    acc_test_loc = gnn_model_manager.evaluate_model(gen_dataset=dataset,
                                                    metrics=[Metric("Accuracy", mask=mask_loc)])[mask_loc]['Accuracy']
    print(f"Accuracy on test loc: {acc_test_loc}")


def test_qattack():
    from attacks.QAttack import qattack
    my_device = device('cpu')

    # Load dataset
    # full_name = ("single-graph", "Planetoid", 'Cora')
    full_name = ('single-graph', 'pytorch-geometric-other', 'KarateClub')
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

    num_steps = 100
    gnn_model_manager.train_model(gen_dataset=dataset,
                                  steps=num_steps,
                                  save_model_flag=False)

    evasion_attack_config = ConfigPattern(
        _class_name="QAttack",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
        }
    )

    gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

    # Evaluate model

    # acc_train = gnn_model_manager.evaluate_model(gen_dataset=dataset,
    #                                              metrics=[Metric("Accuracy", mask='train')])['train']['Accuracy']

    acc_test = gnn_model_manager.evaluate_model(gen_dataset=dataset,
                                                metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
    # print(f"Accuracy on train: {acc_train}. Accuracy on test: {acc_test}")
    print(f"Accuracy on test: {acc_test}")

    # Node for attack
    # node_idx = 0
    #
    # # Model prediction on a node before an evasion attack on it
    # gnn_model_manager.gnn.eval()
    # with torch.no_grad():
    #     probabilities = torch.exp(gnn_model_manager.gnn(dataset.data.x, dataset.data.edge_index))
    #
    # predicted_class = probabilities[node_idx].argmax().item()
    # predicted_probability = probabilities[node_idx][predicted_class].item()
    # real_class = dataset.data.y[node_idx].item()

    # info_before_evasion_attack = {"node_idx": node_idx,
    #                               "predicted_class": predicted_class,
    #                               "predicted_probability": predicted_probability,
    #                               "real_class": real_class}

    # Attack config

    # dataset = gnn_model_manager.evasion_attacker.attack(gnn_model_manager, dataset, None)

    # Attack
    # gnn_model_manager.evaluate_model(gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro')])
    #
    # acc_test = gnn_model_manager.evaluate_model(gen_dataset=dataset,
    #                                             metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
    # print(f"Accuracy on test after attack: {acc_test}")

    # # Model prediction on a node after an evasion attack on it
    # with torch.no_grad():
    #     probabilities = torch.exp(gnn_model_manager.gnn(gnn_model_manager.evasion_attacker.attack_diff.data.x,
    #                                                     gnn_model_manager.evasion_attacker.attack_diff.data.edge_index))
    #
    # predicted_class = probabilities[node_idx].argmax().item()
    # predicted_probability = probabilities[node_idx][predicted_class].item()
    # real_class = dataset.data.y[node_idx].item()
    #
    # info_after_evasion_attack = {"node_idx": node_idx,
    #                              "predicted_class": predicted_class,
    #                              "predicted_probability": predicted_probability,
    #                              "real_class": real_class}
    #
    # print(f"info_before_evasion_attack: {info_before_evasion_attack}")
    # print(f"info_after_evasion_attack: {info_after_evasion_attack}")


def test_jaccard():
    from defense.JaccardDefense import jaccard_def
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

    evasion_attack_config = ConfigPattern(
        _class_name="FGSM",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
            "epsilon": 0.007 * 1,
        }
    )
    # evasion_defense_config = ConfigPattern(
    #     _class_name="JaccardDefender",
    #     _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
    #     _config_class="EvasionDefenseConfig",
    #     _config_kwargs={
    #     }
    # )
    poison_defense_config = ConfigPattern(
        _class_name="JaccardDefender",
        _import_path=POISON_DEFENSE_PARAMETERS_PATH,
        _config_class="PoisonDefenseConfig",
        _config_kwargs={
        }
    )

    # gnn_model_manager.set_poison_attacker(poison_attack_config=poison_attack_config)
    gnn_model_manager.set_poison_defender(poison_defense_config=poison_defense_config)
    gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)
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
                                                              metrics=[Metric("F1", mask='train', average=None),
                                                                       Metric("Accuracy", mask="train")])

        if train_test_split_path is not None:
            dataset.save_train_test_mask(train_test_split_path)
            train_mask, val_mask, test_mask, train_test_sizes = torch.load(train_test_split_path / 'train_test_split')[
                                                                :]
            dataset.train_mask, dataset.val_mask, dataset.test_mask = train_mask, val_mask, test_mask
            data.percent_train_class, data.percent_test_class = train_test_sizes

    warnings.warn("Training was successful")

    metric_loc = gnn_model_manager.evaluate_model(
        gen_dataset=dataset, metrics=[Metric("F1", mask='train', average='macro'),
                                      Metric("Accuracy", mask='train')])
    print("TRAIN", metric_loc)

    metric_loc = gnn_model_manager.evaluate_model(
        gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro'),
                                      Metric("Accuracy", mask='test')])
    print("TEST", metric_loc)


def test_adv_training():
    from defense.evasion_defense import AdvTraining

    my_device = device('cpu')
    # full_name = ("single-graph", "Planetoid", 'Cora')
    full_name = ("single-graph", "Amazon", 'Photo')

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

    evasion_defense_config = ConfigPattern(
        _class_name="AdvTraining",
        _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
        _config_class="EvasionDefenseConfig",
        _config_kwargs={
            # "num_nodes": dataset.dataset.x.shape[0]
        }
    )
    from defense.evasion_defense import EvasionDefender
    from src.aux.utils import all_subclasses
    print([e.name for e in all_subclasses(EvasionDefender)])
    gnn_model_manager.set_evasion_defender(evasion_defense_config=evasion_defense_config)

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


def test_pgd():
    # ______________________ Attack on node ______________________
    my_device = device('cpu')

    # Load dataset
    full_name = ("single-graph", "Planetoid", 'Cora')
    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=full_name,
        dataset_ver_ind=0
    )

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

    acc_test = gnn_model_manager.evaluate_model(gen_dataset=dataset,
                                                metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
    print(f"Accuracy on test: {acc_test}")

    # Node for attack
    node_idx = 650

    # Model prediction on a node before PGD attack on it
    gnn_model_manager.gnn.eval()
    with torch.no_grad():
        probabilities = torch.exp(gnn_model_manager.gnn(dataset.data.x, dataset.data.edge_index))

    predicted_class = probabilities[node_idx].argmax().item()
    predicted_probability = probabilities[node_idx][predicted_class].item()
    real_class = dataset.data.y[node_idx].item()

    info_before_pgd_attack_on_node = {"node_idx": node_idx,
                                      "predicted_class": predicted_class,
                                      "predicted_probability": predicted_probability,
                                      "real_class": real_class}

    # Attack config
    evasion_attack_config = ConfigPattern(
        _class_name="PGD",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
            "is_feature_attack": True,
            "element_idx": node_idx,
            "epsilon": 0.1,
            "learning_rate": 0.001,
            "num_iterations": 500,
            "num_rand_trials": 100
        }
    )

    gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

    # Attack
    _ = gnn_model_manager.evaluate_model(gen_dataset=dataset,
                                         metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']

    # Model prediction on a node after PGD attack on it
    with torch.no_grad():
        probabilities = torch.exp(gnn_model_manager.gnn(gnn_model_manager.evasion_attacker.attack_diff.data.x,
                                                        gnn_model_manager.evasion_attacker.attack_diff.data.edge_index))

    predicted_class = probabilities[node_idx].argmax().item()
    predicted_probability = probabilities[node_idx][predicted_class].item()
    real_class = dataset.data.y[node_idx].item()

    info_after_pgd_attack_on_node = {"node_idx": node_idx,
                                     "predicted_class": predicted_class,
                                     "predicted_probability": predicted_probability,
                                     "real_class": real_class}
    # ____________________________________________________________

    # ______________________ Attack on graph _____________________
    # Load dataset
    full_name = ("multiple-graphs", "TUDataset", 'MUTAG')
    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=full_name,
        dataset_ver_ind=0
    )

    model = model_configs_zoo(dataset=dataset, model_name='gin_gin_gin_lin_lin_con')

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
        gnn=model,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=0)
    )

    gnn_model_manager.gnn.to(my_device)

    num_steps = 200
    gnn_model_manager.train_model(gen_dataset=dataset,
                                  steps=num_steps,
                                  save_model_flag=False)

    acc_test = gnn_model_manager.evaluate_model(gen_dataset=dataset,
                                                metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
    print(f"Accuracy on test: {acc_test}")

    # Graph for attack
    graph_idx = 0

    # Model prediction on a graph before PGD attack on it
    gnn_model_manager.gnn.eval()
    with torch.no_grad():
        probabilities = torch.exp(gnn_model_manager.gnn(dataset.dataset[graph_idx].x,
                                                        dataset.dataset[graph_idx].edge_index))

    predicted_class = probabilities.argmax().item()
    predicted_probability = probabilities[0][predicted_class].item()
    real_class = dataset.dataset[graph_idx].y.item()

    info_before_pgd_attack_on_graph = {"graph_idx": graph_idx,
                                       "predicted_class": predicted_class,
                                       "predicted_probability": predicted_probability,
                                       "real_class": real_class}

    # Attack config
    evasion_attack_config = ConfigPattern(
        _class_name="PGD",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
            "is_feature_attack": True,
            "element_idx": graph_idx,
            "epsilon": 0.1,
            "learning_rate": 0.001,
            "num_iterations": 500,
            "num_rand_trials": 100
        }
    )

    gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

    # Attack
    _ = gnn_model_manager.evaluate_model(gen_dataset=dataset,
                                         metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']

    # Model prediction on a graph after PGD attack on it
    with torch.no_grad():
        probabilities = torch.exp(
            gnn_model_manager.gnn(gnn_model_manager.evasion_attacker.attack_diff.dataset[graph_idx].x,
                                  gnn_model_manager.evasion_attacker.attack_diff.dataset[graph_idx].edge_index))

    predicted_class = probabilities.argmax().item()
    predicted_probability = probabilities[0][predicted_class].item()
    real_class = dataset.dataset[graph_idx].y.item()

    info_after_pgd_attack_on_graph = {"graph_idx": graph_idx,
                                      "predicted_class": predicted_class,
                                      "predicted_probability": predicted_probability,
                                      "real_class": real_class}

    # ____________________________________________________________
    print(f"Before PGD attack on node (Cora dataset): {info_before_pgd_attack_on_node}")
    print(f"After PGD attack on node (Cora dataset): {info_after_pgd_attack_on_node}")
    print(f"Before PGD attack on graph (MUTAG dataset): {info_before_pgd_attack_on_graph}")
    print(f"After PGD attack on graph (MUTAG dataset): {info_after_pgd_attack_on_graph}")


if __name__ == '__main__':
    import random

    random.seed(10)
    test_attack_defense()
    # torch.manual_seed(5000)
    # test_gnnguard()
    # test_jaccard()
    # test_pgd()
