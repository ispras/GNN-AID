import json
import os
import random
import warnings

import torch

from aux.custom_decorators import timing_decorator
from aux.utils import EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH, EXPLAINERS_INIT_PARAMETERS_PATH, root_dir, \
    EVASION_DEFENSE_PARAMETERS_PATH, EVASION_ATTACK_PARAMETERS_PATH, POISON_ATTACK_PARAMETERS_PATH
from explainers.explainers_manager import FrameworkExplainersManager
from models_builder.gnn_constructor import FrameworkGNNConstructor
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from src.aux.configs import ModelModificationConfig, ConfigPattern, ModelConfig
from src.aux.utils import POISON_DEFENSE_PARAMETERS_PATH
from src.base.datasets_processing import DatasetManager
from src.models_builder.models_zoo import model_configs_zoo
from defense.JaccardDefense import jaccard_def
from attacks.metattack import meta_gradient_attack
from defense.GNNGuard import gnnguard


def load_result_dict(path):
    if os.path.exists(path):
        with open(path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
    return data


def save_result_dict(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(data, file)


def get_model_by_name(model_name, dataset):
    return model_configs_zoo(dataset=dataset, model_name=model_name)


def explainer_run_config_for_node(explainer_name, node_ind, explainer_kwargs=None):
    if explainer_kwargs is None:
        explainer_kwargs = {}
    explainer_kwargs["element_idx"] = node_ind
    return ConfigPattern(
        _config_class="ExplainerRunConfig",
        _config_kwargs={
            "mode": "local",
            "kwargs": {
                "_class_name": explainer_name,
                "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                "_config_class": "Config",
                "_config_kwargs": explainer_kwargs
            }
        }
    )

@timing_decorator
def run_interpretation_test(explainer_name, dataset_full_name, model_name):
    steps_epochs = 10
    num_explaining_nodes = 1
    explaining_metrics_params = {
        "stability_graph_perturbations_nums": 1,
        "stability_feature_change_percent": 0.05,
        "stability_node_removal_percent": 0.05,
        "consistency_num_explanation_runs": 1,
    }
    # steps_epochs = 200
    # num_explaining_nodes = 30
    # explaining_metrics_params = {
    #     "stability_graph_perturbations_nums": 5,
    #     "stability_feature_change_percent": 0.05,
    #     "stability_node_removal_percent": 0.05,
    #     "consistency_num_explanation_runs": 5
    # }
    explainer_kwargs_by_explainer_name = {
        'GNNExplainer(torch-geom)': {},
        'SubgraphX': {"max_nodes": 5},
        'Zorro': {},
    }
    dataset_key_name = "_".join(dataset_full_name)
    metrics_path = root_dir / "experiments" / "explainers_metrics"
    dataset_metrics_path = metrics_path / f"{model_name}_{dataset_key_name}_{explainer_name}_metrics.json"

    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=dataset_full_name,
        dataset_ver_ind=0
    )
    explainer_kwargs = explainer_kwargs_by_explainer_name[explainer_name]

    restart_experiment = False
    if restart_experiment:

        node_indices = random.sample(range(dataset.data.x.shape[0]), num_explaining_nodes)
        result_dict = {
            "num_nodes": num_explaining_nodes,
            "nodes": list(node_indices),
            "metrics_params": explaining_metrics_params,
            "explainer_kwargs": explainer_kwargs
        }
        # save_result_dict(dataset_metrics_path, result_dict)
    else:
        result_dict = load_result_dict(dataset_metrics_path)
        if "nodes" not in result_dict:
            node_indices = random.sample(range(dataset.data.x.shape[0]), num_explaining_nodes)
            result_dict["nodes"] = list(node_indices)
            result_dict["metrics_params"] = explaining_metrics_params
            result_dict["num_nodes"] = num_explaining_nodes
            result_dict["explainer_kwargs"] = explainer_kwargs
            save_result_dict(dataset_metrics_path, result_dict)
        node_indices = result_dict["nodes"]
        explaining_metrics_params = result_dict["metrics_params"]


    node_id_to_explainer_run_config = \
        {node_id: explainer_run_config_for_node(explainer_name, node_id, explainer_kwargs) for node_id in node_indices}

    experiment_name_to_experiment = [
        ("Unprotected", calculate_unprotected_metrics),
        ("Jaccard_defence", calculate_jaccard_defence_metrics),
        ("AdvTraining_defence", calculate_adversial_defence_metrics),
        ("GNNGuard_defence", calculate_gnnguard_defence_metrics),
    ]

    for experiment_name, calculate_fn in experiment_name_to_experiment:
        if experiment_name not in result_dict:
            print(f"Calculation of explanation metrics with defence: {experiment_name} started.")
            explaining_metrics_params["experiment_name"] = experiment_name
            metrics = calculate_fn(
                explainer_name,
                steps_epochs,
                explaining_metrics_params,
                dataset,
                node_id_to_explainer_run_config,
                model_name
            )
            result_dict[experiment_name] = metrics
            print(f"Calculation of explanation metrics with defence: {experiment_name} completed. Metrics:\n{metrics}")
            save_result_dict(dataset_metrics_path, result_dict)


@timing_decorator
def calculate_unprotected_metrics(
        explainer_name,
        steps_epochs,
        explaining_metrics_params,
        dataset,
        node_id_to_explainer_run_config,
        model_name
):
    save_model_flag = True
    device = torch.device('cpu')

    data, results_dataset_path = dataset.data, dataset.results_dir

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

    gnn = get_model_by_name(model_name, dataset)

    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )
    gnn_model_manager.gnn.to(device)
    data.x = data.x.float()
    # data.y = data.y.float()
    data = data.to(device)

    warnings.warn("Start training")
    try:
        print("Loading model executor")
        gnn_model_manager.load_model_executor()
        print("Loaded model")
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

    explainer_init_config = ConfigPattern(
        _class_name=explainer_name,
        _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
        _config_class="ExplainerInitConfig",
        _config_kwargs={}
    )

    explainer = FrameworkExplainersManager(
        init_config=explainer_init_config,
        dataset=dataset, gnn_manager=gnn_model_manager,
        explainer_name=explainer_name,
    )

    explanation_metrics = explainer.evaluate_metrics(node_id_to_explainer_run_config, explaining_metrics_params)
    return explanation_metrics


@timing_decorator
def calculate_jaccard_defence_metrics(
        explainer_name,
        steps_epochs,
        explaining_metrics_params,
        dataset,
        node_id_to_explainer_run_config,
        model_name
):
    save_model_flag = True
    device = torch.device('cpu')

    data, results_dataset_path = dataset.data, dataset.results_dir

    gnn = get_model_by_name(model_name, dataset)
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
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )
    gnn_model_manager.gnn.to(device)
    data.x = data.x.float()
    data = data.to(device)

    poison_defense_config = ConfigPattern(
        _class_name="JaccardDefender",
        _import_path=POISON_DEFENSE_PARAMETERS_PATH,
        _config_class="PoisonDefenseConfig",
        _config_kwargs={
        }
    )

    gnn_model_manager.set_poison_defender(poison_defense_config=poison_defense_config)
    warnings.warn("Start training")
    try:
        print("Loading model executor")
        gnn_model_manager.load_model_executor()
        print("Loaded model")
    except FileNotFoundError:
        print("Training started.")
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

    explainer_init_config = ConfigPattern(
        _class_name=explainer_name,
        _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
        _config_class="ExplainerInitConfig",
        _config_kwargs={}
    )

    explainer = FrameworkExplainersManager(
        init_config=explainer_init_config,
        dataset=dataset, gnn_manager=gnn_model_manager,
        explainer_name=explainer_name,
    )

    explanation_metrics = explainer.evaluate_metrics(node_id_to_explainer_run_config, explaining_metrics_params)
    return explanation_metrics


@timing_decorator
def calculate_adversial_defence_metrics(
        explainer_name,
        steps_epochs,
        explaining_metrics_params,
        dataset,
        node_id_to_explainer_run_config,
        model_name
):
    save_model_flag = True
    device = torch.device('cpu')

    data, results_dataset_path = dataset.data, dataset.results_dir

    gnn = get_model_by_name(model_name, dataset)
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
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )
    gnn_model_manager.gnn.to(device)
    data.x = data.x.float()
    data = data.to(device)

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
            "attack_config": fgsm_evasion_attack_config0  # evasion_attack_config
        }
    )

    from defense.evasion_defense import EvasionDefender
    from src.aux.utils import all_subclasses
    print([e.name for e in all_subclasses(EvasionDefender)])
    gnn_model_manager.set_evasion_defender(evasion_defense_config=at_evasion_defense_config)

    warnings.warn("Start training")
    try:
        print("Loading model executor")
        gnn_model_manager.load_model_executor()
        print("Loaded model")
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

    explainer_init_config = ConfigPattern(
        _class_name=explainer_name,
        _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
        _config_class="ExplainerInitConfig",
        _config_kwargs={}
    )

    explainer = FrameworkExplainersManager(
        init_config=explainer_init_config,
        dataset=dataset, gnn_manager=gnn_model_manager,
        explainer_name=explainer_name,
    )

    explanation_metrics = explainer.evaluate_metrics(node_id_to_explainer_run_config, explaining_metrics_params)
    return explanation_metrics


@timing_decorator
def calculate_gnnguard_defence_metrics(
        explainer_name,
        steps_epochs,
        explaining_metrics_params,
        dataset,
        node_id_to_explainer_run_config,
        model_name
):
    save_model_flag = True
    device = torch.device('cpu')

    data, results_dataset_path = dataset.data, dataset.results_dir

    gnn = get_model_by_name(model_name, dataset)
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
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )
    gnn_model_manager.gnn.to(device)
    data.x = data.x.float()
    data = data.to(device)

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

    gnn_model_manager.set_poison_defender(poison_defense_config=gnnguard_poison_defense_config)

    warnings.warn("Start training")
    try:
        print("Loading model executor")
        gnn_model_manager.load_model_executor()
        print("Loaded model")
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

    explainer_init_config = ConfigPattern(
        _class_name=explainer_name,
        _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
        _config_class="ExplainerInitConfig",
        _config_kwargs={}
    )

    explainer = FrameworkExplainersManager(
        init_config=explainer_init_config,
        dataset=dataset, gnn_manager=gnn_model_manager,
        explainer_name=explainer_name,
    )

    explanation_metrics = explainer.evaluate_metrics(node_id_to_explainer_run_config, explaining_metrics_params)
    return explanation_metrics


if __name__ == '__main__':
    # random.seed(777)

    explainers = [
        # 'GNNExplainer(torch-geom)',
        # 'SubgraphX',
        "Zorro",
    ]

    models = [
        # 'gcn_gcn',
        'gat_gat',
        # 'sage_sage',
    ]
    datasets = [
        ("single-graph", "Planetoid", 'Cora'),
        # ("single-graph", "Amazon", 'Photo'),
    ]
    for explainer in explainers:
        for dataset_full_name in datasets:
            for model_name in models:
                run_interpretation_test(explainer, dataset_full_name, model_name)
    # dataset_full_name = ("single-graph", "Amazon", 'Photo')
    # run_interpretation_test(dataset_full_name)
