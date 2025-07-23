import json
import os
import random
import warnings

import torch

from aux.custom_decorators import timing_decorator, retry
from aux.utils import EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH, EXPLAINERS_INIT_PARAMETERS_PATH, root_dir, \
    EVASION_DEFENSE_PARAMETERS_PATH, EVASION_ATTACK_PARAMETERS_PATH
from explainers.explainers_manager import FrameworkExplainersManager
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from data_structures.configs import ModelModificationConfig, ConfigPattern
from src.aux.utils import POISON_DEFENSE_PARAMETERS_PATH
from datasets.datasets_manager import DatasetManager
from src.models_builder.models_zoo import model_configs_zoo
from defenses.jaccard_defense import jaccard_def
from attacks.metattack import meta_gradient_attack
from defenses.gnn_guard import gnnguard


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


def explainer_run_config_for_obj(explainer_name, obj_ind, explainer_kwargs=None):
    if explainer_kwargs is None:
        explainer_kwargs = {}
    explainer_kwargs["element_idx"] = obj_ind
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


@retry(max_tries=5)
@timing_decorator
def run_interpretation_test(explainer_name, dataset_full_name, model_name, iter=0):
    steps_epochs = 50  # FOR POWERGRAPH 50, PREVIOUS - 200
    num_explaining_objs = 5
    explaining_metrics_params = {
        "stability_graph_perturbations_nums": 3,
        "stability_feature_change_percent": 0.05,
        "stability_node_removal_percent": 0.05,
        "consistency_num_explanation_runs": 3,
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
        'PGExplainer(dig)': {},
        'GSAT': {}
    }
    dataset_key_name = "_".join(dataset_full_name)
    metrics_path = root_dir / "experiments" / "explainers_metrics"
    dataset_metrics_path = metrics_path / f"{model_name}_{dataset_key_name}_{explainer_name}_iter_{iter}_metrics.json"

    subdataset = None
    if len(dataset_full_name) > 3:
        subdataset = dataset_full_name[-1]
        dataset_full_name = dataset_full_name[:3]
    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=dataset_full_name,
        dataset_ver_ind=0,
        init_kwargs={'name': subdataset}
    )
    explainer_kwargs = explainer_kwargs_by_explainer_name[explainer_name]

    restart_experiment = False
    if restart_experiment:

        expl_indices = random.sample(range(dataset.data.y.shape[0]), num_explaining_objs)
        result_dict = {   # Graph Classification -> objs = graphs; Node Classification -> objs = nodes
            "num_objs": num_explaining_objs,
            "obj_indices": list(expl_indices),
            "metrics_params": explaining_metrics_params,
            "explainer_kwargs": explainer_kwargs
        }
        # save_result_dict(dataset_metrics_path, result_dict)
    else:
        result_dict = load_result_dict(dataset_metrics_path)
        if "obj_indices" not in result_dict:
            expl_indices = random.sample(range(dataset.data.y.shape[0]), num_explaining_objs)
            result_dict["obj_indices"] = list(expl_indices)
            result_dict["metrics_params"] = explaining_metrics_params
            result_dict["num_objs"] = num_explaining_objs
            result_dict["explainer_kwargs"] = explainer_kwargs
            save_result_dict(dataset_metrics_path, result_dict)
        obj_indices = result_dict["obj_indices"]
        explaining_metrics_params = result_dict["metrics_params"]

    obj_id_to_explainer_run_config = \
        {obj_id: explainer_run_config_for_obj(explainer_name, obj_id, explainer_kwargs) for obj_id in obj_indices}

    experiment_name_to_experiment = [
        ("Unprotected", calculate_unprotected_metrics),
        ("Jaccard_defence", calculate_jaccard_defence_metrics),
        ("GNNGuard_defence", calculate_gnnguard_defence_metrics),
        ("AdvTraining_defence", calculate_adversial_defence_metrics),
        ("AutoEncoderDefender", calculate_autoencoder_defence_metrics),
        ("QuantizationDefender", calculate_quantization_defence_metrics),
        ("GradientRegularizationDefender", calculate_gradientregularization_defence_metrics),
        ("DistillationDefender", calculate_distillation_defence_metrics),

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
                obj_id_to_explainer_run_config,
                model_name,
                iteration=iter
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
        obj_id_to_explainer_run_config,
        model_name,
        iteration: int = 0
):
    dataset.train_test_split(percent_train_class=0.8, percent_test_class=0.1)  # FOR POWERGRAPH

    save_model_flag = True
    device = torch.device('cpu')

    data, results_dataset_path = dataset.data, dataset.results_dir

    # if explainer_name == 'GSAT':
    #     lr = 0.01
    # else:
    #     lr = 0.001
    lr = 0.001
    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "batch": 32,  # FOR POWERGRAPH
            "mask_features": [],
            "optimizer": {
                # "_config_class": "Config",
                "_class_name": "Adam",
                # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                # "_class_import_info": ["torch.optim"],
                "_config_kwargs": {
                    "lr": lr  # FOR GSAT
                },
            }
        }
    )

    gnn = get_model_by_name(model_name, dataset)

    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=iteration, epochs=steps_epochs)
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

    explanation_metrics = explainer.evaluate_metrics(obj_id_to_explainer_run_config, explaining_metrics_params)
    return explanation_metrics


@timing_decorator
def calculate_jaccard_defence_metrics(
        explainer_name,
        steps_epochs,
        explaining_metrics_params,
        dataset,
        obj_id_to_explainer_run_config,
        model_name,
        iteration: int = 0
):
    dataset.train_test_split(percent_train_class=0.8, percent_test_class=0.1)  # FOR POWERGRAPH

    save_model_flag = True
    device = torch.device('cpu')

    data, results_dataset_path = dataset.data, dataset.results_dir

    gnn = get_model_by_name(model_name, dataset)
    lr = 0.001
    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "batch": 32,  # FOR POWERGRAPH
            "mask_features": [],
            "optimizer": {
                # "_config_class": "Config",
                "_class_name": "Adam",
                # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                # "_class_import_info": ["torch.optim"],
                "_config_kwargs": {
                    "lr": lr  # FOR GSAT
                },
            }
        }
    )
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=iteration, epochs=steps_epochs)
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

    explanation_metrics = explainer.evaluate_metrics(obj_id_to_explainer_run_config, explaining_metrics_params)
    return explanation_metrics


@timing_decorator
def calculate_adversial_defence_metrics(
        explainer_name,
        steps_epochs,
        explaining_metrics_params,
        dataset,
        obj_id_to_explainer_run_config,
        model_name,
        iteration: int = 0
):
    dataset.train_test_split(percent_train_class=0.8, percent_test_class=0.1)  # FOR POWERGRAPH

    save_model_flag = True
    device = torch.device('cpu')

    data, results_dataset_path = dataset.data, dataset.results_dir

    gnn = get_model_by_name(model_name, dataset)
    lr = 0.001
    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "batch": 32,  # FOR POWERGRAPH
            "mask_features": [],
            "optimizer": {
                # "_config_class": "Config",
                "_class_name": "Adam",
                # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                # "_class_import_info": ["torch.optim"],
                "_config_kwargs": {
                    "lr": lr  # FOR GSAT
                },
            }
        }
    )
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=iteration, epochs=steps_epochs)
    )
    gnn_model_manager.gnn.to(device)
    data.x = data.x.float()
    data = data.to(device)

    fgsm_evasion_attack_config0 = ConfigPattern(
        _class_name="FGSM",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
            "epsilon": 0.05 * 1,
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

    from defenses.evasion_defense import EvasionDefender
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

    explanation_metrics = explainer.evaluate_metrics(obj_id_to_explainer_run_config, explaining_metrics_params)
    return explanation_metrics


@timing_decorator
def calculate_gnnguard_defence_metrics(
        explainer_name,
        steps_epochs,
        explaining_metrics_params,
        dataset,
        obj_id_to_explainer_run_config,
        model_name,
        iteration: int = 0
):
    dataset.train_test_split(percent_train_class=0.8, percent_test_class=0.1)  # FOR POWERGRAPH

    save_model_flag = True
    device = torch.device('cpu')

    data, results_dataset_path = dataset.data, dataset.results_dir

    gnn = get_model_by_name(model_name, dataset)
    lr = 0.001
    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "batch": 32,  # FOR POWERGRAPH
            "mask_features": [],
            "optimizer": {
                # "_config_class": "Config",
                "_class_name": "Adam",
                # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                # "_class_import_info": ["torch.optim"],
                "_config_kwargs": {
                    "lr": lr  # FOR GSAT
                },
            }
        }
    )
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=iteration, epochs=steps_epochs)
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
            "train_iters": 50,  # FOR POWERGRAPH 50, PREVIOUS - 200
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

    explanation_metrics = explainer.evaluate_metrics(obj_id_to_explainer_run_config, explaining_metrics_params)
    return explanation_metrics


@timing_decorator
def calculate_autoencoder_defence_metrics(
        explainer_name,
        steps_epochs,
        explaining_metrics_params,
        dataset,
        obj_id_to_explainer_run_config,
        model_name,
        iteration: int = 0
):
    dataset.train_test_split(percent_train_class=0.8, percent_test_class=0.1)  # FOR POWERGRAPH

    save_model_flag = True
    device = torch.device('cpu')

    data, results_dataset_path = dataset.data, dataset.results_dir

    gnn = get_model_by_name(model_name, dataset)
    lr = 0.001
    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "batch": 32,  # FOR POWERGRAPH
            "mask_features": [],
            "optimizer": {
                # "_config_class": "Config",
                "_class_name": "Adam",
                # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                # "_class_import_info": ["torch.optim"],
                "_config_kwargs": {
                    "lr": lr  # FOR GSAT
                },
            }
        }
    )
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=iteration, epochs=steps_epochs)
    )
    gnn_model_manager.gnn.to(device)
    data.x = data.x.float()
    data = data.to(device)

    autoencoder_defense_config = ConfigPattern(
        _class_name="AutoEncoderDefender",
        _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
        _config_class="EvasionDefenseConfig",
        _config_kwargs={
        }
    )

    gnn_model_manager.set_evasion_defender(evasion_defense_config=autoencoder_defense_config)

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

    explanation_metrics = explainer.evaluate_metrics(obj_id_to_explainer_run_config, explaining_metrics_params)
    return explanation_metrics


@timing_decorator
def calculate_gradientregularization_defence_metrics(
        explainer_name,
        steps_epochs,
        explaining_metrics_params,
        dataset,
        obj_id_to_explainer_run_config,
        model_name,
        iteration: int = 0
):
    dataset.train_test_split(percent_train_class=0.8, percent_test_class=0.1)  # FOR POWERGRAPH

    save_model_flag = True
    device = torch.device('cpu')

    data, results_dataset_path = dataset.data, dataset.results_dir

    gnn = get_model_by_name(model_name, dataset)
    lr = 0.001
    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "batch": 32,  # FOR POWERGRAPH
            "mask_features": [],
            "optimizer": {
                # "_config_class": "Config",
                "_class_name": "Adam",
                # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                # "_class_import_info": ["torch.optim"],
                "_config_kwargs": {
                    "lr": lr  # FOR GSAT
                },
            }
        }
    )
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=iteration, epochs=steps_epochs)
    )
    gnn_model_manager.gnn.to(device)
    data.x = data.x.float()
    data = data.to(device)

    gradientregularization_defense_config = ConfigPattern(
        _class_name="GradientRegularizationDefender",
        _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
        _config_class="EvasionDefenseConfig",
        _config_kwargs={
        }
    )

    gnn_model_manager.set_evasion_defender(evasion_defense_config=gradientregularization_defense_config)

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

    explanation_metrics = explainer.evaluate_metrics(obj_id_to_explainer_run_config, explaining_metrics_params)
    return explanation_metrics


@timing_decorator
def calculate_quantization_defence_metrics(
        explainer_name,
        steps_epochs,
        explaining_metrics_params,
        dataset,
        obj_id_to_explainer_run_config,
        model_name,
        iteration: int = 0
):
    dataset.train_test_split(percent_train_class=0.8, percent_test_class=0.1)  # FOR POWERGRAPH

    save_model_flag = True
    device = torch.device('cpu')

    data, results_dataset_path = dataset.data, dataset.results_dir

    gnn = get_model_by_name(model_name, dataset)
    lr = 0.001
    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "batch": 32,  # FOR POWERGRAPH
            "mask_features": [],
            "optimizer": {
                # "_config_class": "Config",
                "_class_name": "Adam",
                # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                # "_class_import_info": ["torch.optim"],
                "_config_kwargs": {
                    "lr": lr  # FOR GSAT
                },
            }
        }
    )
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=iteration, epochs=steps_epochs)
    )
    gnn_model_manager.gnn.to(device)
    data.x = data.x.float()
    data = data.to(device)

    quantization_defense_config = ConfigPattern(
        _class_name="QuantizationDefender",
        _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
        _config_class="EvasionDefenseConfig",
        _config_kwargs={
        }
    )

    gnn_model_manager.set_evasion_defender(evasion_defense_config=quantization_defense_config)

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

    explanation_metrics = explainer.evaluate_metrics(obj_id_to_explainer_run_config, explaining_metrics_params)
    return explanation_metrics


@timing_decorator
def calculate_distillation_defence_metrics(
        explainer_name,
        steps_epochs,
        explaining_metrics_params,
        dataset,
        obj_id_to_explainer_run_config,
        model_name,
        iteration: int = 0
):
    dataset.train_test_split(percent_train_class=0.8, percent_test_class=0.1)  # FOR POWERGRAPH

    save_model_flag = True
    device = torch.device('cpu')

    data, results_dataset_path = dataset.data, dataset.results_dir

    gnn = get_model_by_name(model_name, dataset)
    lr = 0.001
    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "batch": 32,  # FOR POWERGRAPH
            "mask_features": [],
            "optimizer": {
                # "_config_class": "Config",
                "_class_name": "Adam",
                # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                # "_class_import_info": ["torch.optim"],
                "_config_kwargs": {
                    "lr": lr  # FOR GSAT
                },
            }
        }
    )
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=iteration, epochs=steps_epochs)
    )
    gnn_model_manager.gnn.to(device)
    data.x = data.x.float()
    data = data.to(device)

    distillation_defense_config = ConfigPattern(
        _class_name="DistillationDefender",
        _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
        _config_class="EvasionDefenseConfig",
        _config_kwargs={
        }
    )

    gnn_model_manager.set_evasion_defender(evasion_defense_config=distillation_defense_config)

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

    explanation_metrics = explainer.evaluate_metrics(obj_id_to_explainer_run_config, explaining_metrics_params)
    return explanation_metrics


if __name__ == '__main__':
    # random.seed(777)

    explainers = [
        'GNNExplainer(torch-geom)',
        # 'SubgraphX',
        # "Zorro",
        "GSAT",
        "PGExplainer(dig)"
    ]

    models = [
        # 'gcn_gcn',
        # 'gcn_gcn_gcn',
        # 'sage_sage',
        # 'sage_sage_sage',
        # 'gin_gin',
        # 'gat_gat',
        # 'dummy_gcn_gcn_gsat',
        # 'dummy_gcn_gcn_gcn_gsat',
        # 'dummy_sage_sage_gsat',
        # 'dummy_sage_sage_sage_gsat',
        # 'dummy_gin_gin_gsat',
        # 'dummy_gat_gat_gsat',

        'gcn_gcn_lin_gc',
        'gcn_gcn_gcn_lin_gc',
        'sage_sage_lin_gc',
        'sage_sage_sage_lin_gc',
        'gin_gin_lin_gc',
        'gat_gat_lin_gc',
        'dummy_gcn_gcn_gsat_lin_gc',
        'dummy_gcn_gcn_gcn_gsat_lin_gc',
        'dummy_sage_sage_gsat_lin_gc',
        'dummy_sage_sage_sage_gsat_lin_gc',
        'dummy_gin_gin_gsat_lin_gc',
        'dummy_gat_gat_gsat_lin_gc',
    ]

    datasets = [
        # ("single-graph", "Planetoid", 'Cora'),
        # ("single-graph", "Planetoid", 'CiteSeer'),
        # ("single-graph", "Planetoid", 'PubMed'),
        # ("single-graph", "Amazon", 'Computers'),
        # ("single-graph", "Amazon", 'Photo'),
        ("example", 'custom', 'powergraph', 'uk'),
        ("example", 'custom', 'powergraph', 'ieee24'),
        ("example", 'custom', 'powergraph', 'ieee39'),
        # ("example", 'custom', 'powergraph', 'ieee118'),

    ]
    for dataset_full_name in datasets:
        for i in range(3, 9):
            for explainer in explainers:
                for model_name in models:
                    if model_name.startswith('dummy') and explainer != 'GSAT':
                        continue
                    if explainer == 'GSAT' and not (model_name.startswith('dummy')):
                        continue
                    print(f"MODEL: {model_name}, EXPL: {explainer}, i: {i}")
                    run_interpretation_test(explainer, dataset_full_name, model_name, iter=i)
                    # try:
                    #     print(f"Iter: {i}; Model: {model_name}; Dataset: {dataset_full_name[2]}")
                    #     run_interpretation_test(explainer, dataset_full_name, model_name, iter=i)
                    # except Exception as e:
                    #     print(f"ERROR: {e}")
                    #     continue



    # dataset_full_name = ("single-graph", "Amazon", 'Photo')
    # run_interpretation_test(dataset_full_name)
