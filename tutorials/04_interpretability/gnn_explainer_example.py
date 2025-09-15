import torch
from torch import device

from data_structures.configs import ConfigPattern
from aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH
from explainers.explainers_manager import FrameworkExplainersManager

from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from base.datasets_processing import DatasetManager
from models_builder.models_zoo import model_configs_zoo


def gnnexplainer_test():
    my_device = device('cuda' if torch.cuda.is_available() else 'cpu')

    full_name = ("single-graph", "Planetoid", 'Cora')
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

    metric_loc = gnn_model_manager.evaluate_model(
        gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro')])
    print(metric_loc)

    # --- Explain model ---
    # Here we can choose the desired explainer and set the parameters of the class constructor of our explainer.
    # If you do not specify parameters, they will be set by default. You can see the default init explainer parameter
    # values in ./metainfo/explainers_init_parameters.json
    explainer_init_config = ConfigPattern(
        _class_name="GNNExplainer(torch-geom)",
        _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
        _config_class="ExplainerInitConfig",
        _config_kwargs={
            "epochs": 150
        }
    )
    # Define an explainer manager class.
    explainer_GNNExpl = FrameworkExplainersManager(
        init_config=explainer_init_config,
        dataset=dataset, gnn_manager=gnn_model_manager,
        explainer_name='GNNExplainer(torch-geom)',
    )
    # Here we can specify the run parameters of the explainer, such as the number of the element from the dataset
    # for which we want to get an explanation. You can see the default init explainer parameter values in
    # ./metainfo/explainers_local_run_parameters.json and ./metainfo/explainers_global_run_parameters.json
    # for the local and global explainer types respectively
    explainer_run_config = ConfigPattern(
        _config_class="ExplainerRunConfig",
        _config_kwargs={
            "mode": "local",
            "kwargs": {
                "_class_name": "GNNExplainer(torch-geom)",
                "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                "_config_class": "Config",
                "_config_kwargs": {
                    "element_idx": 1
                },
            }
        }
    )
    # Now we can run the experiment and get an explanation of how the model works on the input data sample.
    # The resulting explanation is saved in the experiments folder and has its own unique number for the given
    # dataset, model, and explainer configurations.
    explainer_GNNExpl.conduct_experiment(explainer_run_config)


if __name__ == '__main__':
    import random

    random.seed(10)
    gnnexplainer_test()