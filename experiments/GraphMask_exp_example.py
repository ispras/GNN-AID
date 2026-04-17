import argparse
import warnings

import torch
from torch import device
from torch.cuda import is_available

from data_structures.configs import ModelModificationConfig, ModelManagerConfig, ConfigPattern, \
    Task, DatasetConfig
from aux.utils import EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH, EXPLAINERS_INIT_PARAMETERS_PATH
from datasets.ptg_datasets import LibPTGDataset
from explainers.explainers_manager import FrameworkExplainersManager
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from datasets.datasets_manager import DatasetManager
import json

from models_builder.models_zoo import model_configs_zoo


class ConfigurationLoader:

    def process_configuration(self, raw_configuration):
        return raw_configuration

    def load(self, filename):
        with open(filename, 'r') as json_file:
            json_data = json_file.read()
            configuration = json.loads(json_data)

        return self.process_configuration(configuration)


def test_graph_mask(configuration):
    my_device = device('cuda' if is_available() else 'cpu')
    my_device = device('cpu')


    dataset = DatasetManager.get_by_config(
        DatasetConfig((LibPTGDataset.data_folder, "Homogeneous", "Planetoid", "Cora")),
        LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.NODE_CLASSIFICATION})
    )

    gcn2 = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')

    gnn_model_manager_config = ModelManagerConfig(**{
        "mask_features": [],
        "optimizer": {
            "_class_name": "Adam",
            "_config_kwargs": {},
        }
    })

    steps_epochs = 200
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gcn2,
        dataset_path=dataset.prepared_dir,
        manager_config=gnn_model_manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )

    data.x = data.x.float()
    gnn_model_manager.gnn.to(my_device)
    data = data.to(my_device)

    # save_model_flag = False
    save_model_flag = True

    warnings.warn("Start training")
    try:
        gnn_model_manager.load_model_executor()
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

    # Explain node 10
    node = 10

    warnings.warn("Start GraphMask")
    explainer_init_config = ConfigPattern(
        _class_name="GraphMask",
        _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
        _config_class="ExplainerInitConfig",
        _config_kwargs={
            # "class_name": "SubgraphX",
        }
    )
    explainer_run_config = ConfigPattern(
        _config_class="ExplainerRunConfig",
        _config_kwargs={
            "mode": "local",
            "kwargs": {
                "_class_name": "GraphMask",
                "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                "_config_class": "Config",
                "_config_kwargs": {

                },
            }
        }
    )
    explainer_GraphMask = FrameworkExplainersManager(
        init_config=explainer_init_config,
        dataset=dataset, gnn_manager=gnn_model_manager,
        explainer_name='GraphMask',
    )
    explainer_GraphMask.conduct_experiment(explainer_run_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train according to a specified configuration file.')
    parser.add_argument("--configuration", default="configurations_graph_mask.json")
    args = parser.parse_args()

    configuration_loader = ConfigurationLoader()
    configuration = configuration_loader.load(args.configuration)

    test_graph_mask(configuration)