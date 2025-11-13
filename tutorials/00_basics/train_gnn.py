import torch
from pathlib import Path
from torch import device

from datasets.ptg_datasets import LibPTGDataset
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from data_structures.configs import ConfigPattern, DatasetConfig, Task
from datasets.datasets_manager import DatasetManager
from models_builder.models_zoo import model_configs_zoo


def train_gnn():
    my_device = device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load dataset ---
    # Here you can upload your own dataset, specified in the basic input data format.
    # It is also possible to use most of the existing datasets in the PyG.
    # You can see all supported datasets in the ./metainfo/torch_geom_index_all.json
    gen_dataset = DatasetManager.get_by_config(
        DatasetConfig((LibPTGDataset.data_folder, "Homogeneous", "Planetoid", "Cora")),
        LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.NODE_CLASSIFICATION})
    )
    gen_dataset.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
    gen_dataset.data.to(my_device)

    # --- Construct model ---
    # Here you can use your own model, or quickly construct a gnn using the GNN-AID functionality for constructing
    # models. Examples of model configurations can be found in ./src/models_builder/models_zoo
    # As an example we will use a simple two-layer GNN.
    gnn = model_configs_zoo(dataset=gen_dataset, model_name='gcn_gcn')

    # Here we can set the configuration of the optimizer parameters. If you do not specify parameters, they will
    # be set by default. You can see the default optimizer parameter values in ./metainfo/optimizers_parameters.json
    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "mask_features": [],
            "optimizer": {
                "_class_name": "Adam",
                "_config_kwargs": {
                    "lr": 0.01
                },
            }
        }
    )

    # Create a class object that allows us to train our model.
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=gen_dataset.prepared_dir,
        manager_config=manager_config,
    )
    gnn_model_manager.gnn.to(my_device)

    # --- Train model ---
    gnn_model_manager.train_model(gen_dataset=gen_dataset, steps=200)

    # Save weights of our model
    gnn_model_manager.save_model(path=str(Path(__file__).resolve().parent) + '/weights')

    # Let's load our model and measure the quality metric
    gnn_model_manager.load_model(path=str(Path(__file__).resolve().parent) + '/weights')

    metric_loc = gnn_model_manager.evaluate_model(
        gen_dataset=gen_dataset, metrics=[Metric("F1", mask='test', average='macro'),
                                      Metric("Accuracy", mask='test')])
    print(metric_loc)


if __name__ == '__main__':
    import random

    random.seed(10)
    train_gnn()