import torch
from pathlib import Path
from torch import device

from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from data_structures.configs import ConfigPattern
from datasets_block.datasets_manager import DatasetManager
from models_builder.models_zoo import model_configs_zoo


def train_gnn():
    my_device = device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load dataset ---
    # Here you can upload your own dataset, specified in the basic input data format.
    # It is also possible to use most of the existing datasets in the PyG.
    # You can see all supported datasets in the ./metainfo/torch_geom_index_all.json
    full_name = ("Homogeneous", "Planetoid", 'Cora')
    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=full_name,
        dataset_ver_ind=0
    )
    dataset.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
    dataset.dataset.data.to(my_device)

    # --- Construct model ---
    # Here you can use your own model, or quickly construct a gnn using the GNN-AID functionality for constructing
    # models. Examples of model configurations can be found in ./src/models_builder/models_zoo
    # As an example we will use a simple two-layer GNN.
    gnn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')

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
        dataset_path=results_dataset_path,
        manager_config=manager_config,
    )
    gnn_model_manager.gnn.to(my_device)

    # --- Train model ---
    gnn_model_manager.train_model(gen_dataset=dataset, steps=200)

    # Save weights of our model
    gnn_model_manager.save_model(path=str(Path(__file__).resolve().parent) + '/weights')

    # Let's load our model and measure the quality metric
    gnn_model_manager.load_model(path=str(Path(__file__).resolve().parent) + '/weights')

    metric_loc = gnn_model_manager.evaluate_model(
        gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro'),
                                      Metric("Accuracy", mask='test')])
    print(metric_loc)


if __name__ == '__main__':
    import random

    random.seed(10)
    train_gnn()