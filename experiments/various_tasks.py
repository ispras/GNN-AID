import torch
from torch import device

from data_structures.configs import DatasetConfig, DatasetVarConfig, FeatureConfig, Task, \
    ConfigPattern, ModelModificationConfig
from datasets.datasets_manager import DatasetManager
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from models_builder.models_zoo import model_configs_zoo


def regression():
    dc = DatasetConfig(('example', 'example'))
    dvc = DatasetVarConfig(features=FeatureConfig(node_attr=['a']), labeling='regression',
                           task=Task.NODE_REGRESSION, dataset_ver_ind=0)

    gen_dataset = DatasetManager.get_by_config(dc, dvc)

    print(gen_dataset.data)

    gnn = model_configs_zoo(dataset=gen_dataset, model_name='gcn_gcn')
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

    steps_epochs = 10
    my_device = device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=gen_dataset.prepared_dir,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )

    gnn_model_manager.gnn.to(my_device)
    gen_dataset.data.to(my_device)

    gen_dataset.train_test_split()
    gnn_model_manager.train_model(
        gen_dataset=gen_dataset, steps=steps_epochs,
        save_model_flag=False,
        metrics=[Metric("F1", mask='train', average=None)]
    )
    print("Training was successful")


if __name__ == '__main__':
    regression()
