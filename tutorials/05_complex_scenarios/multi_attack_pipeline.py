import warnings

import torch
from torch import device

from data_structures.configs import ModelModificationConfig, ConfigPattern
from datasets.datasets_manager import DatasetManager
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from models_builder.models_zoo import model_configs_zoo
from src.aux.utils import POISON_ATTACK_PARAMETERS_PATH, EVASION_ATTACK_PARAMETERS_PATH

my_device = device('cuda' if torch.cuda.is_available() else 'cpu')

# We load Cora dataset and GIN_2l model as we did it in previous tutorials

full_name = ("Homogeneous", "Planetoid", 'Cora')

torch.manual_seed(1234)

dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
    full_name=full_name,
    dataset_ver_ind=0
)

gnn = model_configs_zoo(dataset=dataset, model_name='gin_gin')

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

save_model_flag = False

# data.x = data.x.float()
gnn_model_manager.gnn.to(my_device)
data = data.to(my_device)
dataset.dataset.data.to(my_device)

# Here we set both poison and evasion attacks configs

poison_attack_config = ConfigPattern(
    _class_name="CLGAAttack",
    _import_path=POISON_ATTACK_PARAMETERS_PATH,
    _config_class="PoisonAttackConfig",
    _config_kwargs={
        "num_epochs": 300
    }
)

fgsm_evasion_attack_config = ConfigPattern(
    _class_name="FGSM",
    _import_path=EVASION_ATTACK_PARAMETERS_PATH,
    _config_class="EvasionAttackConfig",
    _config_kwargs={
        "epsilon": 0.002 * 1,
        "is_feature_attack": True,
    }
)

# And pass them to the manager. So both evasion and poison attack can be set simultaneously
gnn_model_manager.set_poison_attacker(poison_attack_config=poison_attack_config)
gnn_model_manager.set_evasion_attacker(evasion_attack_config=fgsm_evasion_attack_config)

warnings.warn("Start training")
dataset.train_test_split()

gnn_model_manager.epochs = gnn_model_manager.modification.epochs = 0
train_test_split_path = gnn_model_manager.train_model(gen_dataset=dataset, steps=steps_epochs,
                                                      save_model_flag=save_model_flag,
                                                      metrics=[Metric("F1", mask='train', average=None)])

if train_test_split_path is not None:
    dataset.save_train_test_mask(train_test_split_path)
    train_mask, val_mask, test_mask, train_test_sizes = torch.load(train_test_split_path / 'train_test_split')[:]
    dataset.train_mask, dataset.val_mask, dataset.test_mask = train_mask, val_mask, test_mask
    data.percent_train_class, data.percent_test_class = train_test_sizes

warnings.warn("Training was successful")

metric_loc = gnn_model_manager.evaluate_model(
    gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro'),
                                  Metric("Accuracy", mask='test')])
print(metric_loc)

