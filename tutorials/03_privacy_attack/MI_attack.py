import numpy as np

from attacks.mi_attacks import MIAttacker
from aux.utils import MI_ATTACK_PARAMETERS_PATH, OPTIMIZERS_PARAMETERS_PATH
from datasets.datasets_manager import DatasetManager
from datasets.ptg_datasets import LibPTGDataset
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from data_structures.configs import ModelModificationConfig, ConfigPattern, DatasetConfig, Task
from models_builder.models_zoo import model_configs_zoo
from torch import manual_seed

manual_seed(1234)

# We load Cora dataset and GCN_2l model as we did it in 01/02 tutorials
gen_dataset_sg_cora = DatasetManager.get_by_config(
    DatasetConfig((LibPTGDataset.data_folder, "Homogeneous", "Planetoid", "Cora")),
    LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.NODE_CLASSIFICATION})
)

gen_dataset_sg_cora.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
results_dataset_path_sg_cora = gen_dataset_sg_cora.prepared_dir

default_config = ModelModificationConfig(
    model_ver_ind=0,
)

manager_config = ConfigPattern(
    _config_class="ModelManagerConfig",
    _config_kwargs={
        "mask_features": [],
        "optimizer": {
            "_config_class": "Config",
            "_class_name": "Adam",
            "_import_path": OPTIMIZERS_PARAMETERS_PATH,
            "_class_import_info": ["torch.optim"],
            "_config_kwargs": {"weight_decay": 5e-4},
        }
    }
)

# MI Attack config being set here
# You can see the available MI attack types and their default parameters in
    # ./metainfo/mi_attack_parameters.json
mi_attack_config = ConfigPattern(
    _class_name="NaiveMIAttacker",
    _import_path=MI_ATTACK_PARAMETERS_PATH,
    _config_class="MIAttackConfig",
    _config_kwargs={
        'threshold': 0.2
    }
)

gcn_gcn_sg_example = model_configs_zoo(dataset=gen_dataset_sg_cora, model_name='gcn_gcn')

gnn_model_manager_sg_cora = FrameworkGNNModelManager(
    gnn=gcn_gcn_sg_example,
    dataset_path=results_dataset_path_sg_cora,
    modification=default_config,
    manager_config=manager_config,
)

gnn_model_manager_sg_cora.set_mi_attacker(mi_attack_config=mi_attack_config)

attack_cnt = 100
# seed = 42
seed = None
if seed is not None:
    np.random.seed(seed)
target_list = np.random.choice(gen_dataset_sg_cora.num_nodes, size=attack_cnt, replace=False)

gnn_model_manager_sg_cora.train_model(gen_dataset=gen_dataset_sg_cora, steps=100,
                                      metrics=[Metric("Accuracy", mask='test')])
mask_loc = Metric.create_mask_by_target_list(y_true=gen_dataset_sg_cora.labels, target_list=target_list)
metric_loc = gnn_model_manager_sg_cora.evaluate_model(gen_dataset=gen_dataset_sg_cora,
                                                      metrics=[Metric("F1", mask=mask_loc, average='macro'),
                                                               Metric("Accuracy", mask=mask_loc)])
print(metric_loc)

for mask, res in gnn_model_manager_sg_cora.mi_attacker.results.items():
    print(f"MI Attack accuracy:"
          f" {MIAttacker.compute_single_attack_accuracy(mask, res, gen_dataset_sg_cora.train_mask)}")