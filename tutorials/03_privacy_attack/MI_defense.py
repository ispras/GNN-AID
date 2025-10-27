import numpy as np

from attacks.mi_attacks import MIAttacker
from aux.utils import MI_ATTACK_PARAMETERS_PATH, OPTIMIZERS_PARAMETERS_PATH, MI_DEFENSE_PARAMETERS_PATH
from base.datasets_processing import DatasetManager
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from data_structures.configs import ModelModificationConfig, ConfigPattern
from models_builder.models_zoo import model_configs_zoo
from torch import manual_seed

manual_seed(1234)

# Load Cora dataset and GCN_2l model
dataset_sg_cora, _, results_dataset_path_sg_cora = DatasetManager.get_by_full_name(
    full_name=("single-graph", "Planetoid", "Cora",),
    dataset_ver_ind=0
)

gen_dataset_sg_cora = dataset_sg_cora
gen_dataset_sg_cora.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
results_dataset_path_sg_cora = gen_dataset_sg_cora.results_dir

default_config = ModelModificationConfig(
    model_ver_ind=0,
)

mi_attack_config = ConfigPattern(
    _class_name="NaiveMIAttacker",
    _import_path=MI_ATTACK_PARAMETERS_PATH,
    _config_class="MIAttackConfig",
    _config_kwargs={
        'threshold': 0.3
    }
)

# MI Defense config being set here
# You can see the available MI defense types and their default parameters in
    # ./metainfo/mi_defense_parameters.json

mi_defense_config = ConfigPattern(
    _class_name="NoiseMIDefender",
    _import_path=MI_DEFENSE_PARAMETERS_PATH,
    _config_class="MIDefenseConfig",
    _config_kwargs={
        'temperature': 50
    }
)

gcn_gcn_sg_cora = model_configs_zoo(dataset=gen_dataset_sg_cora, model_name='gcn_gcn')

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

gnn_model_manager_sg_cora = FrameworkGNNModelManager(
    gnn=gcn_gcn_sg_cora,
    dataset_path=results_dataset_path_sg_cora,
    modification=default_config,
    manager_config=manager_config,
)

# The attacker and the defender being passed to the model manager
gnn_model_manager_sg_cora.set_mi_attacker(mi_attack_config=mi_attack_config)
gnn_model_manager_sg_cora.set_mi_defender(mi_defense_config=mi_defense_config)

attack_cnt = 100
# seed = 42
seed = None
if seed is not None:
    np.random.seed(seed)
target_list = np.random.choice(gen_dataset_sg_cora.dataset.data.x.shape[0], size=attack_cnt, replace=False)

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