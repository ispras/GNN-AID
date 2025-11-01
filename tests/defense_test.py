import unittest
import numpy as np

# Monkey patch main dirs - before other imports
from aux.utils import monkey_patch_directories
monkey_patch_directories()

from attacks.mi_attacks import MIAttacker
from datasets.datasets_manager import DatasetManager
from datasets.ptg_datasets import LibPTGDataset
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from data_structures.configs import ModelModificationConfig, DatasetConfig, DatasetVarConfig, \
    ConfigPattern, FeatureConfig
from models_builder.models_zoo import model_configs_zoo

from aux.utils import POISON_DEFENSE_PARAMETERS_PATH, \
    OPTIMIZERS_PARAMETERS_PATH, MI_ATTACK_PARAMETERS_PATH, MI_DEFENSE_PARAMETERS_PATH, \
    import_all_from_package

import defenses
import_all_from_package(defenses)  # to import all subclasses properly


class DefenseTest(unittest.TestCase):
    def setUp(self):
        print('setup')

        # Init datasets
        # Single-Graph - Example
        self.gen_dataset_sg_example = DatasetManager.get_by_config(
            DatasetConfig(("example", "example")),
            DatasetVarConfig(features=FeatureConfig(node_attr=['a']),
                             labeling='binary',
                             dataset_ver_ind=0)
        )
        self.gen_dataset_sg_example.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
        self.results_dataset_path_sg_example = self.gen_dataset_sg_example.prepared_dir

        #Single-graph - Cora
        self.gen_dataset_sg_cora, _, results_dataset_path_sg_cora = DatasetManager.get_by_full_name(
            full_name=(LibPTGDataset.data_folder, "Homogeneous", "Planetoid", "Cora"),
            dataset_ver_ind=0
        )

        # self.gen_dataset_sg_cora = DatasetManager.get_by_config(
        #     DatasetConfig(
        #         domain="single-graph",
        #         group="Planetoid",
        #         graph="Cora"),
        #     DatasetVarConfig(dataset_ver_ind=0)
        # )
        self.gen_dataset_sg_cora.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
        self.results_dataset_path_sg_cora = self.gen_dataset_sg_cora.prepared_dir

        self.default_config = ModelModificationConfig(
            model_ver_ind=0,
        )

        self.manager_config = ConfigPattern(
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

    def test_gnnguard(self):
        poison_defense_config = ConfigPattern(
            _class_name="GNNGuard",
            _import_path=POISON_DEFENSE_PARAMETERS_PATH,
            _config_class="PoisonDefenseConfig",
            _config_kwargs={
                # "num_nodes": self.gen_dataset_sg_example.dataset.x.shape[0]  # is there more fancy way?
            }
        )

        gat_gat_sg_example = model_configs_zoo(dataset=self.gen_dataset_sg_example, model_name='gat_gat')

        gnn_model_manager_sg_example = FrameworkGNNModelManager(
            gnn=gat_gat_sg_example,
            dataset_path=self.results_dataset_path_sg_example,
            modification=self.default_config,
            manager_config=self.manager_config,
        )

        gnn_model_manager_sg_example.set_poison_defender(poison_defense_config=poison_defense_config)

        gnn_model_manager_sg_example.train_model(gen_dataset=self.gen_dataset_sg_example, steps=100,
                                                 metrics=[Metric("Accuracy", mask='test')])
        metric_loc = gnn_model_manager_sg_example.evaluate_model(gen_dataset=self.gen_dataset_sg_example,
                                                                 metrics=[Metric("F1", mask='test', average='macro')])
        print(metric_loc)

    def test_noise_mi_defender_cora(self):
        mi_attack_config = ConfigPattern(
            _class_name="NaiveMIAttacker",
            _import_path=MI_ATTACK_PARAMETERS_PATH,
            _config_class="MIAttackConfig",
            _config_kwargs={
                'threshold': 0.3
            }
        )

        mi_defense_config = ConfigPattern(
            _class_name="NoiseMIDefender",
            _import_path=MI_DEFENSE_PARAMETERS_PATH,
            _config_class="MIDefenseConfig",
            _config_kwargs={
                'temperature': 50
            }
        )

        gcn_gcn_sg_cora = model_configs_zoo(dataset=self.gen_dataset_sg_cora, model_name='gcn_gcn')

        gnn_model_manager_sg_cora = FrameworkGNNModelManager(
            gnn=gcn_gcn_sg_cora,
            dataset_path=self.results_dataset_path_sg_cora,
            modification=self.default_config,
            manager_config=self.manager_config,
        )

        gnn_model_manager_sg_cora.set_mi_attacker(mi_attack_config=mi_attack_config)
        gnn_model_manager_sg_cora.set_mi_defender(mi_defense_config=mi_defense_config)

        attack_cnt = 100
        # seed = 42
        seed = None
        if seed is not None:
            np.random.seed(seed)
        target_list = np.random.choice(self.gen_dataset_sg_cora.dataset.data.x.shape[0], size=attack_cnt, replace=False)

        gnn_model_manager_sg_cora.train_model(gen_dataset=self.gen_dataset_sg_cora, steps=100,
                                              metrics=[Metric("Accuracy", mask='test')])
        mask_loc = Metric.create_mask_by_target_list(y_true=self.gen_dataset_sg_cora.labels, target_list=target_list)
        metric_loc = gnn_model_manager_sg_cora.evaluate_model(gen_dataset=self.gen_dataset_sg_cora,
                                                              metrics=[Metric("F1", mask=mask_loc, average='macro'),
                                                                       Metric("Accuracy", mask=mask_loc)])
        print(metric_loc)

        for mask, res in gnn_model_manager_sg_cora.mi_attacker.results.items():
            print(f"MI Attack accuracy:"
                  f" {MIAttacker.compute_single_attack_accuracy(mask, res, self.gen_dataset_sg_cora.train_mask)}")


if __name__ == '__main__':
    unittest.main()
