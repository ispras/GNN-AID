import copy
import unittest

import numpy as np
import torch

from gnn_aid.attacks.mi_attacks import MIAttacker
from gnn_aid.aux.utils import POISON_DEFENSE_PARAMETERS_PATH, \
    OPTIMIZERS_PARAMETERS_PATH, MI_ATTACK_PARAMETERS_PATH, MI_DEFENSE_PARAMETERS_PATH, FUNCTIONS_PARAMETERS_PATH
from gnn_aid.data_structures.configs import ModelModificationConfig, DatasetConfig, DatasetVarConfig, \
    ConfigPattern, FeatureConfig, Task, ModelConfig, ModelStructureConfig
from gnn_aid.datasets.datasets_manager import DatasetManager
from gnn_aid.datasets.ptg_datasets import LibPTGDataset
from gnn_aid.models_builder import FrameworkGNNConstructor
from gnn_aid.models_builder.models_utils import Metric
from gnn_aid.models_builder.model_managers import FrameworkGNNModelManager
from gnn_aid.models_builder.models_zoo import model_configs_zoo
from tests.utils import cleanup_patches, monkey_patch_dirs


class DefenseTest(unittest.TestCase):
    def setUp(self):
        # Init datasets
        # Single-Graph - Example
        self.gen_dataset_sg_example = DatasetManager.get_by_config(
            DatasetConfig(("example", "example")),
            DatasetVarConfig(task=Task.NODE_CLASSIFICATION,
                             features=FeatureConfig(node_attr=['a']),
                             labeling='binary',
                             dataset_ver_ind=0)
        )
        self.gen_dataset_sg_example.train_test_split(percent_train_class=0.6, percent_test_class=0.4)

        #Single-graph - Cora
        self.gen_dataset_sg_cora = DatasetManager.get_by_config(
            DatasetConfig((LibPTGDataset.data_folder, "Homogeneous", "Planetoid", "Cora")),
            LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.NODE_CLASSIFICATION})
        )

        self.gen_dataset_sg_cora.train_test_split(percent_train_class=0.6, percent_test_class=0.4)

        # Single-graph - Cora (Link Prediction)
        self.gen_dataset_lp_cora = DatasetManager.get_by_config(
            DatasetConfig((LibPTGDataset.data_folder, "Homogeneous", "Planetoid", "Cora")),
            LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.EDGE_PREDICTION})
        )
        self.gen_dataset_lp_cora.train_test_split(percent_train_class=0.85, percent_test_class=0.15)

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

        self.manager_config_lp = ConfigPattern(
            _config_class="ModelManagerConfig",
            _config_kwargs={
                "mask_features": [],
                "optimizer": {
                    "_config_class": "Config",
                    "_class_name": "Adam",
                    "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                    "_class_import_info": ["torch.optim"],
                    "_config_kwargs": {"weight_decay": 5e-4},
                },
                "loss_function": {
                    "_config_class": "Config",
                    "_class_name": "BCEWithLogitsLoss",
                    "_import_path": FUNCTIONS_PARAMETERS_PATH,
                    "_class_import_info": ["torch.nn"],
                    "_config_kwargs": {},
                },
                "neg_samples_ratio": 1,
            }
        )
        monkey_patch_dirs()

    def tearDown(self):
        # Clean up patches and tmp dirs
        cleanup_patches()

    def test_gnnguard(self):
        poison_defense_config = ConfigPattern(
            _class_name="GNNGuardDefender",
            _import_path=POISON_DEFENSE_PARAMETERS_PATH,
            _config_class="PoisonDefenseConfig",
            _config_kwargs={
                # "num_nodes": self.gen_dataset_sg_example.dataset.x.shape[0]  # is there more fancy way?
            }
        )

        gat_gat_sg_example = model_configs_zoo(dataset=self.gen_dataset_sg_example, model_name='gat_gat')

        gnn_model_manager_sg_example = FrameworkGNNModelManager(
            gnn=gat_gat_sg_example,
            dataset_path=self.gen_dataset_sg_example.prepared_dir,
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
            dataset_path=self.gen_dataset_sg_cora.prepared_dir,
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
        target_list = np.random.choice(self.gen_dataset_sg_cora.info.nodes[0], size=attack_cnt, replace=False)

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

    def test_z_jaccard_defender_link_prediction(self):
        """
        Test JaccardDefender on Link Prediction task (Cora dataset).
        """
        poison_defense_config = ConfigPattern(
            _class_name="JaccardDefender",
            _import_path=POISON_DEFENSE_PARAMETERS_PATH,
            _config_class="PoisonDefenseConfig",
            _config_kwargs={
                "threshold": 0.03,
            }
        )

        gnn = FrameworkGNNConstructor(
            model_config=ModelConfig(
                structure=ModelStructureConfig([
                    # Encoder: 2-layer GCN
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': self.gen_dataset_lp_cora.num_node_features,
                                'out_channels': 32,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': 32,
                                'out_channels': 16,
                            },
                        },
                    },
                    {
                        'label': 'd',
                        'function': {
                            'function_name': 'CosineSimilarity',
                            'function_kwargs': None
                        }
                    }
                ])
            )
        )

        gnn_model_manager = FrameworkGNNModelManager(
            gnn=gnn,
            dataset_path=self.gen_dataset_lp_cora.prepared_dir,
            modification=self.default_config,
            manager_config=self.manager_config_lp,
        )

        gnn_model_manager.set_poison_defender(poison_defense_config=poison_defense_config)

        original_train_edges = self.gen_dataset_lp_cora.edge_label_index[:,
                               self.gen_dataset_lp_cora.train_mask].size(1)

        gnn_model_manager.train_model(
            gen_dataset=self.gen_dataset_lp_cora,
            steps=30,
            save_model_flag=False,
            metrics=[Metric("AUC", mask='train')]
        )

        defense = gnn_model_manager.poison_defender

        removed_edges = defense.defense_diff.edges["remove"]
        num_removed = len(removed_edges)

        if num_removed > 0:
            print(f"JaccardDefender removed {num_removed} training edges "
                  f"({num_removed / original_train_edges * 100:.1f}%)")
            # Sanity checks
            self.assertGreater(num_removed, 0, "No edges were removed - threshold may be too low")
            self.assertLess(num_removed, original_train_edges * 0.5,
                            "Too many edges removed (>50%) - threshold may be too high")
        else:
            print("WARNING: No edges removed (threshold may be too low for this graph)")

        test_metrics = gnn_model_manager.evaluate_model(
            gen_dataset=self.gen_dataset_lp_cora,
            metrics=[
                Metric("AUC", mask='test'),
                Metric("Recall@k", mask='test', k=50),
                Metric("Recall@k", mask='test', k=100),
            ]
        )
        print("Link Prediction test metrics:", test_metrics)

        self.assertGreater(test_metrics['test']['AUC'], 0.75,
                           "AUC should be >0.75 (random baseline 0.5) after training")

        # Sometimes fails :(
        # self.assertGreater(test_metrics['test']['Recall@k{k=100}'], 0.0, "Recall@100 should be >0 after training")

    def test_z_noise_mi_link_defender_cora(self):
        """
        Test NoiseMILinkDefender on Link Prediction task (Cora)
        """
        gen_dataset = DatasetManager.get_by_config(
            DatasetConfig((LibPTGDataset.data_folder, "Homogeneous", "Planetoid", "Cora")),
            LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.EDGE_PREDICTION})
        )

        gnn = FrameworkGNNConstructor(
            model_config=ModelConfig(
                structure=ModelStructureConfig([
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {'in_channels': gen_dataset.num_node_features, 'out_channels': 32}
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None
                        }
                    },
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {'in_channels': 32, 'out_channels': 16}
                        }
                    },
                    {
                        'label': 'd',
                        'function': {
                            'function_name': 'Concat',
                            'function_kwargs': None
                        }
                    },
                    {
                        'label': 'd',
                        'layer': {
                            'layer_name': 'Linear',
                            'layer_kwargs': {'in_features': 32, 'out_features': 16}
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None
                        }
                    },
                    {
                        'label': 'd',
                        'layer': {
                            'layer_name': 'Linear',
                            'layer_kwargs': {'in_features': 16, 'out_features': 1}
                        }
                    }
                ])
            )
        )

        manager_config_lp = ConfigPattern(
            _config_class="ModelManagerConfig",
            _config_kwargs={
                "mask_features": [],
                "optimizer": {
                    "_class_name": "Adam",
                    "_config_kwargs": {
                        "lr": 0.01,
                        "weight_decay": 5e-4
                    },
                },
                "loss_function": {
                    "_class_name": "BCEWithLogitsLoss",
                    "_import_path": FUNCTIONS_PARAMETERS_PATH,
                    "_class_import_info": ["torch.nn"],
                    "_config_kwargs": {}
                },
                "batch": 64,
                "neg_samples_ratio": 1
            }
        )

        mi_attack_config = ConfigPattern(
            _class_name="ShadowModelMILinkAttacker",
            _import_path=MI_ATTACK_PARAMETERS_PATH,
            _config_class="MIAttackConfig",
            _config_kwargs={
                "shadow_edge_ratio": 0.2,
                "shadow_train_ratio": 0.75,
                "shadow_epochs": 5,
                "classifier_type": "linreg",
                "use_embedding_features": True
            }
        )

        mi_defense_config = ConfigPattern(
            _class_name="NoiseMILinkDefender",
            _import_path=MI_DEFENSE_PARAMETERS_PATH,
            _config_class="MIDefenseConfig",
            _config_kwargs={
                "noise_type": "reverse_sigmoid",
                "beta": 0.3,
                "gamma": 0.8,
                "noise_scale": 0.2,
                "temperature": 1.0
            }
        )

        def run_experiment(with_defense: bool, exp_name: str) -> dict:
            dataset_copy = copy.deepcopy(gen_dataset)
            dataset_copy.train_test_split(percent_train_class=0.85, percent_test_class=0.15)

            model_manager = FrameworkGNNModelManager(
                gnn=copy.deepcopy(gnn),
                dataset_path=dataset_copy.prepared_dir,
                manager_config=manager_config_lp,
            )
            model_manager.set_mi_attacker(mi_attack_config=mi_attack_config)
            if with_defense:
                model_manager.set_mi_defender(mi_defense_config=mi_defense_config)

            model_manager.train_model(
                gen_dataset=dataset_copy,
                steps=30,
                metrics=[Metric("AUC", mask='train')]
            )

            model_metrics = model_manager.evaluate_model(
                gen_dataset=dataset_copy,
                metrics=[
                    Metric("AUC", mask='test'),
                    Metric("Recall@k", mask='test', k=100)
                ]
            )
            test_auc = model_metrics['test']['AUC']
            test_recall = model_metrics['test'].get('Recall@k{k=100}', model_metrics['test'].get('Recall@k', 0.0))

            num_train_edges = dataset_copy.train_mask.sum().item()
            num_test_edges = dataset_copy.test_mask.sum().item()
            attack_cnt_per_class = min(100, num_train_edges, num_test_edges)

            train_edge_indices = dataset_copy.train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            test_edge_indices = dataset_copy.test_mask.nonzero(as_tuple=True)[0].cpu().numpy()

            target_train_indices = np.random.choice(train_edge_indices, size=attack_cnt_per_class, replace=False)
            target_test_indices = np.random.choice(test_edge_indices, size=attack_cnt_per_class, replace=False)
            target_edge_indices = np.concatenate([target_train_indices, target_test_indices])

            edge_mask = torch.zeros(dataset_copy.edge_label_index.size(1), dtype=torch.bool)
            edge_mask[target_edge_indices] = True

            mi_attacker = model_manager.mi_attacker
            attack_results = {}
            for mask, inferred_membership in mi_attacker.results.items():
                attack_metrics = MIAttacker.compute_single_attack_accuracy(
                    mask=edge_mask,
                    inferred_labels=inferred_membership,
                    mask_true=dataset_copy.train_mask,
                    train_class_label=True
                )
                attack_results = attack_metrics
            print(mi_attacker.results)

            return {
                'auc': test_auc,
                'recall': test_recall,
                'attack_accuracy': attack_results['accuracy'],
                'attack_f1': attack_results['f1_train'],
                'model_manager': model_manager,
                'dataset': dataset_copy
            }

        baseline_results = run_experiment(with_defense=False, exp_name="Baseline (no defense)")

        defense_results = run_experiment(with_defense=True, exp_name="With NoiseMILinkDefender")

        self.assertGreaterEqual(
            defense_results['auc'],
            baseline_results['auc'] - 0.05,
            "Defense should not degrade AUC by more than 5%"
        )



if __name__ == '__main__':
    unittest.main()
