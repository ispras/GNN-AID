import unittest
import numpy as np
import torch

from attacks.mi_attacks import MIAttacker
from datasets.datasets_manager import DatasetManager
from datasets.ptg_datasets import LibPTGDataset
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from data_structures.configs import ModelModificationConfig, DatasetConfig, DatasetVarConfig, \
    ConfigPattern, FeatureConfig, Task
from models_builder.models_zoo import model_configs_zoo
from aux.utils import POISON_ATTACK_PARAMETERS_PATH, EVASION_ATTACK_PARAMETERS_PATH, \
    OPTIMIZERS_PARAMETERS_PATH, MI_ATTACK_PARAMETERS_PATH
from tests.utils import monkey_patch_dirs, cleanup_patches


class AttacksTest(unittest.TestCase):
    def setUp(self):
        # os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Monkey for home coding

        print('setup')
        self.my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Init datasets
        # Multi-Graphs - Example
        self.gen_dataset_mg_small = DatasetManager.get_by_config(
            DatasetConfig(('example', 'example8')),
            DatasetVarConfig(task=Task.GRAPH_CLASSIFICATION,
                             features=FeatureConfig(node_attr=['a']),
                             labeling='binary',
                             dataset_ver_ind=0)
        )

        self.gen_dataset_mg_small.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
        self.results_dataset_path_mg_small = self.gen_dataset_mg_small.prepared_dir
        self.gen_dataset_mg_small.data.to(self.my_device)


        # Single-Graph - Example
        self.gen_dataset_sg_example = DatasetManager.get_by_config(
            DatasetConfig(("example", "example")),
            DatasetVarConfig(task=Task.NODE_CLASSIFICATION,
                             features=FeatureConfig(node_attr=['a']),
                             labeling='binary',
                             dataset_ver_ind=0)
        )
        self.gen_dataset_sg_example.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
        self.gen_dataset_sg_example.data.to(self.my_device)

        # Single-graph - Cora
        self.gen_dataset_sg_cora = DatasetManager.get_by_config(
            DatasetConfig((LibPTGDataset.data_folder, "Homogeneous", "Planetoid", "Cora")),
            LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.NODE_CLASSIFICATION})
        )
        self.gen_dataset_sg_cora.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
        self.gen_dataset_sg_cora.data.to(self.my_device)

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

        monkey_patch_dirs()

    def tearDown(self):
        # Clean up patches and tmp dirs
        cleanup_patches()


    def test_metattack_full(self):
        poison_attack_config = ConfigPattern(
            _class_name="MetaAttackFull",
            _import_path=POISON_ATTACK_PARAMETERS_PATH,
            _config_class="PoisonAttackConfig",
            _config_kwargs={
                "num_nodes": self.gen_dataset_sg_example.dataset.x.shape[0]  # is there more fancy way?
            }
        )

        gat_gat_sg_example = model_configs_zoo(dataset=self.gen_dataset_sg_example, model_name='gat_gat')

        gnn_model_manager_sg_example = FrameworkGNNModelManager(
            gnn=gat_gat_sg_example,
            dataset_path=self.gen_dataset_sg_example.prepared_dir,
            modification=self.default_config,
            manager_config=self.manager_config,
        )

        gnn_model_manager_sg_example.set_poison_attacker(poison_attack_config=poison_attack_config)

        gnn_model_manager_sg_example.train_model(gen_dataset=self.gen_dataset_sg_example, steps=100,
                                                 metrics=[Metric("Accuracy", mask='test')])
        metric_loc = gnn_model_manager_sg_example.evaluate_model(gen_dataset=self.gen_dataset_sg_example,
                                                                 metrics=[Metric("F1", mask='test', average='macro')])
        print(metric_loc)

    def test_metattack_bug(self):
        poison_attack_config = ConfigPattern(
            _class_name="MetaAttackApprox",
            _import_path=POISON_ATTACK_PARAMETERS_PATH,
            _config_class="PoisonAttackConfig",
            _config_kwargs={
                "attack_structure": True,
                "attack_features": True,
                "num_nodes": self.gen_dataset_sg_example.dataset.x.shape[0]  # is there more fancy way?
            }
        )

        gcn_sg_example = model_configs_zoo(dataset=self.gen_dataset_sg_example, model_name='gcn')

        gnn_model_manager_sg_example = FrameworkGNNModelManager(
            gnn=gcn_sg_example,
            dataset_path=self.gen_dataset_sg_example.prepared_dir,
            modification=self.default_config,
            manager_config=self.manager_config,
        )

        gnn_model_manager_sg_example.set_poison_attacker(poison_attack_config=poison_attack_config)

        gnn_model_manager_sg_example.train_model(gen_dataset=self.gen_dataset_sg_example, steps=100,
                                                 metrics=[Metric("Accuracy", mask='test')])
        metric_loc = gnn_model_manager_sg_example.evaluate_model(gen_dataset=self.gen_dataset_sg_example,
                                                                 metrics=[Metric("F1", mask='test', average='macro')])
        print(metric_loc)

    def test_metattack_approx(self):
        torch.manual_seed(100)  # DEBUG

        poison_attack_config = ConfigPattern(
            _class_name="MetaAttackApprox",
            _import_path=POISON_ATTACK_PARAMETERS_PATH,
            _config_class="PoisonAttackConfig",
            _config_kwargs={
                "num_nodes": self.gen_dataset_sg_example.dataset.x.shape[0]  # is there more fancy way?
            }
        )

        gat_gat_sg_example = model_configs_zoo(dataset=self.gen_dataset_sg_example, model_name='gat_gat')

        gnn_model_manager_sg_example = FrameworkGNNModelManager(
            gnn=gat_gat_sg_example,
            dataset_path=self.gen_dataset_sg_example.prepared_dir,
            modification=self.default_config,
            manager_config=self.manager_config,
        )

        gnn_model_manager_sg_example.set_poison_attacker(poison_attack_config=poison_attack_config)

        gnn_model_manager_sg_example.train_model(gen_dataset=self.gen_dataset_sg_example, steps=100,
                                                 metrics=[Metric("Accuracy", mask='test')])
        metric_loc = gnn_model_manager_sg_example.evaluate_model(gen_dataset=self.gen_dataset_sg_example,
                                                                 metrics=[Metric("F1", mask='test', average='macro'),
                                                                          Metric("Accuracy", mask='test')])
        print(metric_loc)

    def test_qattack_Cora(self):
        # TODO complete test?

        gcn_gcn_sg_cora = model_configs_zoo(dataset=self.gen_dataset_sg_cora, model_name='gcn_gcn')

        gnn_model_manager_sg_cora = FrameworkGNNModelManager(
            gnn=gcn_gcn_sg_cora,
            dataset_path=self.gen_dataset_sg_cora.prepared_dir,
            modification=self.default_config,
            manager_config=self.manager_config,
        )

        evasion_attack_config = ConfigPattern(
            _class_name="QAttack",
            _import_path=EVASION_ATTACK_PARAMETERS_PATH,
            _config_class="EvasionAttackConfig",
            _config_kwargs={
            }
        )

        gnn_model_manager_sg_cora.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

        gnn_model_manager_sg_cora.train_model(gen_dataset=self.gen_dataset_sg_cora, steps=100,
                                              metrics=[Metric("Accuracy", mask='test')])
        metric_loc = gnn_model_manager_sg_cora.evaluate_model(gen_dataset=self.gen_dataset_sg_cora,
                                                              metrics=[Metric("F1", mask='test', average='macro'),
                                                                       Metric("Accuracy", mask='test')])
        print(metric_loc)

    def test_qattack_example(self):
        gcn_gcn_sg_example = model_configs_zoo(dataset=self.gen_dataset_sg_example, model_name='gcn_gcn')

        gnn_model_manager_sg_example = FrameworkGNNModelManager(
            gnn=gcn_gcn_sg_example,
            dataset_path=self.gen_dataset_sg_example.prepared_dir,
            modification=self.default_config,
            manager_config=self.manager_config,
        )

        evasion_attack_config = ConfigPattern(
            _class_name="QAttack",
            _import_path=EVASION_ATTACK_PARAMETERS_PATH,
            _config_class="EvasionAttackConfig",
            _config_kwargs={
                'individual_size': 3,
            }
        )

        gnn_model_manager_sg_example.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

        gnn_model_manager_sg_example.train_model(gen_dataset=self.gen_dataset_sg_example, steps=100,
                                                 metrics=[Metric("Accuracy", mask='test')])
        metric_loc = gnn_model_manager_sg_example.evaluate_model(gen_dataset=self.gen_dataset_sg_example,
                                                                 metrics=[Metric("F1", mask='test', average='macro'),
                                                                          Metric("Accuracy", mask='test')])
        print(metric_loc)

    def test_mi_naive(self):
        mi_attack_config = ConfigPattern(
            _class_name="NaiveMIAttacker",
            _import_path=MI_ATTACK_PARAMETERS_PATH,
            _config_class="MIAttackConfig",
            _config_kwargs={
                'threshold': 0.61
            }
        )

        gcn_gcn_sg_example = model_configs_zoo(dataset=self.gen_dataset_sg_example, model_name='gcn_gcn')

        gnn_model_manager_sg_example = FrameworkGNNModelManager(
            gnn=gcn_gcn_sg_example,
            dataset_path=self.gen_dataset_sg_example.prepared_dir,
            modification=self.default_config,
            manager_config=self.manager_config,
        )

        gnn_model_manager_sg_example.set_mi_attacker(mi_attack_config=mi_attack_config)

        attack_cnt = 2
        # seed = 42
        seed = None
        if seed is not None:
            np.random.seed(seed)
        target_list = np.random.choice(self.gen_dataset_sg_example.info.nodes[0], size=attack_cnt,
                                       replace=False)

        gnn_model_manager_sg_example.train_model(gen_dataset=self.gen_dataset_sg_example, steps=100,
                                                 metrics=[Metric("Accuracy", mask='test')])
        mask_loc = Metric.create_mask_by_target_list(y_true=self.gen_dataset_sg_example.labels, target_list=target_list)
        metric_loc = gnn_model_manager_sg_example.evaluate_model(gen_dataset=self.gen_dataset_sg_example,
                                                                 metrics=[Metric("F1", mask=mask_loc, average='macro'),
                                                                          Metric("Accuracy", mask=mask_loc)])
        print(metric_loc)

        for mask, res in gnn_model_manager_sg_example.mi_attacker.results.items():
            print(f"MI Attack accuracy:"
                  f" {MIAttacker.compute_single_attack_accuracy(mask, res, self.gen_dataset_sg_example.data.y)}")

    def test_mi_naive_cora(self):
        mi_attack_config = ConfigPattern(
            _class_name="NaiveMIAttacker",
            _import_path=MI_ATTACK_PARAMETERS_PATH,
            _config_class="MIAttackConfig",
            _config_kwargs={
                'threshold': 0.2
            }
        )

        gcn_gcn_sg_example = model_configs_zoo(dataset=self.gen_dataset_sg_cora, model_name='gcn_gcn')

        gnn_model_manager_sg_cora = FrameworkGNNModelManager(
            gnn=gcn_gcn_sg_example,
            dataset_path=self.gen_dataset_sg_cora.prepared_dir,
            modification=self.default_config,
            manager_config=self.manager_config,
        )

        gnn_model_manager_sg_cora.set_mi_attacker(mi_attack_config=mi_attack_config)

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

    def test_fgsm_SG(self):
        gcn_gcn = model_configs_zoo(dataset=self.gen_dataset_sg_example, model_name='gcn_gcn')

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
            gnn=gcn_gcn,
            dataset_path=self.gen_dataset_sg_example.prepared_dir,
            manager_config=manager_config,
            modification=ModelModificationConfig(model_ver_ind=0, epochs=0)
        )

        gnn_model_manager.gnn.to(self.my_device)
        gnn_model_manager.train_model(gen_dataset=self.gen_dataset_sg_example, steps=200, save_model_flag=False)

        # ---------- Attack on structure ----------
        evasion_attack_config = ConfigPattern(
            _class_name="FGSM",
            _import_path=EVASION_ATTACK_PARAMETERS_PATH,
            _config_class="EvasionAttackConfig",
            _config_kwargs={
                "is_feature_attack": False,
                "element_idx": 0,
                "epsilon": 0.5,
            }
        )

        gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

        # Attack
        _ = gnn_model_manager.evaluate_model(gen_dataset=self.gen_dataset_sg_example,
                                             metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
        # ---------- ------------------- ----------

        # ---------- Attack on feature ----------
        evasion_attack_config = ConfigPattern(
            _class_name="FGSM",
            _import_path=EVASION_ATTACK_PARAMETERS_PATH,
            _config_class="EvasionAttackConfig",
            _config_kwargs={
                "is_feature_attack": True,
                "element_idx": 0,
                "epsilon": 0.5,
            }
        )

        gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

        # Attack
        _ = gnn_model_manager.evaluate_model(gen_dataset=self.gen_dataset_sg_example,
                                             metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
        # ---------- ----------------- ----------

    def test_fgsm_MG(self):
        gcn_gcn = model_configs_zoo(dataset=self.gen_dataset_mg_small, model_name='gin_gin_gin_lin_lin_con')

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
            gnn=gcn_gcn,
            dataset_path=self.results_dataset_path_mg_small,
            manager_config=manager_config,
            modification=ModelModificationConfig(model_ver_ind=0, epochs=0)
        )

        gnn_model_manager.gnn.to(self.my_device)
        gnn_model_manager.train_model(gen_dataset=self.gen_dataset_mg_small, steps=200, save_model_flag=False)

        # ---------- Attack on structure ----------
        evasion_attack_config = ConfigPattern(
            _class_name="FGSM",
            _import_path=EVASION_ATTACK_PARAMETERS_PATH,
            _config_class="EvasionAttackConfig",
            _config_kwargs={
                "is_feature_attack": False,
                "element_idx": 0,
                "epsilon": 0.5,
            }
        )

        gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

        # Attack
        _ = gnn_model_manager.evaluate_model(gen_dataset=self.gen_dataset_mg_small,
                                             metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
        # ---------- ------------------- ----------

        # ---------- Attack on feature ----------
        evasion_attack_config = ConfigPattern(
            _class_name="FGSM",
            _import_path=EVASION_ATTACK_PARAMETERS_PATH,
            _config_class="EvasionAttackConfig",
            _config_kwargs={
                "is_feature_attack": True,
                "element_idx": 0,
                "epsilon": 0.5,
            }
        )

        gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

        # Attack
        _ = gnn_model_manager.evaluate_model(gen_dataset=self.gen_dataset_mg_small,
                                             metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
        # ---------- ----------------- ----------

    def test_pgd_SG(self):
        gcn_gcn = model_configs_zoo(dataset=self.gen_dataset_sg_example, model_name='gcn_gcn')

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
            gnn=gcn_gcn,
            dataset_path=self.gen_dataset_sg_example.prepared_dir,
            manager_config=manager_config,
            modification=ModelModificationConfig(model_ver_ind=0, epochs=0)
        )

        gnn_model_manager.gnn.to(self.my_device)
        gnn_model_manager.train_model(gen_dataset=self.gen_dataset_sg_example, steps=200, save_model_flag=False)

        # ---------- Attack on structure ----------
        evasion_attack_config = ConfigPattern(
            _class_name="PGD",
            _import_path=EVASION_ATTACK_PARAMETERS_PATH,
            _config_class="EvasionAttackConfig",
            _config_kwargs={
                "is_feature_attack": False,
                "element_idx": 0,
                "num_iterations": 10,
                "epsilon": 0.7,
            }
        )

        gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

        # Attack
        _ = gnn_model_manager.evaluate_model(gen_dataset=self.gen_dataset_sg_example,
                                             metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
        # ---------- ------------------- ----------

        # ---------- Attack on feature ----------
        # Attack config
        evasion_attack_config = ConfigPattern(
            _class_name="PGD",
            _import_path=EVASION_ATTACK_PARAMETERS_PATH,
            _config_class="EvasionAttackConfig",
            _config_kwargs={
                "is_feature_attack": True,
                "element_idx": 0,
                "num_iterations": 10,
                "epsilon": 0.7,
            }
        )

        gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

        # Attack
        _ = gnn_model_manager.evaluate_model(gen_dataset=self.gen_dataset_sg_example,
                                             metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
        # ---------- ----------------- ----------

    def test_pgd_MG(self):
        gcn_gcn = model_configs_zoo(dataset=self.gen_dataset_mg_small, model_name='gin_gin_gin_lin_lin_con')

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
            gnn=gcn_gcn,
            dataset_path=self.results_dataset_path_mg_small,
            manager_config=manager_config,
            modification=ModelModificationConfig(model_ver_ind=0, epochs=0)
        )

        gnn_model_manager.gnn.to(self.my_device)
        gnn_model_manager.train_model(gen_dataset=self.gen_dataset_mg_small, steps=200, save_model_flag=False)

        # ---------- Attack on structure ----------
        evasion_attack_config = ConfigPattern(
            _class_name="PGD",
            _import_path=EVASION_ATTACK_PARAMETERS_PATH,
            _config_class="EvasionAttackConfig",
            _config_kwargs={
                "is_feature_attack": False,
                "element_idx": 0,
                "num_iterations": 10,
                "epsilon": 0.7,
            }
        )

        gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

        # Attack
        _ = gnn_model_manager.evaluate_model(gen_dataset=self.gen_dataset_mg_small,
                                             metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
        # ---------- ------------------- ----------

        # ---------- Attack on feature ----------
        evasion_attack_config = ConfigPattern(
            _class_name="PGD",
            _import_path=EVASION_ATTACK_PARAMETERS_PATH,
            _config_class="EvasionAttackConfig",
            _config_kwargs={
                "is_feature_attack": True,
                "element_idx": 0,
                "num_iterations": 10,
                "epsilon": 0.7,
            }
        )

        gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

        # Attack
        _ = gnn_model_manager.evaluate_model(gen_dataset=self.gen_dataset_mg_small,
                                             metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
        # ---------- ----------------- ----------

    def test_rewatt_SG(self):
        gat_gat_sg_example = model_configs_zoo(dataset=self.gen_dataset_sg_example, model_name='gat_gat')

        gnn_model_manager_sg_example = FrameworkGNNModelManager(
            gnn=gat_gat_sg_example,
            dataset_path=self.gen_dataset_sg_example.prepared_dir,
            modification=self.default_config,
            manager_config=self.manager_config,
        )

        gnn_model_manager_sg_example.gnn.to(self.my_device)

        gnn_model_manager_sg_example.train_model(gen_dataset=self.gen_dataset_sg_example, steps=100, metrics=[Metric("Accuracy", mask='test')])

        # Attack config
        evasion_attack_config = ConfigPattern(
            _class_name="ReWatt",
            _import_path=EVASION_ATTACK_PARAMETERS_PATH,
            _config_class="EvasionAttackConfig",
            _config_kwargs={
                "element_idx": 0,
                "eps": 0.5,
                "epochs": 10,
            }
        )

        gnn_model_manager_sg_example.set_evasion_attacker(evasion_attack_config=evasion_attack_config)
        metric_loc = gnn_model_manager_sg_example.evaluate_model(gen_dataset=self.gen_dataset_sg_example,
                                                                 metrics=[Metric("F1", mask='test', average='macro'),
                                                                          Metric("Accuracy", mask='test')])
        print(metric_loc)

    def test_rewatt_MG(self):
        gcn_gcn_mg_small = model_configs_zoo(dataset=self.gen_dataset_mg_small, model_name='gin_gin_gin_lin_lin')

        gnn_model_manager_mg_small = FrameworkGNNModelManager(
            gnn=gcn_gcn_mg_small,
            dataset_path=self.results_dataset_path_mg_small,
            modification=self.default_config,
            manager_config=self.manager_config,
        )

        gnn_model_manager_mg_small.gnn.to(self.my_device)

        gnn_model_manager_mg_small.train_model(gen_dataset=self.gen_dataset_mg_small, steps=100,
                                                 metrics=[Metric("Accuracy", mask='test')])

        # Attack config
        evasion_attack_config = ConfigPattern(
            _class_name="ReWatt",
            _import_path=EVASION_ATTACK_PARAMETERS_PATH,
            _config_class="EvasionAttackConfig",
            _config_kwargs={
                "element_idx": 0,
                "eps": 0.5,
                "epochs": 10,
            }
        )

        gnn_model_manager_mg_small.set_evasion_attacker(evasion_attack_config=evasion_attack_config)
        metric_loc = gnn_model_manager_mg_small.evaluate_model(gen_dataset=self.gen_dataset_mg_small,
                                                                 metrics=[Metric("F1", mask='test', average='macro'),
                                                                          Metric("Accuracy", mask='test')])
        print(metric_loc)

    def test_nettack(self):
        gcn_gcn_sg_cora = model_configs_zoo(dataset=self.gen_dataset_sg_cora, model_name='gcn_gcn')

        gnn_model_manager_sg_cora = FrameworkGNNModelManager(
            gnn=gcn_gcn_sg_cora,
            dataset_path=self.gen_dataset_sg_cora.prepared_dir,
            modification=self.default_config,
            manager_config=self.manager_config,
        )

        gnn_model_manager_sg_cora.gnn.to(self.my_device)

        gnn_model_manager_sg_cora.train_model(gen_dataset=self.gen_dataset_sg_cora, steps=100, metrics=[Metric("Accuracy", mask='test')])

        # Attack config
        evasion_attack_config = ConfigPattern(
            _class_name="Nettack",
            _import_path=EVASION_ATTACK_PARAMETERS_PATH,
            _config_class="EvasionAttackConfig",
            _config_kwargs={
                "node_idx": 0,
            }
        )

        gnn_model_manager_sg_cora.set_evasion_attacker(evasion_attack_config=evasion_attack_config)
        metric_loc = gnn_model_manager_sg_cora.evaluate_model(gen_dataset=self.gen_dataset_sg_cora,
                                                                 metrics=[Metric("F1", mask='test', average='macro'),
                                                                          Metric("Accuracy", mask='test')])
        print(metric_loc)

    def test_mi_shadow_cora(self):
        mi_attack_config = ConfigPattern(
            _class_name="ShadowModelMIAttacker",
            _import_path=MI_ATTACK_PARAMETERS_PATH,
            _config_class="MIAttackConfig",
            _config_kwargs={
                "shadow_epochs": 200,
                "shadow_data_ratio": 0.1
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

        attack_cnt = 500
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


if __name__ == '__main__':
    unittest.main()
