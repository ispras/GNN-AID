import unittest
import torch

from base.datasets_processing import DatasetManager
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from aux.configs import ModelModificationConfig, DatasetConfig, DatasetVarConfig, ConfigPattern
from models_builder.models_zoo import model_configs_zoo
from aux.utils import POISON_ATTACK_PARAMETERS_PATH, EVASION_ATTACK_PARAMETERS_PATH, \
    OPTIMIZERS_PARAMETERS_PATH, import_all_from_package

import attacks
import_all_from_package(attacks)  # to import all subclasses properly


class AttacksTest(unittest.TestCase):
    def setUp(self):
        print('setup')
        self.my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Init datasets
        # Multi-Graphs - Example
        self.dataset_mg_small, _, results_dataset_path_sg_small = DatasetManager.get_by_full_name(
            full_name=("multiple-graphs", "custom", "small",),
            features={'attr': {'a': 'as_is'}},
            labeling='binary',
            dataset_ver_ind=0
        )

        self.gen_dataset_mg_small = DatasetManager.get_by_config(
            DatasetConfig(
                domain="multiple-graphs",
                group="custom",
                graph="small"),
            DatasetVarConfig(features={'attr': {'a': 'as_is'}},
                             labeling='binary',
                             dataset_ver_ind=0)
        )

        self.gen_dataset_mg_small.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
        self.results_dataset_path_mg_small = self.gen_dataset_mg_small.results_dir


        # Single-Graph - Example
        self.dataset_sg_example, _, results_dataset_path_sg_example = DatasetManager.get_by_full_name(
            full_name=("single-graph", "custom", "example",),
            features={'attr': {'a': 'as_is'}},
            labeling='binary',
            dataset_ver_ind=0
        )

        self.gen_dataset_sg_example = DatasetManager.get_by_config(
            DatasetConfig(
                domain="single-graph",
                group="custom",
                graph="example"),
            DatasetVarConfig(features={'attr': {'a': 'as_is'}},
                             labeling='binary',
                             dataset_ver_ind=0)
        )
        self.gen_dataset_sg_example.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
        self.results_dataset_path_sg_example = self.gen_dataset_sg_example.results_dir

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

        #Single-Graph - Cora


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
            dataset_path=self.results_dataset_path_sg_example,
            modification=self.default_config,
            manager_config=self.manager_config,
        )

        gnn_model_manager_sg_example.set_poison_attacker(poison_attack_config=poison_attack_config)

        gnn_model_manager_sg_example.train_model(gen_dataset=self.gen_dataset_sg_example, steps=100, metrics=[Metric("Accuracy", mask='test')])
        metric_loc = gnn_model_manager_sg_example.evaluate_model(gen_dataset=self.gen_dataset_sg_example, metrics=[Metric("F1", mask='test', average='macro')])
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
            dataset_path=self.results_dataset_path_sg_example,
            modification=self.default_config,
            manager_config=self.manager_config,
        )

        gnn_model_manager_sg_example.set_poison_attacker(poison_attack_config=poison_attack_config)

        gnn_model_manager_sg_example.train_model(gen_dataset=self.gen_dataset_sg_example, steps=100, metrics=[Metric("Accuracy", mask='test')])
        metric_loc = gnn_model_manager_sg_example.evaluate_model(gen_dataset=self.gen_dataset_sg_example,
                                                                 metrics=[Metric("F1", mask='test', average='macro'),
                                                                          Metric("Accuracy", mask='test')])
        print(metric_loc)

    def test_qattack_Cora(self):
        evasion_attack_config = ConfigPattern(
            _class_name="QAttack",
            _import_path=EVASION_ATTACK_PARAMETERS_PATH,
            _config_class="EvasionAttackConfig",
            _config_kwargs={
            }
        )

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
            dataset_path=self.results_dataset_path_sg_example,
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
            dataset_path=self.results_dataset_path_sg_example,
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


if __name__ == '__main__':
    unittest.main()