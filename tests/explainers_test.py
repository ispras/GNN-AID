import collections.abc
collections.Callable = collections.abc.Callable
import unittest
import warnings

from gnn_aid.aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH, \
    EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH, FUNCTIONS_PARAMETERS_PATH
from gnn_aid.datasets.datasets_manager import DatasetManager
from gnn_aid.datasets.ptg_datasets import LibPTGDataset
from gnn_aid.explainers.explainers_manager import FrameworkExplainersManager
from gnn_aid.data_structures.configs import FeatureConfig, Task
from gnn_aid.models_builder.models_utils import Metric
from gnn_aid.models_builder.model_managers import FrameworkGNNModelManager, ProtGNNModelManager, GSATModelManager
from gnn_aid.data_structures.configs import DatasetConfig, DatasetVarConfig, ConfigPattern, ModelModificationConfig
from gnn_aid.models_builder.models_zoo import model_configs_zoo
from tests.utils import cleanup_patches, monkey_patch_dirs


# TODO PGM,PGE tests + test re-work -> more use-cases

class ExplainersTest(unittest.TestCase):
    def setUp(self) -> None:
        # Init datasets
        # Single-Graph - Example
        gen_dataset_sg_example = DatasetManager.get_by_config(
            DatasetConfig(("example", "example")),
            DatasetVarConfig(task=Task.NODE_CLASSIFICATION,
                             features=FeatureConfig(node_attr=['a']),
                             labeling='binary',
                             dataset_ver_ind=0)
        )
        gen_dataset_sg_example.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
        self.dataset_sg_example = gen_dataset_sg_example
        results_dataset_path_sg_example = gen_dataset_sg_example.prepared_dir

        #Single-graph - Cora
        self.gen_dataset_sg_cora = DatasetManager.get_by_config(
            DatasetConfig((LibPTGDataset.data_folder, "Homogeneous", "Planetoid", "Cora")),
            LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.NODE_CLASSIFICATION})
        )

        self.gen_dataset_sg_cora.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
        self.results_dataset_path_sg_cora = self.gen_dataset_sg_cora.prepared_dir

        # Multi-graphs - Small
        self.dataset_mg_small = DatasetManager.get_by_config(
            DatasetConfig(('example', 'example8')),
            DatasetVarConfig(task=Task.GRAPH_CLASSIFICATION,
                             features=FeatureConfig(node_attr=['a']),
                             labeling='binary',
                             dataset_ver_ind=0)
        )
        self.dataset_mg_small.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
        results_dataset_path_mg_small = self.dataset_mg_small.prepared_dir

        # Multi-graphs - MUTAG
        self.dataset_mg_mutag = DatasetManager.get_by_config(
            DatasetConfig((LibPTGDataset.data_folder, "Homogeneous", "TUDataset", "MUTAG")),
            LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.GRAPH_CLASSIFICATION})
        )

        gen_dataset_mg_mutag = self.dataset_mg_mutag
        gen_dataset_mg_mutag.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
        dataset_mg_mutag = gen_dataset_mg_mutag
        results_dataset_path_mg_mutag = gen_dataset_mg_mutag.prepared_dir

        # Init models
        gcn2_sg_example = model_configs_zoo(dataset=gen_dataset_sg_example, model_name='gcn_gcn')

        gnn_model_manager_sg_example_manager_config = ConfigPattern(
            _config_class="ModelManagerConfig",
            _config_kwargs={
                "batch": 10000,
                "mask_features": []
            }
        )
        self.gnn_model_manager_sg_example = FrameworkGNNModelManager(
            gnn=gcn2_sg_example,
            dataset_path=results_dataset_path_sg_example,
            manager_config=gnn_model_manager_sg_example_manager_config
        )

        self.gnn_model_manager_sg_example.train_model(gen_dataset=gen_dataset_sg_example, steps=50,
                                                      save_model_flag=False,
                                                      metrics=[Metric("F1", mask='test')])

        # TODO Kirill, tmp comment work and tests with Prot
        gin3_lin2_prot_mg_small = model_configs_zoo(
            dataset=self.dataset_mg_small, model_name='gin_gin_gin_lin_lin_prot')
        gin3_lin2_prot_mg_mutag = model_configs_zoo(
            dataset=dataset_mg_mutag, model_name='gin_gin_gin_lin_lin_prot'
        )
        gin3_lin1_mg_mutag = model_configs_zoo(
            dataset=dataset_mg_mutag, model_name='gin_gin_gin_lin')

        gnn_model_manager_mg_mutag_manager_config = ConfigPattern(
            _config_class="ModelManagerConfig",
            _config_kwargs={
                "batch": 24,
                "mask_features": []
            }
        )
        self.gnn_model_manager_mg_mutag = FrameworkGNNModelManager(
            gnn=gin3_lin1_mg_mutag,
            dataset_path=results_dataset_path_mg_mutag,
            manager_config=gnn_model_manager_mg_mutag_manager_config
        )

        self.gnn_model_manager_mg_mutag.train_model(
            gen_dataset=dataset_mg_mutag, steps=50, save_model_flag=False,
            metrics=[Metric("F1", mask='test')])

        gin3_lin2_mg_small_manager_config = ConfigPattern(
            _config_class="ModelManagerConfig",
            _config_kwargs={
                "batch": 10000,
                "mask_features": []
            }
        )

        self.prot_gnn_mm_mg_small = ProtGNNModelManager(
            gnn=gin3_lin2_prot_mg_small, dataset_path=results_dataset_path_mg_small,
            # manager_config=gin3_lin2_mg_small_manager_config,
        )
        self.prot_gnn_mm_mutag = ProtGNNModelManager(gnn=gin3_lin2_prot_mg_mutag, dataset_path=results_dataset_path_mg_mutag)
        # TODO Misha use as training params: clst=clst, sep=sep, save_thrsh=save_thrsh, lr=lr

        best_acc = self.prot_gnn_mm_mg_small.train_model(
            gen_dataset=self.dataset_mg_small, steps=100, metrics=[])

        # uncomment for ProtGNN big test
        # self.prot_gnn_mm_mutag.train_model(
        #     gen_dataset=gen_dataset_mg_mutag, steps=40, metrics=[])

        gin3_lin2_mg_small = model_configs_zoo(
            dataset=self.dataset_mg_small, model_name='gin_gin_gin_lin_lin')
        self.gnn_model_manager_mg_small = FrameworkGNNModelManager(
            gnn=gin3_lin2_mg_small,
            dataset_path=results_dataset_path_mg_small,
            manager_config=gin3_lin2_mg_small_manager_config
        )
        self.gnn_model_manager_mg_small.train_model(
            gen_dataset=self.dataset_mg_small, steps=50, save_model_flag=False,
            metrics=[Metric("F1", mask='test')])

        self.dummy_gcn_2_gsat = model_configs_zoo(dataset=self.gen_dataset_sg_cora, model_name="dummy_gcn_gcn_gsat")
        # dummy_gcn_2_gsat = model_configs_zoo(dataset=self.gen_dataset_sg_cora, model_name="gcn_gcn")


        self.gsat_config = ConfigPattern(
            _config_class="ModelManagerConfig",
            _config_kwargs={
                "mask_features": [],
                "optimizer": {
                    # "_config_class": "Config",
                    "_class_name": "Adam",
                    # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                    # "_class_import_info": ["torch.optim"],
                    "_config_kwargs": {
                        "lr": 0.01
                    },
                }
            }
        )
        # self.gsat_config = ConfigPattern(
        #     _config_class="ModelManagerConfig",
        #     _config_kwargs={
        #         "mask_features": [],
        #     }
        # )

        self.default_config = ModelModificationConfig(
            model_ver_ind=0,
        )

        self.gsat_gnn_mm_sg_cora = GSATModelManager(
            gnn=self.dummy_gcn_2_gsat,
            manager_config=self.gsat_config,
            modification=self.default_config,
            dataset_path=self.results_dataset_path_sg_cora
        )

        # gsat_gnn_mm_sg_cora = FrameworkGNNModelManager(
        #     gnn=dummy_gcn_2_gsat,
        #     manager_config=self.manager_config,
        #     modification=self.default_config,
        #     dataset_path=self.results_dataset_path_sg_cora
        # )

        self.gsat_gnn_mm_sg_cora.train_model(gen_dataset=self.gen_dataset_sg_cora, steps=300, metrics=[])
        metric_loc = self.gsat_gnn_mm_sg_cora.evaluate_model(
            gen_dataset=self.gen_dataset_sg_cora, metrics=[Metric("F1", mask='test', average='macro')])
        print(metric_loc)
        sg_cora_model_path = self.gsat_gnn_mm_sg_cora.model_path_info() / 'model'
        self.gsat_gnn_mm_sg_cora.load_model_executor(path=sg_cora_model_path)

        # Cora for link pred
        dc = DatasetConfig((LibPTGDataset.data_folder, 'Homogeneous', 'Planetoid', 'Cora'))
        dvc = LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.EDGE_PREDICTION})

        self.gen_dataset_sg_cora_link = DatasetManager.get_by_config(dc, dvc)
        self.gen_dataset_sg_cora_link.train_test_split(percent_train_class=0.85, percent_test_class=0.1)
        self.results_dataset_path_sg_cora_link = self.gen_dataset_sg_cora_link.prepared_dir

        self.sage_cossim = model_configs_zoo(dataset=self.gen_dataset_sg_cora_link, model_name="sage_cossim")

        manager_config = ConfigPattern(
            _config_class="ModelManagerConfig",
            _config_kwargs={
                "batch": 64,
                "mask_features": [],
                "optimizer": {
                    "_class_name": "Adam",
                    "_config_kwargs": {},
                },
                "loss_function": {
                    "_config_class": "Config",
                    "_class_name": "CrossEntropyLoss",
                    "_import_path": FUNCTIONS_PARAMETERS_PATH,
                    "_class_import_info": ["torch.nn"],
                    "_config_kwargs": {},
                },
            }
        )
        self.sage_cossim_mm = FrameworkGNNModelManager(
            gnn=self.sage_cossim,
            dataset_path=self.gen_dataset_sg_cora_link.prepared_dir,
            manager_config=manager_config,
            modification=ModelModificationConfig(model_ver_ind=0, epochs=0)
        )
        self.sage_cossim_mm.train_model(
            gen_dataset=self.gen_dataset_sg_cora_link, steps=10,
            save_model_flag=False,
            metrics=[Metric("F1", mask='train', average=None)]
        )

        monkey_patch_dirs()

    def tearDown(self):
        # Clean up patches and tmp dirs
        cleanup_patches()

    def test_PGE_SG(self):
        # FIXME not working with another tests
        warnings.warn("Start PGExplainer(dig)")
        explainer_init_config = ConfigPattern(
            _class_name="PGExplainer(dig)",
            _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
            _config_class="ExplainerInitConfig",
            _config_kwargs={
            }
        )
        explainer_run_config = ConfigPattern(
            _config_class="ExplainerRunConfig",
            _config_kwargs={
                "mode": "local",
                "kwargs": {
                    "_class_name": "PGExplainer(dig)",
                    "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                    "_config_class": "Config",
                    "_config_kwargs": {
                        'element_idx': 0,
                    },
                }
            }
        )
        explainer_PGE = FrameworkExplainersManager(
            init_config=explainer_init_config,
            dataset=self.dataset_sg_example, gnn_manager=self.gnn_model_manager_sg_example,
            explainer_name='PGExplainer(dig)',
        )
        explainer_PGE.conduct_experiment(explainer_run_config)

    def test_PGE_MG(self):
        warnings.warn("Start PGExplainer(dig)")
        explainer_init_config = ConfigPattern(
            _class_name="PGExplainer(dig)",
            _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
            _config_class="ExplainerInitConfig",
            _config_kwargs={
            }
        )
        explainer_run_config = ConfigPattern(
            _config_class="ExplainerRunConfig",
            _config_kwargs={
                "mode": "local",
                "kwargs": {
                    "_class_name": "PGExplainer(dig)",
                    "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                    "_config_class": "Config",
                    "_config_kwargs": {

                    },
                }
            }
        )
        explainer_PGE = FrameworkExplainersManager(
            init_config=explainer_init_config,
            dataset=self.dataset_mg_mutag, gnn_manager=self.gnn_model_manager_mg_mutag,
            explainer_name='PGExplainer(dig)',
        )
        explainer_PGE.conduct_experiment(explainer_run_config)

    def test_PGM_SG(self):
        warnings.warn("Start PGMExplainer")
        explainer_init_config = ConfigPattern(
            _class_name="PGMExplainer",
            _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
            _config_class="ExplainerInitConfig",
            _config_kwargs={
            }
        )
        explainer_run_config = ConfigPattern(
            _config_class="ExplainerRunConfig",
            _config_kwargs={
                "mode": "local",
                "kwargs": {
                    "_class_name": "PGMExplainer",
                    "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                    "_config_class": "Config",
                    "_config_kwargs": {

                    },
                }
            }
        )
        explainer_PGM = FrameworkExplainersManager(
            init_config=explainer_init_config,
            dataset=self.dataset_sg_example, gnn_manager=self.gnn_model_manager_sg_example,
            explainer_name='PGMExplainer',
        )
        explainer_PGM.conduct_experiment(explainer_run_config)

    def test_PGM_MG(self):
        warnings.warn("Start PGMExplainer")
        explainer_init_config = ConfigPattern(
            _class_name="PGMExplainer",
            _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
            _config_class="ExplainerInitConfig",
            _config_kwargs={
            }
        )
        explainer_run_config = ConfigPattern(
            _config_class="ExplainerRunConfig",
            _config_kwargs={
                "mode": "local",
                "kwargs": {
                    "_class_name": "PGMExplainer",
                    "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                    "_config_class": "Config",
                    "_config_kwargs": {

                    },
                }
            }
        )
        explainer_PGM = FrameworkExplainersManager(
            init_config=explainer_init_config,
            dataset=self.dataset_mg_mutag, gnn_manager=self.gnn_model_manager_mg_mutag,
            explainer_name='PGMExplainer',
        )
        explainer_PGM.conduct_experiment(explainer_run_config)

    def test_Zorro(self):
        warnings.warn("Start Zorro")
        explainer_init_config = ConfigPattern(
            _class_name="Zorro",
            _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
            _config_class="ExplainerInitConfig",
            _config_kwargs={
            }
        )
        explainer_run_config = ConfigPattern(
            _config_class="ExplainerRunConfig",
            _config_kwargs={
                "mode": "local",
                "kwargs": {
                    "_class_name": "Zorro",
                    "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                    "_config_class": "Config",
                    "_config_kwargs": {

                    },
                }
            }
        )
        explainer_Zorro = FrameworkExplainersManager(
            init_config=explainer_init_config,
            dataset=self.dataset_sg_example, gnn_manager=self.gnn_model_manager_sg_example,
            explainer_name='Zorro',
        )
        explainer_Zorro.conduct_experiment(explainer_run_config)

    def test_ProtGNN(self):
        warnings.warn("Start ProtGNN")
        explainer_init_config = ConfigPattern(
            _class_name="ProtGNN",
            _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
            _config_class="ExplainerInitConfig",
            _config_kwargs={
            }
        )
        explainer_run_config = ConfigPattern(
            _config_class="ExplainerRunConfig",
            _config_kwargs={
                "mode": "global",
                "kwargs": {
                    "_class_name": "ProtGNN",
                    "_import_path": EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH,
                    "_config_class": "Config",
                    "_config_kwargs": {

                    },
                }
            }
        )
        explainer_Prot = FrameworkExplainersManager(
            init_config=explainer_init_config,
            dataset=self.dataset_mg_small, gnn_manager=self.prot_gnn_mm_mg_small,
            explainer_name='ProtGNN',
        )

        explainer_Prot.conduct_experiment(explainer_run_config)

    def test_ProtGNN_big(self):
        # uncomment model train for this test - big one

        warnings.warn("Start ProtGNN")
        explainer_init_config = ConfigPattern(
            _class_name="ProtGNN",
            _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
            _config_class="ExplainerInitConfig",
            _config_kwargs={
            }
        )
        explainer_run_config = ConfigPattern(
            _config_class="ExplainerRunConfig",
            _config_kwargs={
                "mode": "global",
                "kwargs": {
                    "_class_name": "ProtGNN",
                    "_import_path": EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH,
                    "_config_class": "Config",
                    "_config_kwargs": {

                    },
                }
            }
        )
        explainer_Prot = FrameworkExplainersManager(
            init_config=explainer_init_config,
            dataset=self.dataset_mg_mutag, gnn_manager=self.prot_gnn_mm_mutag,
            explainer_name='ProtGNN',
        )

        explainer_Prot.conduct_experiment(explainer_run_config)


    def test_GNNExpl_PYG_SG(self):
        warnings.warn("Start GNNExplainer_PYG")
        explainer_init_config = ConfigPattern(
            _class_name="GNNExplainer(torch-geom)",
            _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
            _config_class="ExplainerInitConfig",
            _config_kwargs={
            }
        )
        explainer_run_config = ConfigPattern(
            _config_class="ExplainerRunConfig",
            _config_kwargs={
                "mode": "local",
                "kwargs": {
                    "_class_name": "GNNExplainer(torch-geom)",
                    "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                    "_config_class": "Config",
                    "_config_kwargs": {

                    },
                }
            }
        )
        explainer_GNNExpl = FrameworkExplainersManager(
            init_config=explainer_init_config,
            dataset=self.dataset_sg_example, gnn_manager=self.gnn_model_manager_sg_example,
            explainer_name='GNNExplainer(torch-geom)',
        )
        explainer_GNNExpl.conduct_experiment(explainer_run_config)

    def test_GNNExpl_PYG_LINK(self):
        warnings.warn("Start GNNExplainer_PYG")
        explainer_init_config = ConfigPattern(
            _class_name="GNNExplainer(torch-geom)",
            _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
            _config_class="ExplainerInitConfig",
            _config_kwargs={
            }
        )
        explainer_run_config = ConfigPattern(
            _config_class="ExplainerRunConfig",
            _config_kwargs={
                "mode": "local",
                "kwargs": {
                    "_class_name": "GNNExplainer(torch-geom)",
                    "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                    "_config_class": "Config",
                    "_config_kwargs": {
                        "element_idx": (1, 2)
                    },
                }
            }
        )
        explainer_GNNExpl = FrameworkExplainersManager(
            init_config=explainer_init_config,
            dataset=self.gen_dataset_sg_cora_link, gnn_manager=self.sage_cossim_mm,
            explainer_name='GNNExplainer(torch-geom)',
        )
        explainer_GNNExpl.conduct_experiment(explainer_run_config)

    def test_GNNExpl_PYG_MG(self):
        warnings.warn("Start GNNExplainer_PYG")
        explainer_init_config = ConfigPattern(
            _class_name="GNNExplainer(torch-geom)",
            _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
            _config_class="ExplainerInitConfig",
            _config_kwargs={
            }
        )
        explainer_run_config = ConfigPattern(
            _config_class="ExplainerRunConfig",
            _config_kwargs={
                "mode": "local",
                "kwargs": {
                    "_class_name": "GNNExplainer(torch-geom)",
                    "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                    "_config_class": "Config",
                    "_config_kwargs": {

                    },
                }
            }
        )
        explainer_GNNExpl = FrameworkExplainersManager(
            init_config=explainer_init_config,
            dataset=self.dataset_mg_small, gnn_manager=self.gnn_model_manager_mg_small,
            explainer_name='GNNExplainer(torch-geom)',
        )
        explainer_GNNExpl.conduct_experiment(explainer_run_config)

    # def test_NeuralAnalysis_MG(self):
    #     warnings.warn("Start Neural Analysis")
    #     explainer_init_config = ConfigPattern(
    #         _class_name="NeuralAnalysis",
    #         _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
    #         _config_class="ExplainerInitConfig",
    #         _config_kwargs={
    #         }
    #     )
    #     explainer_run_config = ConfigPattern(
    #         _config_class="ExplainerRunConfig",
    #         _config_kwargs={
    #             "mode": "global",
    #             "kwargs": {
    #                 "_class_name": "NeuralAnalysis",
    #                 "_import_path": EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH,
    #                 "_config_class": "Config",
    #                 "_config_kwargs": {
    #
    #                 },
    #             }
    #         }
    #     )
    #     explainer = FrameworkExplainersManager(
    #         init_config=explainer_init_config,
    #         dataset=self.dataset_mg_mutag, gnn_manager=self.gnn_model_manager_mg_mutag,
    #         explainer_name='NeuralAnalysis',
    #     )
    #     explainer.conduct_experiment(explainer_run_config)

    def test_GSAT_SG(self):
        warnings.warn("Start GSATExplainer")
        explainer_init_config = ConfigPattern(
            _class_name="GSAT",
            _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
            _config_class="ExplainerInitConfig",
            _config_kwargs={
            }
        )
        explainer_run_config = ConfigPattern(
            _config_class="ExplainerRunConfig",
            _config_kwargs={
                "mode": "local",
                "kwargs": {
                    "_class_name": "GSAT",
                    "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                    "_config_class": "Config",
                    "_config_kwargs": {
                        'element_idx': 0,
                    },
                }
            }
        )
        explainer_GSAT = FrameworkExplainersManager(
            init_config=explainer_init_config,
            dataset=self.gen_dataset_sg_cora, gnn_manager=self.gsat_gnn_mm_sg_cora,
            explainer_name='GSAT',
        )
        explanation = explainer_GSAT.conduct_experiment(explainer_run_config)


if __name__ == '__main__':
    unittest.main()
