import collections
import collections.abc
collections.Callable = collections.abc.Callable

import unittest
import warnings
import shutil
import signal
from time import time

# Monkey patch EXPLANATIONS_DIR - before other imports
from aux import utils
tmp_dir = utils.EXPLANATIONS_DIR / (utils.EXPLANATIONS_DIR.name + str(time()))
utils.EXPLANATIONS_DIR = tmp_dir

from aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH, \
    EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH
from base.datasets_processing import DatasetManager
from explainers.explainers_manager import FrameworkExplainersManager
from models_builder.gnn_models import FrameworkGNNModelManager, ProtGNNModelManager, Metric
from aux.configs import ModelManagerConfig, DatasetConfig, DatasetVarConfig, ExplainerRunConfig, \
    ExplainerInitConfig, ConfigPattern
from models_builder.models_zoo import model_configs_zoo

# from src.aux import utils
# from src.aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH, \
#     EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH
# from src.base.datasets_processing import DatasetManager
# from src.explainers.explainers_manager import FrameworkExplainersManager
# from src.models_builder.gnn_models import FrameworkGNNModelManager, ProtGNNModelManager, Metric
# from src.aux.configs import ModelManagerConfig, DatasetConfig, DatasetVarConfig, ExplainerRunConfig, \
#     ExplainerInitConfig, ConfigPattern
# from src.models_builder.models_zoo import model_configs_zoo


def my_ctrlc_handler(signal, frame):
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, my_ctrlc_handler)


# TODO PGM,PGE tests + test re-work -> more use-cases

class ExplainersTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    def setUp(self) -> None:
        # Init datasets
        # Single-Graph - Example
        self.dataset_sg_example, _, results_dataset_path_sg_example = DatasetManager.get_by_full_name(
            full_name=("single-graph", "custom", "example",),
            features={'attr': {'a': 'as_is'}},
            labeling='binary',
            dataset_ver_ind=0
        )

        gen_dataset_sg_example = DatasetManager.get_by_config(
            DatasetConfig(
                domain="single-graph",
                group="custom",
                graph="example"),
            DatasetVarConfig(features={'attr': {'a': 'as_is'}},
                             labeling='binary',
                             dataset_ver_ind=0)
        )
        gen_dataset_sg_example.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
        self.dataset_sg_example = gen_dataset_sg_example
        results_dataset_path_sg_example = gen_dataset_sg_example.results_dir

        # Multi-graphs - Small
        self.dataset_mg_small, _, results_dataset_path_mg_small = DatasetManager.get_by_full_name(
            full_name=("multiple-graphs", "custom", "small",),
            features={'attr': {'a': 'as_is'}},
            labeling='binary',
            dataset_ver_ind=0
        )

        gen_dataset_mg_small = DatasetManager.get_by_config(
            DatasetConfig(
                domain="multiple-graphs",
                group="custom",
                graph="small"),
            DatasetVarConfig(features={'attr': {'a': 'as_is'}},
                             labeling='binary',
                             dataset_ver_ind=0)
        )
        gen_dataset_mg_small.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
        dataset_mg_small = gen_dataset_mg_small
        results_dataset_path_mg_small = gen_dataset_mg_small.results_dir

        # Multi-graphs - MUTAG
        self.dataset_mg_mutag, _, results_dataset_path_mg_mutag = DatasetManager.get_by_full_name(
            full_name=("multiple-graphs", "TUDataset", "MUTAG",),
            dataset_ver_ind=0
        )

        gen_dataset_mg_mutag = self.dataset_mg_mutag
        gen_dataset_mg_mutag.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
        dataset_mg_mutag = gen_dataset_mg_mutag
        results_dataset_path_mg_mutag = gen_dataset_mg_mutag.results_dir

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

        gin3_lin2_prot_mg_small = model_configs_zoo(
            dataset=dataset_mg_small, model_name='gin_gin_gin_lin_lin_prot')
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
        # TODO Misha use as training params: clst=clst, sep=sep, save_thrsh=save_thrsh, lr=lr

        best_acc = self.prot_gnn_mm_mg_small.train_model(
            gen_dataset=gen_dataset_mg_small, steps=100, metrics=[])

        gin3_lin2_mg_small = model_configs_zoo(
            dataset=gen_dataset_mg_small, model_name='gin_gin_gin_lin_lin')
        self.gnn_model_manager_mg_small = FrameworkGNNModelManager(
            gnn=gin3_lin2_mg_small,
            dataset_path=results_dataset_path_mg_small,
            manager_config=gin3_lin2_mg_small_manager_config
        )
        self.gnn_model_manager_mg_small.train_model(
            gen_dataset=gen_dataset_mg_small, steps=50, save_model_flag=False,
            metrics=[Metric("F1", mask='test')])

    def test_SubgraphX_SG(self):
        warnings.warn("Start SubgraphX")
        explainer_init_config = ConfigPattern(
            _class_name="SubgraphX",
            _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
            _config_class="ExplainerInitConfig",
            _config_kwargs={
                # "class_name": "SubgraphX",
            }
        )
        explainer_run_config = ConfigPattern(
            _config_class="ExplainerRunConfig",
            _config_kwargs={
                "mode": "local",
                "kwargs": {
                    "_class_name": "SubgraphX",
                    "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                    "_config_class": "Config",
                    "_config_kwargs": {
                        "element_idx": 0, "max_nodes": 5
                    },
                }
            }
        )
        explainer_SubgraphX = FrameworkExplainersManager(
            init_config=explainer_init_config,
            dataset=self.dataset_sg_example, gnn_manager=self.gnn_model_manager_sg_example,
            explainer_name="SubgraphX",
        )
        explainer_SubgraphX.conduct_experiment(explainer_run_config)

    def test_SubgraphX_MG(self):
        warnings.warn("Start SubgraphX")
        explainer_init_config = ConfigPattern(
            _class_name="SubgraphX",
            _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
            _config_class="ExplainerInitConfig",
            _config_kwargs={
                # "class_name": "SubgraphX",
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
                        "element_idx": 0, "max_nodes": 5
                    },
                }
            }
        )
        explainer_SubgraphX = FrameworkExplainersManager(
            init_config=explainer_init_config,
            dataset=self.dataset_mg_mutag, gnn_manager=self.gnn_model_manager_mg_mutag,
            explainer_name="SubgraphX",
        )
        explainer_SubgraphX.conduct_experiment(explainer_run_config)

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

    def test_GraphMask(self):
        warnings.warn("Start GraphMask")
        explainer_init_config = ConfigPattern(
            _class_name="GraphMask",
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
                    "_class_name": "GraphMask",
                    "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                    "_config_class": "Config",
                    "_config_kwargs": {

                    },
                }
            }
        )
        explainer_GraphMask = FrameworkExplainersManager(
            init_config=explainer_init_config,
            dataset=self.dataset_sg_example, gnn_manager=self.gnn_model_manager_sg_example,
            explainer_name='GraphMask',
        )
        explainer_GraphMask.conduct_experiment(explainer_run_config)

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

    def test_GNNExpl_DIG_SG(self):
        warnings.warn("Start GNNExplainer_DIG")
        explainer_init_config = ConfigPattern(
            _class_name="GNNExplainer(dig)",
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
                    "_class_name": "GNNExplainer(dig)",
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
            explainer_name='GNNExplainer(dig)',
        )
        explainer_GNNExpl.conduct_experiment(explainer_run_config)

    def test_GNNExpl_DIG_MG(self):
        warnings.warn("Start GNNExplainer_DIG")
        explainer_init_config = ConfigPattern(
            _class_name="GNNExplainer(dig)",
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
                    "_class_name": "GNNExplainer(dig)",
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
            explainer_name='GNNExplainer(dig)',
        )
        explainer_GNNExpl.conduct_experiment(explainer_run_config)

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

    def test_NeuralAnalysis_MG(self):
        warnings.warn("Start Neural Analysis")
        explainer_init_config = ConfigPattern(
            _class_name="NeuralAnalysis",
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
                    "_class_name": "NeuralAnalysis",
                    "_import_path": EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH,
                    "_config_class": "Config",
                    "_config_kwargs": {

                    },
                }
            }
        )
        explainer = FrameworkExplainersManager(
            init_config=explainer_init_config,
            dataset=self.dataset_mg_mutag, gnn_manager=self.gnn_model_manager_mg_mutag,
            explainer_name='NeuralAnalysis',
        )
        explainer.conduct_experiment(explainer_run_config)


if __name__ == '__main__':
    unittest.main()
