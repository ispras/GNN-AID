import collections
import collections.abc
collections.Callable = collections.abc.Callable
import unittest
import shutil
import signal
from time import time

# Monkey patch MODELS_DIR - before other imports
from aux import utils
tmp_dir = utils.MODELS_DIR / (utils.MODELS_DIR.name + str(time()))
utils.MODELS_DIR = tmp_dir

from base.datasets_processing import DatasetManager
from models_builder.gnn_models import FrameworkGNNModelManager, ProtGNNModelManager, Metric
from aux.configs import ModelManagerConfig, ModelModificationConfig, DatasetConfig, DatasetVarConfig, ConfigPattern
from models_builder.models_zoo import model_configs_zoo


def my_ctrlc_handler(signal, frame):
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, my_ctrlc_handler)


class ModelsTest(unittest.TestCase):

    @classmethod
    def tearDownClass(cls) -> None:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    def setUp(self) -> None:
        # Monkey patch
        print('setup')

        # Init datasets
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

        # Multi-graphs - Small
        self.dataset_mg_small, _, results_dataset_path_mg_small = DatasetManager.get_by_full_name(
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
        self.gen_dataset_mg_small.train_test_split(percent_train_class=0.6, percent_test_class=0.2)
        self.results_dataset_path_mg_small = self.gen_dataset_mg_small.results_dir
        self.default_config = ModelModificationConfig(
            model_ver_ind=0,
        )

        self.manager_config = ConfigPattern(
            _config_class="ModelManagerConfig",
            _config_kwargs={
                "mask_features": [],
                # "optimizer": {
                #     # "_config_class": "Config",
                #     "_class_name": "Adam",
                #     # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                #     # "_class_import_info": ["torch.optim"],
                #     "_config_kwargs": {},
                # }
            }
        )

    def test_combo_model_on_single_graph(self):
        gat_gin_lin_sg_example = model_configs_zoo(dataset=self.gen_dataset_sg_example, model_name='gat_gin_lin')

        gnn_model_manager_sg_example = FrameworkGNNModelManager(
            gnn=gat_gin_lin_sg_example,
            dataset_path=self.results_dataset_path_sg_example,
            modification=self.default_config,
            manager_config=self.manager_config,
        )

        gnn_model_manager_sg_example.train_model(gen_dataset=self.gen_dataset_sg_example, steps=50,
                                                 save_model_flag=True,
                                                 metrics=[Metric("F1", mask='test')])
        metric_loc = gnn_model_manager_sg_example.evaluate_model(
            gen_dataset=self.gen_dataset_sg_example, metrics=[Metric("F1", mask='test', )])
        print(metric_loc)

        sg_example_model_path = gnn_model_manager_sg_example.model_path_info() / 'model'
        gnn_model_manager_sg_example.load_model_executor(path=sg_example_model_path)

    def test_combo_model_differ_acts_on_single_graph(self):
        gat_gcn_sage_gcn_gcn = model_configs_zoo(dataset=self.gen_dataset_sg_example, model_name="gat_gcn_sage_gcn_gcn")

        gnn_model_manager_sg_example = FrameworkGNNModelManager(
            gnn=gat_gcn_sage_gcn_gcn,
            dataset_path=self.results_dataset_path_sg_example,
            modification=self.default_config,
            manager_config=self.manager_config,
        )

        gnn_model_manager_sg_example.train_model(gen_dataset=self.gen_dataset_sg_example, steps=50,
                                                 save_model_flag=True,
                                                 metrics=[Metric("F1", mask='test')])
        metric_loc = gnn_model_manager_sg_example.evaluate_model(
            gen_dataset=self.gen_dataset_sg_example, metrics=[Metric("F1", mask='test', )])
        print(metric_loc)

        sg_example_model_path = gnn_model_manager_sg_example.model_path_info() / 'model'
        gnn_model_manager_sg_example.load_model_executor(path=sg_example_model_path)

    def test_model_on_multiple_graph(self):
        gin3_lin2_mg_small = model_configs_zoo(dataset=self.gen_dataset_mg_small,
                                               model_name='gin_gin_gin_lin_lin')

        gnn_mm_mg_small = FrameworkGNNModelManager(
            gnn=gin3_lin2_mg_small,
            dataset_path=self.results_dataset_path_mg_small,
            modification=self.default_config,
            manager_config=self.manager_config,
        )

        gnn_mm_mg_small.train_model(gen_dataset=self.gen_dataset_mg_small, steps=100,
                                    metrics=[Metric("F1", mask='val'),
                                             Metric("F1", mask='test')])
        metric_loc = gnn_mm_mg_small.evaluate_model(
            gen_dataset=self.gen_dataset_mg_small, metrics=[Metric("F1", mask='test', average='macro')])
        print(metric_loc)

    def test_model_on_multiple_graph_with_skip_connection(self):
        gin3_lin2_conn_mg_small = model_configs_zoo(dataset=self.gen_dataset_mg_small,
                                                    model_name='gin_gin_gin_lin_lin')

        gnn_mm_conn_mg_small = FrameworkGNNModelManager(
            gnn=gin3_lin2_conn_mg_small,
            manager_config=self.manager_config,
            modification=self.default_config,
            dataset_path=self.results_dataset_path_mg_small)

        gnn_mm_conn_mg_small.train_model(gen_dataset=self.gen_dataset_mg_small, steps=100,
                                         metrics=[Metric("F1", mask='test')])
        metric_loc = gnn_mm_conn_mg_small.evaluate_model(
            gen_dataset=self.gen_dataset_mg_small, metrics=[Metric("F1", mask='test', average='macro')])
        print(metric_loc)

    def test_model_on_multiple_graph_with_prot(self):
        gin3_lin2_prot_mg_small = model_configs_zoo(dataset=self.gen_dataset_mg_small,
                                                    model_name='gin_gin_gin_lin_lin_prot')

        prot_gnn_mm_mg_small = ProtGNNModelManager(
            gnn=gin3_lin2_prot_mg_small,
            manager_config=self.manager_config,
            modification=self.default_config,
            dataset_path=self.results_dataset_path_mg_small)

        best_acc = prot_gnn_mm_mg_small.train_model(gen_dataset=self.gen_dataset_mg_small, steps=100, metrics=[])
        metric_loc = prot_gnn_mm_mg_small.evaluate_model(
            gen_dataset=self.gen_dataset_mg_small, metrics=[Metric("F1", mask='test', average='macro')])
        print(metric_loc)
        mg_small_model_path = prot_gnn_mm_mg_small.model_path_info() / 'model'
        prot_gnn_mm_mg_small.load_model_executor(path=mg_small_model_path)


if __name__ == '__main__':
    unittest.main()
