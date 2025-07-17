import warnings

from datasets.datasets_manager import DatasetManager
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from data_structures.configs import ModelManagerConfig, DatasetConfig, DatasetVarConfig
from models_builder.models_zoo import model_configs_zoo


def backend_demo():
    # my_device = device('cuda' if is_available() else 'cpu')

    # Init datasets VK and Cora
    # dataset_cora, _, results_dataset_path_cora = DatasetManager.get_by_full_name(
    #     full_name=(LibPTGDataset.data_folder, "single-graph", "Planetoid", "Cora"),
    #     dataset_ver_ind=0)
    # dataset_comp, _, results_dataset_path_comp = DatasetManager.get_by_full_name(
    #     full_name=("single-graph", "Amazon", "Computers",),
    #     dataset_ver_ind=0
    # )
    # dataset_mg_example, _, results_dataset_path_mg_example = DatasetManager.get_by_full_name(
    #     full_name=("multiple-graphs", "custom", "example",),
    #     features=FeatureConfig(node_attr=['type']),
    #     labeling='binary',
    #     dataset_ver_ind=0
    # )
    dataset_cora = DatasetManager.get_by_config(
        DatasetConfig(
            domain="single-graph",
            group="Planetoid",
            graph="Cora"),
    )
    dataset_cora.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
    results_dataset_path_cora = dataset_cora.results_dir

    dataset_comp = DatasetManager.get_by_config(
        DatasetConfig(
            domain="single-graph",
            group="Amazon",
            graph="Computers"),
    )
    dataset_comp.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
    results_dataset_path_comp = dataset_comp.results_dir

    gen_dataset_mg_example = DatasetManager.get_by_config(
        DatasetConfig(
            domain="multiple-graphs",
            group="custom",
            graph="example"),
        DatasetVarConfig(features=FeatureConfig(node_attr=['type']),
                         labeling='binary',
                         dataset_ver_ind=0)
    )
    gen_dataset_mg_example.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
    dataset_mg_example = gen_dataset_mg_example
    results_dataset_path_mg_example = gen_dataset_mg_example.results_dir

    # Init gnns and gnn_model_managers
    gat2_cora = model_configs_zoo(dataset=dataset_cora, model_name='gat_gat')
    gcn2_comp = model_configs_zoo(dataset=dataset_comp, model_name='gcn_gcn')
    gin3_lin2_mg_example = model_configs_zoo(dataset=dataset_mg_example, model_name='gin_gin_gin_lin_lin')

    gnn_model_manager_comp = FrameworkGNNModelManager(
        gnn=gcn2_comp,
        dataset_path=results_dataset_path_comp,
        manager_config=ModelManagerConfig(batch=2500, mask_features=[])
    )

    gnn_model_manager_cora = FrameworkGNNModelManager(
        gnn=gat2_cora,
        dataset_path=results_dataset_path_cora,
        manager_config=ModelManagerConfig(batch=10000, mask_features=[])
    )

    gnn_model_manager_mg_example = FrameworkGNNModelManager(
        gnn=gin3_lin2_mg_example,
        dataset_path=results_dataset_path_mg_example,
        manager_config=ModelManagerConfig(batch=10000, mask_features=[])
    )

    # Train models
    warnings.warn("Start training Cora")
    gnn_model_manager_cora.train_model(gen_dataset=dataset_cora, steps=50, save_model_flag=False,
                                       metrics=[Metric("BalancedAccuracy", mask='test')])
    warnings.warn("Training was  successful")

    warnings.warn("Start training Computers")
    gnn_model_manager_comp.train_model(gen_dataset=dataset_comp, steps=50, save_model_flag=False,
                                       metrics=[Metric("Accuracy", mask='test')])
    warnings.warn("Training was  successful")

    warnings.warn("Start training example")
    gnn_model_manager_mg_example.train_model(gen_dataset=dataset_mg_example, steps=50, save_model_flag=False,
                                             metrics=[Metric("F1", mask='test')])
    warnings.warn("Training was  successful")

    # # Explain Computers
    # warnings.warn("Start GraphMask")
    # explainer_GraphMask = FrameworkExplainersManager(
    #     explainer_name='GraphMask',
    #     dataset=dataset_cora, gnn_manager=gnn_model_manager_cora,
    #     explainer_ver_ind=0,
    # )
    # explainer_GraphMask.conduct_experiment(ExplainerRunConfig("local"))

    # # Explain Computers
    # warnings.warn("Start GNNExplainer(torch-geom)")
    # explainer_GNNExplainer = FrameworkExplainersManager(
    #     explainer_name='GNNExplainer(torch-geom)',
    #     dataset=dataset_comp, gnn_manager=gnn_model_manager_comp,
    #     explainer_ver_ind=0,
    # )
    # explainer_GNNExplainer.conduct_experiment(ExplainerRunConfig("local"))


if __name__ == '__main__':
    backend_demo()
