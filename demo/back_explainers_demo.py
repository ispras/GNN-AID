import warnings
from torch import device
from torch.cuda import is_available

from datasets.datasets_manager import DatasetManager
from data_structures.configs import DatasetConfig, DatasetVarConfig, ExplainerInitConfig, ExplainerRunConfig
from explainers.explainers_manager import FrameworkExplainersManager
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from models_builder.models_zoo import model_configs_zoo


def explainers_test():
    my_device = device('cuda' if is_available() else 'cpu')

    # Init datasets
    dataset_mg_small, _, results_dataset_path_mg_small = DatasetManager.get_by_full_name(
        full_name=("multiple-graphs", "custom", "small",),
        features={'attr': {'a': 'as_is'}},
        labeling='binary',
        dataset_ver_ind=0
    )

    dataset_sg_example, _, results_dataset_path_sg_example = DatasetManager.get_by_full_name(
        full_name=("single-graph", "custom", "example",),
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
    gen_dataset_sg_example = DatasetManager.get_by_config(
        DatasetConfig(
            domain="single-graph",
            group="custom",
            graph="example"),
        DatasetVarConfig(features={'attr': {'a': 'as_is'}},
                         labeling='binary',
                         dataset_ver_ind=0)
    )
    gen_dataset_mg_small.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
    gen_dataset_sg_example.train_test_split(percent_train_class=0.6, percent_test_class=0.4)


    dataset_mg_small = gen_dataset_mg_small
    results_dataset_path_mg_small = gen_dataset_mg_small.results_dir

    dataset_sg_example = gen_dataset_sg_example
    results_dataset_path_sg_example = gen_dataset_sg_example.results_dir

    # Init gnns and gnn_model_managers
    # gat2_cora = model_configs_zoo(dataset=dataset_cora, model_name='gat_gat')
    # gcn2_comp = model_configs_zoo(dataset=dataset_comp, model_name='gcn_gcn')
    # gin3_lin2_mg_example = model_configs_zoo(dataset=dataset_mg_example, model_name='gin_gin_gin_lin_lin')

    gin3_lin2_mg_small = model_configs_zoo(dataset=dataset_mg_small, model_name='gin_gin_gin_lin_lin')

    gnn_model_manager_mg_small = FrameworkGNNModelManager(
        gnn=gin3_lin2_mg_small,
        dataset_path=results_dataset_path_mg_small,
    )

    # gin3_lin2_sg_example = model_configs_zoo(dataset=dataset_sg_example, model_name='gin_gin_gin_lin_lin')
    gin3_lin2_sg_example = model_configs_zoo(dataset=dataset_sg_example, model_name='gcn_gcn_lin')

    gnn_model_manager_sg_example = FrameworkGNNModelManager(
        gnn=gin3_lin2_sg_example,
        dataset_path=results_dataset_path_sg_example,
    )

    # Train models
    # warnings.warn("Start training small")
    # gnn_model_manager_mg_small.train_model(gen_dataset=dataset_mg_small, steps=50, save_model_flag=False,
    #                                          metrics=[Metric("F1", mask='test')])
    # warnings.warn("Training was  successful")

    warnings.warn("Start training example")
    gnn_model_manager_sg_example.train_model(gen_dataset=dataset_sg_example, steps=50, save_model_flag=False,
                                             metrics=[Metric("F1", mask='test')])
    warnings.warn("Training was  successful")

    # Explain
    # warnings.warn("Start SubgraphX")
    # explainer_SubgraphX = FrameworkExplainersManager(
    #     explainer_name='SubgraphX',
    #     dataset=dataset_mg_small, gnn_manager=gnn_model_manager_mg_small,
    #     explainer_ver_ind=0,
    # )
    # explainer_SubgraphX.conduct_experiment()

    warnings.warn("Start SubgraphX")
    explainer_SubgraphX = FrameworkExplainersManager(
        init_config=ExplainerInitConfig('SubgraphX'),
        dataset=dataset_sg_example, gnn_manager=gnn_model_manager_sg_example,
    )
    explainer_SubgraphX.conduct_experiment(ExplainerRunConfig('SubgraphX', mode="local"))

    warnings.warn("Start Zorro")
    explainer_SubgraphX = FrameworkExplainersManager(
        init_config=ExplainerInitConfig('Zorro'),
        dataset=dataset_sg_example, gnn_manager=gnn_model_manager_sg_example,
    )
    explainer_SubgraphX.conduct_experiment(ExplainerRunConfig('Zorro', mode="local"))

    warnings.warn("Start GraphMask")
    explainer_SubgraphX = FrameworkExplainersManager(
        init_config=ExplainerInitConfig('GraphMask'),
        dataset=dataset_sg_example, gnn_manager=gnn_model_manager_sg_example,
    )
    explainer_SubgraphX.conduct_experiment(ExplainerRunConfig('GraphMask', mode="local"))

    # warnings.warn("Start GraphMask")
    # explainer_GraphMask = FrameworkExplainersManager(
    #     explainer_name='GraphMask',
    #     dataset=dataset_cora, gnn_manager=gnn_model_manager_cora,
    #     explainer_ver_ind=0,
    # )
    # explainer_GraphMask.conduct_experiment(ExplainerRunConfig("local"))

    # # Explain Computers
    # FIXME GNNExplainer not working yet, waiting for Misha S fix
    # warnings.warn("Start GNNExplainer(torch-geom)")
    # explainer_GNNExplainer = FrameworkExplainersManager(
    #     explainer_name='GNNExplainer(torch-geom)',
    #     dataset=dataset_comp, gnn_manager=gnn_model_manager_comp,
    #     explainer_ver_ind=0,
    # )
    # explainer_GNNExplainer.conduct_experiment(ExplainerRunConfig("local"))


if __name__ == '__main__':
    explainers_test()
