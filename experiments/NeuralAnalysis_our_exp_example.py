from data_structures.configs import DatasetConfig, Task
from datasets.datasets_manager import DatasetManager
from datasets.ptg_datasets import LibPTGDataset
from models_builder.models_zoo import model_configs_zoo
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from explainers.explainers_manager import FrameworkExplainersManager


from torch import device
from torch.cuda import is_available


def test_neural_analysis(percent_train_class: float = 0.8, percent_test_class: float = 0.1):
    my_device = device('cuda' if is_available() else 'cpu')

    print('start loading data======================')
    dataset = DatasetManager.get_by_config(
        DatasetConfig((LibPTGDataset.data_folder, "Homogeneous", "TUDataset", "MUTAG")),
        LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.GRAPH_CLASSIFICATION})
    )
    data = dataset.data

    dataset.train_test_split(percent_train_class=percent_train_class, percent_test_class=percent_test_class)

    print('start training model====================')
    model = model_configs_zoo(dataset=dataset, model_name='gin_gin_gin_lin')
    #model = model_configs_zoo(dataset=dataset, model_name='gin_gin_gin_lin_lin')

    gnn_model_manager = FrameworkGNNModelManager(
        gnn=model,
        dataset_path=dataset.prepared_dir,
        batch=24
    )

    gnn_model_manager.train_model(gen_dataset=dataset, steps=50, save_model_flag=False, metrics=[Metric("F1", mask='test')])

    #gnn_model_manager.train_1_step(dataset)

    # steps_epochs = 100
    #     # # save_model_flag = False
    #     # save_model_flag = True
    #     #
    #     # data.x = data.x.float()
    #     # gnn_model_manager.gnn.to(my_device)
    #     # data = data.to(my_device)
    #     #
    #     # warnings.warn("Start training")
    #     # dataset.train_test_split()

    explainer_neuron = FrameworkExplainersManager(explainer_name='NeuralAnalysis', dataset=dataset,
                                                  gnn_manager=gnn_model_manager, explainer_ver_ind=0)

    explainer_neuron.conduct_experiment()

if __name__ == "__main__":
    test_neural_analysis()