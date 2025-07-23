import torch

import warnings

from aux.data_info import UserCodeInfo
from aux.utils import import_by_name, model_managers_info_by_names_list, TECHNICAL_PARAMETER_KEY
from models_builder.gnn_models import Metric
from data_structures.configs import CONFIG_CLASS_NAME
from datasets.datasets_manager import DatasetManager


def test_Konst_model():
    # my_device = device('cuda' if is_available() else 'cpu')

    # full_name = None

    # full_name = (LibPTGDataset.data_folder, "multiple-graphs", "TUDataset", "MUTAG")
    # full_name = ("single-graph", "custom", 'karate')
    # full_name = (LibPTGDataset.data_folder, "single-graph", "Planetoid", "Cora")
    # full_name = ("multiple-graphs", "TUDataset", 'PROTEINS')
    full_name = ("example", "single-graph", "example",)

    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=full_name,
        features=FeatureConfig(node_attr=['a', 'b']),
        labeling='threeClasses',
        dataset_ver_ind=0
    )

    user_model_class = 'SimGNN'
    user_model_obj = 'model_3'

    user_models_obj_dict_info = UserCodeInfo.user_models_list_ref()
    user_model_path = ''
    for key, value in user_models_obj_dict_info.items():
        if user_model_class in user_models_obj_dict_info.keys() and user_model_obj in value['obj_names']:
            user_model_path = value['import_path']

    gnn = UserCodeInfo.take_user_model_obj(user_model_path, user_model_obj)
    # gnn = model_configs_zoo(dataset=dataset, model_name='gin_gin_gin_lin_lin')
    # gnn = model_configs_zoo(dataset=dataset, model_name='gin_gin_gin_lin_lin_prot')

    # try:
    #     print(gnn.forward())
    # except NotImplementedError:
    #     print('NotImplementedError')
    # except:
    #     print('Error, but function implement and callable')

    steps_epochs = 1
    save_model_flag = True

    manager_config = {
        CONFIG_CLASS_NAME: "SimGnnMM",
        "lr": 0.01,
        "weight_decay": 5e-4,
        "limit": 30,
        "epochs": steps_epochs,
        "train_test_split": [
            0.6,
            0.4
        ],
    }

    klass = manager_config.pop(CONFIG_CLASS_NAME)
    from models_builder.gnn_models import ModelManagerConfig
    manager_config = ModelManagerConfig(**manager_config)

    # Build model manager

    mm_info = model_managers_info_by_names_list({klass})
    klass = import_by_name(klass, [mm_info[klass][TECHNICAL_PARAMETER_KEY]["import_info"]])
    model_manager = klass(
        gnn=gnn,
        manager_config=manager_config,
        dataset_path=dataset.prepared_dir)

    warnings.warn("Start training")
    dataset.train_test_split()

    try:
        raise FileNotFoundError()
        # gen_gnn_mm.load_model_executor(steps=steps_epochs, model_ver_ind=0)
    except FileNotFoundError:
        train_test_split_path = model_manager.train_model(gen_dataset=dataset, steps=steps_epochs,
                                                          save_model_flag=save_model_flag,
                                                          metrics=[Metric("F1", mask='train', average=None)])

        if train_test_split_path is not None:
            dataset.save_train_test_mask(train_test_split_path)
            train_mask, val_mask, test_mask, train_test_sizes = torch.load(train_test_split_path / 'train_test_split')[
                                                                :]
            dataset.train_mask, dataset.val_mask, dataset.test_mask = train_mask, val_mask, test_mask
            data.percent_train_class, data.percent_test_class = train_test_sizes

    warnings.warn("Training was successful")

    # metric_loc = gen_gnn_mm.evaluate_model(
    #     gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro')])
    # print(metric_loc)

    # Explain node 10
    # node = 10

    # Explanation size
    # max_nodes = 5

    # warnings.warn("Start GNNExplainer_DIG")
    # explainer_GNNExpl = FrameworkExplainersManager(
    #     explainer_name='GNNExplainer(dig)', dataset=dataset,
    #     gnn_manager=gen_gnn_mm, explainer_ver_ind=0)
    # explainer_GNNExpl.conduct_experiment(mode="local", element_idx=0)

    # explanation = explainer_SubgraphX.explanation
    # draw(node, explainer_SubgraphX.explanation_file_path, data)

    # return metric_loc


if __name__ == '__main__':
    test_Konst_model()
