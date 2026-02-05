import json

import torch
from torch import device

from gnn_aid.attacks import Attacker
from gnn_aid.aux.utils import FUNCTIONS_PARAMETERS_PATH, all_subclasses
from gnn_aid.data_structures import ModelStructureConfig, ModelConfig
from gnn_aid.data_structures.configs import DatasetConfig, DatasetVarConfig, FeatureConfig, Task, \
    ConfigPattern, ModelModificationConfig
from gnn_aid.datasets.datasets_manager import DatasetManager
from gnn_aid.datasets.ptg_datasets import LibPTGDataset
from gnn_aid.models_builder import FrameworkGNNConstructor
from gnn_aid.models_builder.models_utils import Metric
from gnn_aid.models_builder.model_managers import FrameworkGNNModelManager
from gnn_aid.models_builder.models_zoo import model_configs_zoo


def node_regression():
    dc = DatasetConfig(('example', 'example'))
    dvc = DatasetVarConfig(task=Task.NODE_REGRESSION, labeling='regression',
                           features=FeatureConfig(node_attr=['a']), dataset_ver_ind=0)

    gen_dataset = DatasetManager.get_by_config(dc, dvc)
    gen_dataset.info.check()

    print(gen_dataset.data)

    # gnn = model_configs_zoo(dataset=gen_dataset, model_name='gcn_gcn')
    # manager_config = ConfigPattern(
    #     _config_class="ModelManagerConfig",
    #     _config_kwargs={
    #         "mask_features": [],
    #         "optimizer": {
    #             "_class_name": "Adam",
    #             "_config_kwargs": {},
    #         }
    #     }
    # )
    #
    # steps_epochs = 10
    # my_device = device('cuda' if torch.cuda.is_available() else 'cpu')
    # gnn_model_manager = FrameworkGNNModelManager(
    #     gnn=gnn,
    #     dataset_path=gen_dataset.prepared_dir,
    #     manager_config=manager_config,
    #     modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    # )
    #
    # gnn_model_manager.gnn.to(my_device)
    # gen_dataset.data.to(my_device)
    #
    # gen_dataset.train_test_split()
    # gnn_model_manager.train_model(
    #     gen_dataset=gen_dataset, steps=steps_epochs,
    #     save_model_flag=False,
    #     metrics=[Metric("F1", mask='train', average=None)]
    # )
    # print("Training was successful")


def graph_regression():
    dc = DatasetConfig(('example', 'example3'))

    labeling_dict = {"0": 0.3, "1": 0.4, "2": 0.2}
    DatasetManager.add_labeling(dc, Task.GRAPH_REGRESSION, "regression", labeling_dict)

    dvc = DatasetVarConfig(task=Task.GRAPH_REGRESSION, labeling='regression',
                           features=FeatureConfig(node_attr=['type']), dataset_ver_ind=0)

    gen_dataset = DatasetManager.get_by_config(dc, dvc)
    gen_dataset.info.check()

    print(gen_dataset.data)


def edge_regression():
    dc = DatasetConfig(('example', 'example'))

    labeling_dict = {
        "10,11": 0.32,
        "11,12": 0.25,
        "11,13": 0.13,
        "11,15": 0.1,
        "12,13": 0.3,
        "12,17": 0.22,
        "15,14": 0.35,
        "15,16": 0.40
    }

    DatasetManager.add_labeling(dc, Task.EDGE_REGRESSION, "regression", labeling_dict)

    dvc = DatasetVarConfig(task=Task.EDGE_REGRESSION, labeling="regression",
                           features=FeatureConfig(node_attr=['a']), dataset_ver_ind=0)

    gen_dataset = DatasetManager.get_by_config(dc, dvc)

    print(gen_dataset.data)


def link_prediction():
    # dc = DatasetConfig(('example', 'example'))
    # dvc = DatasetVarConfig(features=FeatureConfig(node_attr=['a']),
    #                        task=Task.EDGE_PREDICTION, dataset_ver_ind=0)

    dc = DatasetConfig((LibPTGDataset.data_folder, 'Homogeneous', 'Planetoid', 'Cora'))
    dvc = LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.EDGE_PREDICTION})

    gen_dataset = DatasetManager.get_by_config(dc, dvc)
    print(gen_dataset.data)
    gen_dataset.train_test_split(percent_train_class=0.85, percent_test_class=0.1)

    gnn = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'SAGEConv',
                            'layer_kwargs': {
                                'in_channels': gen_dataset.num_node_features,
                                'out_channels': 16,
                            },
                        },
                    },
                    ## Decoder
                    ## var.1) 2-layer perceptron
                    # {
                    #     'label': 'd',
                    #     'function': {
                    #         'function_name': 'Concat',
                    #         'function_kwargs': None
                    #     }
                    # },
                    # {
                    #     'label': 'd',
                    #     'layer': {
                    #         'layer_name': 'Linear',
                    #         'layer_kwargs': {
                    #             'in_features': 16 * 2,
                    #             'out_features': 16,
                    #         },
                    #     },
                    #     'activation': {
                    #         'activation_name': 'ReLU',
                    #         'activation_kwargs': None,
                    #     },
                    # },
                    # {
                    #     'label': 'd',
                    #     'layer': {
                    #         'layer_name': 'Linear',
                    #         'layer_kwargs': {
                    #             'in_features': 16,
                    #             'out_features': 1,
                    #         },
                    #     },
                    # },
                    ## var.2) cosine sim
                    {  #
                        'label': 'd',
                        'function': {
                            'function_name': 'CosineSimilarity',
                            'function_kwargs': None
                        }
                    }
                ]
            )))

    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "batch": 128,
            "mask_features": [],
            "optimizer": {
                "_class_name": "Adam",
                "_config_kwargs": {},
            },
            "loss_function": {
                "_config_class": "Config",
                "_class_name": "BCEWithLogitsLoss",
                "_import_path": FUNCTIONS_PARAMETERS_PATH,
                "_class_import_info": ["torch.nn"],
                "_config_kwargs": {},
            },
            "neg_samples_ratio": 2
        }
    )

    steps_epochs = 3
    # my_device = device('cuda' if torch.cuda.is_available() else 'cpu')
    my_device = 'cpu'
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=gen_dataset.prepared_dir,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )

    gnn_model_manager.gnn.to(my_device)
    gen_dataset.data.to(my_device)

    gnn_model_manager.modification.epochs = 0
    gnn_model_manager.train_model(
        gen_dataset=gen_dataset, steps=steps_epochs,
        save_model_flag=False,
        metrics=[Metric("F1", mask='train', average=None)]
    )
    print("Training was successful")

    res = gnn_model_manager.run_model(
        gen_dataset=gen_dataset,
        mask='all',
        out='predictions'
    )
    print(json.dumps(res.tolist(), indent=2))
    return

    from gnn_aid.aux.utils import EVASION_ATTACK_PARAMETERS_PATH
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

    # атака
    # gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)
    #
    res = gnn_model_manager.evaluate_model(
        gen_dataset=gen_dataset,
        metrics=[
            Metric("Accuracy", mask='all'),
            Metric("AUC", mask='test'),
            # Metric("Precision@k", mask='test', k=50),
            # Metric("Precision@k", mask='test', k=500000),
            # Metric("Recall@k", mask='test', k=500000),
        ]
    )
    print(json.dumps(res, indent=2))

    # explainer
    from gnn_aid.aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH
    from gnn_aid.explainers.explainers_manager import FrameworkExplainersManager
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
        dataset=gen_dataset, gnn_manager=gnn_model_manager,
        explainer_name='GNNExplainer(torch-geom)',
    )
    explainer_GNNExpl.conduct_experiment(explainer_run_config)

    # Example how to get prediction for specific node pair (5,6)
    data = gen_dataset.data
    edge_label_index = torch.tensor([[5], [6]])

    # get embeddings for all nodes
    node_out = gnn(data.x, data.edge_index)

    # Get embeddings for our nodes
    # src = node_out[edge_label_index[0]]
    # dst = node_out[edge_label_index[1]]

    src = node_out[data.edge_index[0]]
    dst = node_out[data.edge_index[1]]

    # Pass to decoder and get the output
    edge_out = gnn.decode(src, dst)

    # 'logits':
    full_out = edge_out
    print(f"Logits: {full_out}")

    # 'answers':
    full_out = gnn.get_answer(full_out.unsqueeze(dim=1))
    # full_out = gnn.get_answer(edge_out=edge_out)
    print(f"Answers: {full_out}")


def ptg_example():
    import os.path as osp

    import torch
    from sklearn.metrics import roc_auc_score

    import torch_geometric.transforms as T
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import negative_sampling

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          add_negative_train_samples=False),
    ])
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, name='Cora', transform=transform)
    # After applying the `RandomLinkSplit` transform, the data is transformed from
    # a data object to a list of tuples (train_data, val_data, test_data), with
    # each element representing the corresponding split.
    train_data, val_data, test_data = dataset[0]

    class Net(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

        def encode(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            return self.conv2(x, edge_index)

        def decode(self, z, edge_label_index):
            return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

        def decode_all(self, z):
            prob_adj = z @ z.t()
            return (prob_adj > 0).nonzero(as_tuple=False).t()

    model = Net(dataset.num_features, 128, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        return loss

    @torch.no_grad()
    def test(data):
        model.eval()
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

    best_val_auc = final_test_auc = 0
    for epoch in range(1, 101):
        loss = train()
        val_auc = test(val_data)
        test_auc = test(test_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')

    print(f'Final Test: {final_test_auc:.4f}')

    z = model.encode(test_data.x, test_data.edge_index)
    final_edge_index = model.decode_all(z)


if __name__ == '__main__':
    # node_regression()
    # graph_regression()

    # edge_regression()
    link_prediction()

    # ptg_example()
    # for c in all_subclasses(Attacker):
    #     print(c)