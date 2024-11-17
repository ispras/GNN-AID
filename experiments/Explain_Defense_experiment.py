"""
Experiment on attacking GNN via GNNExplainer's explanations
"""

import torch
import numpy as np
import warnings
import copy
import json

from dig.sslgraph.dataset import get_node_dataset
from pyscf.fci.cistring import gen_des_str_index
from torch import device
from tqdm import tqdm

from src.aux.utils import POISON_ATTACK_PARAMETERS_PATH, POISON_DEFENSE_PARAMETERS_PATH, EVASION_ATTACK_PARAMETERS_PATH, \
    EVASION_DEFENSE_PARAMETERS_PATH
from src.models_builder.gnn_models import FrameworkGNNModelManager, Metric
from src.aux.configs import ModelModificationConfig, ConfigPattern, CONFIG_OBJ
from src.base.datasets_processing import DatasetManager
from src.models_builder.models_zoo import model_configs_zoo

from explainers.explainer import ProgressBar

from aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH, \
    EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH
from explainers.explainers_manager import FrameworkExplainersManager

from explainers.GNNExplainer.torch_geom_our.out import GNNExplainer
from explainers.SubgraphX.out import SubgraphXExplainer
from explainers.Zorro.out import ZorroExplainer

def test():
    from attacks.EAttack.eattack_attack import EAttack
    from defense.JaccardDefense import jaccard_def
    from defense.evasion_defense import AdvTraining

    my_device = device('cpu')

    # Load dataset
    full_name = ("single-graph", "Planetoid", 'Cora')
    # full_name = ('single-graph', 'pytorch-geometric-other', 'KarateClub')
    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=full_name,
        dataset_ver_ind=0
    )

    # Train model on original dataset and remember the model metric and node predicted probability
    gcn_gcn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')

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
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=0)
    )

    gnn_model_manager.gnn.to(my_device)

    # poison_defense_config = ConfigPattern(
    #     _class_name="JaccardDefense",
    #     _import_path=POISON_DEFENSE_PARAMETERS_PATH,
    #     _config_class="PoisonDefenseConfig",
    #     _config_kwargs={
    #     }
    # )

    evasion_defense_config = ConfigPattern(
        _class_name="AdvTraining",
        _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
        _config_class="EvasionDefenseConfig",
        _config_kwargs={
        }
    )

    #gnn_model_manager.set_poison_defender(poison_defense_config=poison_defense_config)
    gnn_model_manager.set_evasion_defender(evasion_defense_config=evasion_defense_config)

    num_steps = 100
    gnn_model_manager.train_model(gen_dataset=dataset,
                                  steps=num_steps,
                                  save_model_flag=False)

    # Evaluate model

    acc_train = gnn_model_manager.evaluate_model(gen_dataset=dataset,
                                                 metrics=[Metric("Accuracy", mask='train')])['train']['Accuracy']


    acc_test = gnn_model_manager.evaluate_model(gen_dataset=dataset,
                                                metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
    print(f"BEFORE ATTACK\nAccuracy on train: {acc_train}. Accuracy on test: {acc_test}")
    # print(f"Accuracy on test: {acc_test}")

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

    # explainer_init_config = ConfigPattern(
    #     _class_name="SubgraphX",
    #     _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
    #     _config_class="ExplainerInitConfig",
    #     _config_kwargs={
    #     }
    # )
    # explainer_run_config = ConfigPattern(
    #     _config_class="ExplainerRunConfig",
    #     _config_kwargs={
    #         "mode": "local",
    #         "kwargs": {
    #             "_class_name": "SubgraphX",
    #             "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
    #             "_config_class": "Config",
    #             "_config_kwargs": {
    #
    #             },
    #         }
    #     }
    # )


    # node_inds = np.arange(dataset.dataset.data.x.shape[0])
    # explain_node_size = int((0.01 * len(node_inds)))
    # explaind_inds = np.random.choice(node_inds, explain_node_size)

    explaind_inds = [0, 1, 4, 5, 6, 8, 11, 13, 14, 15, 16, 17, 18, 22, 24, 25, 26, 29, 1862, 2130]

    init_kwargs = getattr(explainer_init_config, CONFIG_OBJ).to_dict()
    explainer = GNNExplainer(gen_dataset=dataset, model=gnn_model_manager.gnn, device=my_device, **init_kwargs)

    mode = getattr(explainer_run_config, CONFIG_OBJ).mode
    params = getattr(getattr(explainer_run_config, CONFIG_OBJ).kwargs, CONFIG_OBJ).to_dict()

    explanations = []
    for n in tqdm(explaind_inds):
        params['element_idx'] = n
        explainer.pbar = ProgressBar(None, "er", desc=f'{explainer.name} explaining')
        explainer.run(mode, params, finalize=True)
        explanations.append(copy.deepcopy(explainer.explanation))
        # with open(f"/home/sazonov/PycharmProjects/GNN-AID/experiments/results/expl_{n}.json", "w") as fout:
        #     json.dump(explainer.explanation.)
    out = {int(explaind_inds[i]): explanations[i].dictionary['data']['edges'] for i in range(len(explaind_inds))}
    with open(f"/experiments/cora_reverse_0.01/expl_No_Def.json", "w") as fout:
        json.dump(out, fout)

    # explainer = SubgraphXExplainer(gen_dataset=dataset, model=gnn_model_manager.gnn, device=my_device, **init_kwargs)
    # explainer = ZorroExplainer(gen_dataset=dataset, model=gnn_model_manager.gnn, device=my_device, **init_kwargs)

    # node_inds = np.arange(dataset.dataset.data.x.shape[0])
    # dataset = gen_dataset.dataset.data[mask_tensor]
    # num_nodes = len(node_inds)
    # attacked_node_size = int(num_nodes * self.attack_size)


    # edge_index = dataset.dataset.data.edge_index.tolist()
    # adj_list = {}
    # for u, v in zip(edge_index[0], edge_index[1]):
    #     # if u not in adj_list:
    #     #     adj_list[u] = [v]
    #     # else:
    #     #     adj_list[u].append(v)
    #     if v not in adj_list:
    #         adj_list[v] = [u]
    #     else:
    #         if u not in adj_list[v]:
    #             adj_list[v].append(u)
    # node_inds = [n for n in adj_list.keys() if len(adj_list[n]) > 1]
    # attacked_node_size = int((0.01 * len(node_inds)))
    # attack_inds = np.random.choice(node_inds, attacked_node_size)

    # mask = Metric.create_mask_by_target_list(y_true=dataset.labels, target_list=attack_inds)

    # explainer_GNNExpl.conduct_experiment(explainer_run_config)

    # Evaluate model

    # acc_attack = gnn_model_manager.evaluate_model(gen_dataset=dataset,
    #                                              metrics=[Metric("Accuracy", mask=mask)])[mask]['Accuracy']
    # print(f"AFTER ATTACK\nAccuracy: {acc_attack}")

    # acc_train = gnn_model_manager.evaluate_model(gen_dataset=dataset,
    #                                              metrics=[Metric("Accuracy", mask='train')])['train']['Accuracy']
    #
    #
    # acc_test = gnn_model_manager.evaluate_model(gen_dataset=dataset,
    #                                             metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
    # print(f"AFTER ATTACK\nAccuracy on train: {acc_train}. Accuracy on test: {acc_test}")






if __name__ == "__main__":
    # torch.manual_seed(1000)
    # np.random.seed(1000)
    test()