"""
Experiment on attacking GNN via GNNExplainer's explanations
"""
import copy

import torch
import numpy as np
import warnings

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

from aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH, \
    EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH
from explainers.explainers_manager import FrameworkExplainersManager

from explainers.GNNExplainer.torch_geom_our.out import GNNExplainer
from explainers.Zorro.out import ZorroExplainer
from explainers.PGMExplainer.out import PGMExplainer

def test():
    #from attacks.EAttack.eattack_attack import EAttack
    from attacks.EAttack.experimental_code import EAttack

    my_device = device('cpu')

    # Load dataset
    # full_name = ("single-graph", "Planetoid", 'Cora')
    # full_name = ("single-graph", "Planetoid", "CiteSeer")
    #full_name = ('single-graph', 'Amazon', 'Computers')
    # full_name = ('single-graph', 'pytorch-geometric-other', 'KarateClub')
    # full_name = ('single-graph', 'Amazon', 'Photo')
    full_name = ('single-graph', 'Planetoid', 'PubMed')
    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=full_name,
        dataset_ver_ind=0
    )

    # Train model on original dataset and remember the model metric and node predicted probability
    gcn_gcn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')
    # gcn_gcn_gcn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn_gcn')

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

    num_steps = 200
    # gnn_model_manager.train_model(gen_dataset=dataset,
    #                               steps=num_steps,
    #                               save_model_flag=False)

    # Evaluate model

    # acc_train = gnn_model_manager.evaluate_model(gen_dataset=dataset,
    #                                              metrics=[Metric("Accuracy", mask='train')])['train']['Accuracy']
    #
    #
    # acc_test = gnn_model_manager.evaluate_model(gen_dataset=dataset,
    #                                             metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
    # print(f"BEFORE ATTACK\nAccuracy on train: {acc_train}. Accuracy on test: {acc_test}")
    # print(f"Accuracy on test: {acc_test}")

    explainer_init_config = ConfigPattern(
        _class_name="GNNExplainer(torch-geom)",
        _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
        _config_class="ExplainerInitConfig",
        _config_kwargs={
            # "node_mask_type": "common_attributes",
            # "edge_mask_type": None
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

    # explainer_init_config = ConfigPattern(
    #     _class_name="PGMExplainer",
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
    #             "_class_name": "PGMExplainer",
    #             "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
    #             "_config_class": "Config",
    #             "_config_kwargs": {
    #
    #             },
    #         }
    #     }
    # )

    init_kwargs = getattr(explainer_init_config, CONFIG_OBJ).to_dict()
    explainer = GNNExplainer(gen_dataset=dataset, model=gnn_model_manager.gnn, device=my_device, **init_kwargs)
    # explainer = SubgraphXExplainer(gen_dataset=dataset, model=gnn_model_manager.gnn, device=my_device, **init_kwargs)
    # explainer = ZorroExplainer(gen_dataset=dataset, model=gnn_model_manager.gnn, device=my_device, **init_kwargs)
    # explainer = PGMExplainer(gen_dataset=dataset, model=gnn_model_manager.gnn, device=my_device, **init_kwargs)

    # node_inds = np.arange(dataset.dataset.data.x.shape[0])
    # dataset = gen_dataset.dataset.data[mask_tensor]
    # num_nodes = len(node_inds)
    # attacked_node_size = int(num_nodes * self.attack_size)
    edge_index = dataset.dataset.data.edge_index.tolist()
    adj_list = {}
    for u, v in zip(edge_index[0], edge_index[1]):
        # if u not in adj_list:
        #     adj_list[u] = [v]
        # else:
        #     adj_list[u].append(v)
        if v not in adj_list:
            adj_list[v] = [u]
        else:
            if u not in adj_list[v]:
                adj_list[v].append(u)
    node_inds = [n for n in adj_list.keys() if len(adj_list[n]) > 1]
    # attacked_node_size = int((0.04 * len(node_inds)))
    attacked_node_size = 100
    # attacked_node_size = int((0.002 * len(node_inds)))
    attack_inds = np.random.choice(node_inds, attacked_node_size)
    # with open('/home/sazonov/PycharmProjects/GNN-AID/experiments/inds_pubmed.npy', 'wb') as fin:
    #     np.save(fin, attack_inds)
    with open('/home/sazonov/PycharmProjects/GNN-AID/experiments/inds_pubmed.npy', 'rb') as fin:
        attack_inds = np.load(fin)

    #attack_inds = attack_inds[:25]

    evasion_attack_config = ConfigPattern(
        _class_name="EAttackRandom",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
            'explainer': explainer,
            'run_config': explainer_run_config,
            'mode': 'local',
            'attack_inds': attack_inds,
            'random_rewire': True
        }
    )

    dataset_copy = copy.deepcopy(dataset)

    attack_type = ['random', 'original']
    modes = ['reverse', 'remove', 'add', 'rewire']
    #modes = ['rewire']
    #modes = ['drop']
    # modes = ['reverse']
    #probs = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    probs = [0.05, 0.1, 0.15]
    #probs = [0.005, 0.01, 0.02, 0.03]
    #probs = [0.01, 0.03, 0.05, 0.1, 0.15]
    # probs = [0.05, 0.1, 0.15]
    #probs = [0.00, 0.001, 0.01, 0.015]
    #probs = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    # probs = [0.000]

    with open('/home/sazonov/PycharmProjects/GNN-AID/experiments/results', 'w') as fout:
        for mode in modes:
            for prob in probs:
                for j in tqdm(range(1)):
                    gcn_gcn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')
                    #sage_sage = model_configs_zoo(dataset=dataset, model_name='sage_sage')
                    #gat_gat = model_configs_zoo(dataset=dataset, model_name='gat_gat')

                    gnn_model_manager = FrameworkGNNModelManager(
                        gnn=gcn_gcn,
                        dataset_path=results_dataset_path,
                        manager_config=manager_config,
                        modification=ModelModificationConfig(model_ver_ind=0, epochs=0)
                    )

                    gnn_model_manager.train_model(gen_dataset=dataset,
                                                  steps=num_steps,
                                                  save_model_flag=False)

                    gnn_answers = gnn_model_manager.gnn.get_answer(dataset.dataset.data.x, dataset.dataset.data.edge_index)

                    succ_attack_acc = 0
                    succ_attack_evasion = 0

                    if mode != 'reverse' and mode != 'drop':
                        attack_edges = True
                        attack_features = False
                        attack_mode = 'edge_mode'
                    else:
                        attack_edges = False
                        attack_features = True
                        attack_mode = 'features_mode'

                    for i in tqdm(attack_inds):

                        mask = Metric.create_mask_by_target_list(y_true=dataset.labels, target_list=[i])

                        evasion_attack_config = ConfigPattern(
                            _class_name="EAttackRandom",
                            _import_path=EVASION_ATTACK_PARAMETERS_PATH,
                            _config_class="EvasionAttackConfig",
                            _config_kwargs={
                                'explainer': explainer,
                                'run_config': explainer_run_config,
                                'mode': 'local',
                                'attack_inds': [i],
                                'attack_features': attack_features,
                                'attack_edges': attack_edges,
                                'edge_prob': prob,
                                'feature_prob': prob,
                                attack_mode: mode
                            }
                        )

                        # gnn_model_manager.gnn.get_answer(dataset.dataset.data.x, dataset.dataset.data.edge_index)[i]

                        gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

                        ground_truth = dataset.dataset.data.y[i]
                        model_answ_clear = gnn_answers[i]
                        model_answ_attack = gnn_model_manager.gnn.get_answer(dataset.dataset.data.x, dataset.dataset.data.edge_index)[i]

                        # y_pred, y_true = gnn_model_manager.evaluate_model(gen_dataset=dataset,
                        #                                              metrics=[Metric("Accuracy", mask=mask)])[mask]['Accuracy']
                        y_pred, y_true = gnn_model_manager.evaluate_model(gen_dataset=dataset,
                                                                     metrics=[Metric("Accuracy", mask=mask)])

                        # succ_attack += acc_attack
                        if y_pred != int(model_answ_clear):
                            succ_attack_evasion += 1
                        if y_pred != y_true:
                            succ_attack_acc += 1
                        # print(f"AFTER ATTACK\nAccuracy: {acc_attack}")

                        dataset = copy.deepcopy(dataset_copy)
                    fout.write(f"mode:{mode}\tprob:{prob}\tevasion:{succ_attack_evasion / len(attack_inds)}\tacc:{succ_attack_acc/ len(attack_inds)}\n")
                    fout.flush()
                    #print(f"ACCURACY ON ATTACKED: {succ_attack / len(attack_inds)}")






if __name__ == "__main__":
    test()