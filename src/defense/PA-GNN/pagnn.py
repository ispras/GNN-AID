from typing import Iterable, Any, Optional

import torch.optim as optim
import torch
import scipy as sp
from torch_geometric.utils import to_undirected, subgraph
from tqdm import tqdm
from scipy.sparse.csgraph import connected_components

from src.aux.utils import POISON_ATTACK_PARAMETERS_PATH, POISON_DEFENSE_PARAMETERS_PATH, EVASION_ATTACK_PARAMETERS_PATH, \
    EVASION_DEFENSE_PARAMETERS_PATH
from src.models_builder.gnn_models import FrameworkGNNModelManager, Metric
from src.aux.configs import ModelModificationConfig, ConfigPattern, EvasionAttackConfig
from src.base.datasets_processing import DatasetManager, GeneralDataset
from src.models_builder.models_zoo import model_configs_zoo

from attacks.evasion_attacks import FGSMAttacker
from attacks.QAttack import qattack

from defense.poison_defense import PoisonDefender


class PAGNNDefender(PoisonDefender):
    name = 'PAGNNDefender'

    def __init__(self,
                 attack_name: str = None,
                 attack_config: EvasionAttackConfig = None,
                 attack_type: str = None,
                 clean_cnt: int = 5,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.clean_cnt = clean_cnt

        # TODO re-write when AdvTraining will be re-written- tmp code from AdvTraining due to similarity of methods.
        self.attacker = None
        if not attack_config:
            # build default config
            assert attack_name is not None
            if attack_type == "POISON":
                self.attack_type = "POISON"
                PARAM_PATH = POISON_ATTACK_PARAMETERS_PATH
            else:
                # raise ValueError("Evasion attack prohibited for this method")
                pass
            attack_config = ConfigPattern(
                _class_name=attack_name,
                _import_path=PARAM_PATH,
                _config_class="EvasionAttackConfig",
                _config_kwargs={}
            )
        self.attack_config = attack_config
        if self.attack_config._class_name == "MetaAttackFull":
            raise NotImplementedError("Check MetaAttack here")
            # from attacks.poison_attacks_collection.metattack import meta_gradient_attack
            # self.attack_type = "POISON"
            # self.num_nodes = self.attack_config._config_kwargs["num_nodes"]
            # self.attacker = meta_gradient_attack.MetaAttackFull(num_nodes=self.num_nodes)
            pass
        else:
            raise KeyError(f"There is no {self.attack_config._class_name} class")

        assert self.attacker is not None


    def defense(
            self,
            gen_dataset: GeneralDataset,
            **kwargs
    ) -> GeneralDataset:
        assert 'model' not in kwargs.keys(), 'Model must be provided in PA-GNN algorithm'
        model = kwargs['model']
        self.init()
        self.pretrain()
        self.train()
        self.finetune()
        return gen_dataset

    def init(
            self,
            gen_dataset: GeneralDataset
    ):
        cln_graphs = []
        ptb_graphs = []

        # Step 1: Convert graph to undirected
        edge_index_undirected = to_undirected(gen_dataset.dataset.data.edge_index)
        # Step 2: Randomly shuffle node indices and split into clean_cnt groups
        num_nodes = gen_dataset.dataset.data.x.size(0)
        indices = torch.randperm(num_nodes)
        subgraph_sizes = num_nodes // self.clean_cnt

        # sample clean graphs
        for i in range(self.clean_cnt):
            start = i * subgraph_sizes
            end = num_nodes if i == self.clean_cnt - 1 else (i + 1) * subgraph_sizes
            sub_nodes = indices[start:end]
            # Step 3: Extract subgraph edges
            sub_edge_index, _ = subgraph(sub_nodes, edge_index_undirected, relabel_nodes=True)  # TODO check if relabel needed
            num_sub_nodes = sub_nodes.size(0)
            # Step 4: Create adjacency matrix in sparse format
            adj_matrix = sp.csr_matrix(
                (torch.ones(sub_edge_index.size(1)),
                 (sub_edge_index[0].numpy(), sub_edge_index[1].numpy())),
                shape=(num_sub_nodes, num_sub_nodes)
            )
            # Step 5: Find connected components
            num_components, labels = connected_components(adj_matrix, directed=False)
            # Step 6: Keep the largest component
            largest_component = max(range(num_components), key=lambda c: (labels == c).sum())
            largest_nodes = torch.tensor([i for i in range(num_sub_nodes) if labels[i] == largest_component])
            # Step 7: Extract final connected subgraph
            final_edge_index, _ = subgraph(largest_nodes, sub_edge_index, relabel_nodes=True)
            sub_x = gen_dataset.dataset.data.x[sub_nodes, largest_nodes]
            cln_graphs.append((sub_x, final_edge_index))

        # sample perturbed graphs
        for i in range(self.clean_cnt):



        assert self.cln_graphs is not None
        assert self.ptb_graps is not None

    def pretrain(self):
        pass

    def train(self):
        pass

    def finetune(self):
        pass
