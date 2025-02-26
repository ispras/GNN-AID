from attacks.RL_S2V.utils import edge_index_to_dict_of_lists
from attacks.evasion_attacks import EvasionAttacker
from rl_modules import *
import utils

class RLS2VAttack(EvasionAttacker):
    name = "RLS2VAttacker"

    def __init__(
            self,
            num_steps : int = 500000,
            lr: float = 0.01,
            num_mod: int = 1,
            reward_type: str = 'binary',
            batch_size: int = 10,
            bilin_q: bool = True,
            embed_dim: int = 64,
            mlp_hidden: int = 64,
            max_lv: bool = True,
            gm: str = 'mean_field'
    ):
        """
        :param num_steps: rl training steps
        :param lr:
        :param num_mod: number of modifications for graph
        :param reward_type: binary or nll
        :param batch_size: minibatch size for RL
        :param bilin_q: bilinear q or not
        :param embed_dim: dimension of latent layers
        :param mlp_hidden: mlp hidden layer size
        :param max_lv: max rounds of message passing
        :param gm: mean_field/loopy_bp/gcn

        """
        super().__init__()
        self.num_steps = num_steps
        self.lr = lr
        self.num_mod = num_mod
        self.reward_type = reward_type
        self.batch_size = batch_size
        self.bilin_q = bilin_q
        self.embed_dim = embed_dim
        self.mlp_hidden = mlp_hidden
        self.max_lv = max_lv
        self.gm = gm

        self.env = None
        self.agent = None



    def attack(
            self,
            model_manager,
            gen_dataset,
            mask_tensor: torch.Tensor
    ):
        gnn = model_manager.gnn
        self.setup(gen_dataset, gnn, mask_tensor)
        self.agent.train(num_steps=self.num_steps, lr=self.lr)

        # TODO perform change of graph structure

        pass

    def setup(self, gen_dataset, gnn, mask):
        dict_of_lists = edge_index_to_dict_of_lists(gen_dataset.dataset.data.edge_index)
        idx_test = torch.nonzero(mask, as_tuple=True)[0]
        pred_labels = gnn.get_answer(gen_dataset.x, gen_dataset.edge_index)
        # TODO check correctness here via debug
        acc = pred_labels.eq(gen_dataset.y).double()
        acc_test = acc[mask]

        attack_list = []
        for i in range(len(idx_test)):
            # only attack those misclassifed and degree>0 nodes
            if acc_test[i] > 0 and len(dict_of_lists[idx_test[i]]):
                attack_list.append(idx_test[i])

        total = attack_list
        idx_valid = idx_test

        # meta_list = []
        # num_wrong = 0
        # for i in range(len(idx_valid)):
        #     if acc_test[i] > 0:
        #         if len(dict_of_lists[idx_valid[i]]):
        #             meta_list.append(idx_valid[i])
        #     else:
        #         num_wrong += 1

        self.env = NodeAttackEnv(gen_dataset=gen_dataset, all_targets=total, list_action_space=dict_of_lists,
                                 classifier=gnn, num_mod=self.num_mod, reward_type=self.reward_type)
        self.agent = RLS2VAgent(self.env, gen_dataset, idx_meta=node_idx, idx_test=attack_list, num_wrong=1,
                                list_action_space=dict_of_lists, num_mod=self.num_mod, reward_type=self.reward_type,
                                batch_size=self.batch_size,
                                bilin_q=self.bilin_q, embed_dim=self.embed_dim, mlp_hidden=self.mlp_hidden,
                                max_lv=self.max_lv, gm=self.gm, device=gnn.device)