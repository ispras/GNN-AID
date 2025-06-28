from attacks.RL_S2V.utils import edge_index_to_dict_of_lists
from attacks.evasion_attacks import EvasionAttacker
from attacks.RL_S2V.rl_modules import *
from datasets.gen_dataset import GeneralDataset


class RLS2VAttacker(EvasionAttacker):
    name = "RLS2VAttack"

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
            gm: str = 'mean_field',
            node_idx: int = 0
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
        :param node_idx: index of node to be attacked

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
        self.node_idx = node_idx

        self.env = None
        self.agent = None



    def attack(
            self,
            model_manager,
            gen_dataset,
            mask_tensor: torch.Tensor
    ) -> None:
        mask_tensor = torch.tensor(mask_tensor)
        gnn = model_manager.gnn
        self.setup(gen_dataset, gnn, mask_tensor)
        self.agent.train(num_steps=self.num_steps, lr=self.lr)
        edge_index, edge_weight = self.agent.eval()
        gen_dataset.dataset.data.edge_index = edge_index
        # QUE edge_weight implemented in out framework?

    def setup(
            self,
            gen_dataset: GeneralDataset,
            gnn,
            mask
            ) -> None:
        dict_of_lists = edge_index_to_dict_of_lists(gen_dataset.dataset.data.edge_index)
        idx_test = torch.nonzero(mask, as_tuple=True)[0].tolist()
        pred_labels = gnn.get_answer(gen_dataset.dataset.data.x, gen_dataset.dataset.data.edge_index)
        acc = pred_labels.eq(gen_dataset.dataset.data.y).double()
        acc_test = acc[mask]

        # attack_list = []
        # for i in range(len(idx_test)):
        #     # only attack those misclassifed and degree>0 nodes
        #     if acc_test[i] > 0 and len(dict_of_lists[idx_test[i]]):
        #         attack_list.append(idx_test[i])
        attack_list = [x for x in range(gen_dataset.dataset.data.x.shape[0])]

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

        device = gen_dataset.dataset.data.x.device

        self.env = NodeAttackEnv(gen_dataset=gen_dataset, all_targets=total, list_action_space=dict_of_lists,
                                 classifier=gnn, num_mod=self.num_mod, reward_type=self.reward_type, gm=self.gm)
        self.agent = RLS2VAgent(self.env, gen_dataset, node_idx=self.node_idx, idx_test=attack_list, num_wrong=1,
                                list_action_space=dict_of_lists, num_mod=self.num_mod, reward_type=self.reward_type,
                                batch_size=self.batch_size,
                                bilin_q=self.bilin_q, embed_dim=self.embed_dim, mlp_hidden=self.mlp_hidden,
                                max_lv=self.max_lv, gm=self.gm, device=device)