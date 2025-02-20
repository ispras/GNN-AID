from attacks.RL_S2V.utils import edge_index_to_dict_of_lists
from attacks.evasion_attacks import EvasionAttacker
from rl_modules import *
import utils

class RLS2VAttack(EvasionAttacker):
    name = "RLS2VAttacker"

    def __init__(
            self,
            num_steps : int = 500000,  # RL learning steps
            lr: float = 0.01
    ):
        super().__init__()
        self.num_steps = num_steps
        self.lr = lr

        self.env = None
        self.agent = None



    def attack(
            self,
            gen_dataset,
            mask_tensor: torch.Tensor
    ):

        self.setup(gen_dataset, mask_tensor)

        # TODO check if agent is trained
        if not self.agent.trained:
            self.agent.train(num_steps=self.num_steps, lr=self.lr)

        # TODO perform change of graph structure

        pass

    def setup(self, gen_dataset, mask, new_env=False):
        total = torch.nonzero(mask, as_tuple=True)[0]
        dict_of_lists = edge_index_to_dict_of_lists(gen_dataset.dataset.data.edge_index)

        if self.env is None or new_env:
            self.env = NodeAttackEnv(gen_dataset=gen_dataset, all_targets=total.tolist(), )
            self.agent = RLS2VAgent(self.env)