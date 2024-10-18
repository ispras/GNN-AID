import torch

from defense.defense_base import Defender
from src.aux.utils import import_by_name
from src.aux.configs import ModelModificationConfig, ConfigPattern
from src.aux.utils import POISON_ATTACK_PARAMETERS_PATH, POISON_DEFENSE_PARAMETERS_PATH, EVASION_ATTACK_PARAMETERS_PATH, \
    EVASION_DEFENSE_PARAMETERS_PATH
from attacks.evasion_attacks import FGSMAttacker
from attacks.QAttack import qattack
from torch_geometric import data

import copy

class EvasionDefender(Defender):
    def __init__(self, **kwargs):
        super().__init__()

    def pre_batch(self, **kwargs):
        pass

    def post_batch(self, **kwargs):
        pass


class EmptyEvasionDefender(EvasionDefender):
    name = "EmptyEvasionDefender"

    def pre_batch(self, **kwargs):
        pass

    def post_batch(self, **kwargs):
        pass


class GradientRegularizationDefender(EvasionDefender):
    name = "GradientRegularizationDefender"

    def __init__(self, regularization_strength=0.1):
        super().__init__()
        self.regularization_strength = regularization_strength

    def post_batch(self, model_manager, batch, loss, **kwargs):
        batch.x.requires_grad = True
        outputs = model_manager.gnn(batch.x, batch.edge_index)
        loss_loc = model_manager.loss_function(outputs, batch.y)
        gradients = torch.autograd.grad(outputs=loss_loc, inputs=batch.x,
                                        grad_outputs=torch.ones_like(loss_loc),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = torch.sum(gradients ** 2)
        return {"loss": loss + self.regularization_strength * gradient_penalty}


# TODO Kirill, add code in pre_batch
class QuantizationDefender(EvasionDefender):
    name = "QuantizationDefender"

    def __init__(self, qbit=8):
        super().__init__()
        self.regularization_strength = qbit

    def pre_batch(self, **kwargs):
        pass


class DataWrap:
    def __init__(self, batch) -> None:
        self.data = batch
        self.dataset = self

class AdvTraining(EvasionDefender):
    name = "AdvTraining"

    def __init__(self, attack_name=None, attack_config=None, attack_type=None, device='cpu'):
        super().__init__()
        assert device is not None, "Please specify 'device'!"
        if not attack_config:
            # build default config
            assert attack_name is not None
            if attack_type == "POISON":
                self.attack_type = "POISON"
                PARAM_PATH = POISON_ATTACK_PARAMETERS_PATH
            else:
                self.attack_type = "EVASION"
                PARAM_PATH = EVASION_ATTACK_PARAMETERS_PATH
            attack_config = ConfigPattern(
                _class_name=attack_name,
                _import_path=PARAM_PATH,
                _config_class="EvasionAttackConfig",
                _config_kwargs={}
            )
        self.attack_config = attack_config
        if self.attack_config._class_name == "FGSM":
            self.attack_type = "EVASION"
            # get attack params
            self.epsilon = self.attack_config._config_kwargs["epsilon"]
            # set attacker
            self.attacker = FGSMAttacker(self.epsilon)
        elif self.attack_config._class_name == "QAttack":
            self.attack_type = "EVASION"
            # get attack params
            self.population_size = self.attack_config._config_kwargs["population_size"]
            self.individual_size = self.attack_config._config_kwargs["individual_size"]
            self.generations = self.attack_config._config_kwargs["generations"]
            self.prob_cross = self.attack_config._config_kwargs["prob_cross"]
            self.prob_mutate = self.attack_config._config_kwargs["prob_mutate"]
            # set attacker
            self.attacker = qattack.QAttacker(self.population_size, self.individual_size, 
                                              self.generations, self.prob_cross,
                                              self.prob_mutate)
        elif self.attack_config._class_name == "MetaAttackFull":
            # from attacks.poison_attacks_collection.metattack import meta_gradient_attack
            # self.attack_type = "POISON"
            # self.num_nodes = self.attack_config._config_kwargs["num_nodes"]
            # self.attacker = meta_gradient_attack.MetaAttackFull(num_nodes=self.num_nodes)
            pass
        else:
            raise KeyError(f"There is no {self.attack_config._class_name} class")

    def pre_batch(self, model_manager, batch):
        super().pre_batch(model_manager=model_manager, batch=batch)
        self.perturbed_gen_dataset = data.Data()
        self.perturbed_gen_dataset.data = copy.deepcopy(batch)
        self.perturbed_gen_dataset.dataset = self.perturbed_gen_dataset.data
        self.perturbed_gen_dataset.dataset.data = self.perturbed_gen_dataset.data
        if self.attack_type == "EVASION":
            self.perturbed_gen_dataset = self.attacker.attack(model_manager=model_manager, 
                                                            gen_dataset=self.perturbed_gen_dataset,
                                                            mask_tensor=self.perturbed_gen_dataset.data.train_mask)

    
    def post_batch(self, model_manager, batch, loss) -> dict:
        super().post_batch(model_manager=model_manager, batch=batch, loss=loss)
        # Output on perturbed data
        outputs = model_manager.gnn(self.perturbed_gen_dataset.data.x, self.perturbed_gen_dataset.data.edge_index)
        loss_loc = model_manager.loss_function(outputs, batch.y)
        return {"loss": loss + loss_loc}
