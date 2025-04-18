from attacks.attack_base import Attacker


class MIAttacker(
    Attacker
):
    """ Base class for all membership inference (MI) attack methods.
    """
    def __init__(
            self,
            **kwargs
    ):
        super().__init__()


class EmptyMIAttacker(
    MIAttacker
):
    """ Just a stub for MI attack.
    """
    name = "EmptyMIAttacker"

    def attack(
            self,
            **kwargs
    ):
        pass
