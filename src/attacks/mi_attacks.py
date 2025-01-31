from attacks.attack_base import Attacker


class MIAttacker(
    Attacker
):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__()


class EmptyMIAttacker(
    MIAttacker
):
    name = "EmptyMIAttacker"

    def attack(
            self,
            **kwargs
    ):
        pass
