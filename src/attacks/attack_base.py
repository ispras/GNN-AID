from typing import Type

from base.datasets_processing import GeneralDataset


class Attacker:
    name = "Attacker"

    def __init__(
            self
    ):
        pass

    def attack(
            self,
            **kwargs
    ):
        pass

    def attack_diff(
            self
    ):
        pass

    @staticmethod
    def check_availability(
            gen_dataset: GeneralDataset,
            model_manager: Type
    ):
        return False


