from typing import Type

from base.gen_dataset import GeneralDataset


class Defender:
    name = "Defender"

    def __init__(
            self
    ):
        pass

    def defense_diff(
            self
    ):
        pass

    @staticmethod
    def check_availability(
            gen_dataset: GeneralDataset,
            model_manager: Type
    ):
        return False


