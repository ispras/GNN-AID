from defenses.defense_base import Defender


class MIDefender(
    Defender
):
    """ Base class for all membership inference (MI) defense methods.
    """
    def __init__(
            self,
            **kwargs
    ):
        super().__init__()

    def pre_batch(
            self,
            **kwargs
    ):
        pass

    def post_batch(
            self,
            **kwargs
    ):
        pass


class EmptyMIDefender(MIDefender):
    """ Just a stub for MI defense.
    """
    name = "EmptyMIDefender"

    def pre_batch(
            self,
            **kwargs
    ):
        pass

    def post_batch(
            self,
            **kwargs
    ):
        pass
