from defense.defense_base import Defender


class MIDefender(
    Defender
):
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
