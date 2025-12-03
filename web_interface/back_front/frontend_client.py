import json
from enum import Enum
from typing import Union

from gnn_aid.aux.utils import (
    FUNCTIONS_PARAMETERS_PATH, FRAMEWORK_PARAMETERS_PATH, MODULES_PARAMETERS_PATH,
    EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
    EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH, OPTIMIZERS_PARAMETERS_PATH,
    POISON_ATTACK_PARAMETERS_PATH, POISON_DEFENSE_PARAMETERS_PATH, EVASION_ATTACK_PARAMETERS_PATH,
    EVASION_DEFENSE_PARAMETERS_PATH, MI_ATTACK_PARAMETERS_PATH, MI_DEFENSE_PARAMETERS_PATH)
from .attack_defense_blocks import BeforeTrainBlock, AfterTrainBlock
from .dataset_blocks import DatasetBlock, DatasetVarBlock
from .diagram import Diagram
from .explainer_blocks import (
    ExplainerRunBlock, ExplainerInitBlock, ExplainerWBlock, ExplainerLoadBlock)
from .model_blocks import (
    ModelWBlock, ModelManagerBlock, ModelLoadBlock, ModelConstructorBlock, ModelCustomBlock,
    ModelTrainerBlock)
from .utils import WebInterfaceError, SocketConnect


class ClientMode(Enum):
    analysis = 'analysis'
    interpretation = 'interpretation'
    defense = 'defense'


class FrontendClient:
    """
    Frontend client.
    Keeps data currently loaded at frontend for the client: dataset, model, explainer.
    """

    # Global values.
    # TODO this should be updated regularly or by some event
    storage_index = {  # type -> PrefixStorage
        'D': None, 'DV': None, 'M': None, 'CM': None, 'E': None}
    parameters = {  # type -> Parameters dict
        'F': None, 'FW': None, 'M': None, 'EI': None, 'ER': None, 'O': None,
        'AD-pa': None, 'AD-pd': None, 'AD-ea': None, 'AD-ed': None, 'AD-ma': None, 'AD-md': None}

    @staticmethod
    def get_parameters(
            type: str
    ) -> Union[dict, None]:
        """
        """
        if type not in FrontendClient.parameters:
            WebInterfaceError(f"Unknown 'ask' argument 'type'={type}")

        with open({
                      'F': FUNCTIONS_PARAMETERS_PATH,
                      'FW': FRAMEWORK_PARAMETERS_PATH,
                      'M': MODULES_PARAMETERS_PATH,
                      'EI': EXPLAINERS_INIT_PARAMETERS_PATH,
                      'ELR': EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                      'EGR': EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH,
                      'O': OPTIMIZERS_PARAMETERS_PATH,
                      'AD-pa': POISON_ATTACK_PARAMETERS_PATH,
                      'AD-pd': POISON_DEFENSE_PARAMETERS_PATH,
                      'AD-ea': EVASION_ATTACK_PARAMETERS_PATH,
                      'AD-ed': EVASION_DEFENSE_PARAMETERS_PATH,
                      'AD-ma': MI_ATTACK_PARAMETERS_PATH,
                      'AD-md': MI_DEFENSE_PARAMETERS_PATH,
                  }[type], 'r') as f:
            FrontendClient.parameters[type] = json.load(f)

        return FrontendClient.parameters[type]

    def __init__(
            self,
            socket_connect: SocketConnect,
            mode: ClientMode,
    ):
        self.mode = mode  # mode: analysis, interpretation, defense
        self.socket = socket_connect

        # Build the diagram
        self.diagram = Diagram()

        # Dataset

        self.dcBlock = DatasetBlock("dc", socket=self.socket)
        self.dvcBlock = DatasetVarBlock("dvc", socket=self.socket)
        self.diagram.add_dependency(self.dcBlock, self.dvcBlock)

        # --- Model

        self.mloadBlock = ModelLoadBlock("mload", socket=self.socket)
        self.mconstrBlock = ModelConstructorBlock("mconstr", socket=self.socket)
        self.mcustomBlock = ModelCustomBlock("mcustom", socket=self.socket)

        self.mcBlock = ModelWBlock(
            "mc",
            [self.mloadBlock, self.mconstrBlock, self.mcustomBlock], socket=self.socket)
        self.diagram.add_dependency(self.dvcBlock, self.mcBlock)

        self.mmcBlock = ModelManagerBlock("mmc", socket=self.socket)
        self.diagram.add_dependency(
            [self.dvcBlock, self.mconstrBlock, self.mcustomBlock], self.mmcBlock,
            lambda args: args[0] and (args[1] or args[2]))

        if mode == ClientMode.defense:
            self.btBlock = BeforeTrainBlock("bt", socket=self.socket)
            self.diagram.add_dependency([self.dvcBlock, self.mmcBlock], self.btBlock)

            self.mtBlock = ModelTrainerBlock("mt", socket=self.socket)
            self.diagram.add_dependency(
                [self.dvcBlock, self.btBlock, self.mloadBlock], self.mtBlock,
                lambda args: args[0] and (args[1] or args[2]))

            self.atBlock = AfterTrainBlock("at", socket=self.socket)
            self.diagram.add_dependency([self.dvcBlock, self.mtBlock], self.atBlock)

        else:
            self.mtBlock = ModelTrainerBlock("mt", socket=self.socket)
            self.diagram.add_dependency(
                [self.dvcBlock, self.mmcBlock, self.mloadBlock], self.mtBlock,
                lambda args: args[0] and (args[1] or args[2]))

        # --- Explaining

        self.elBlock = ExplainerLoadBlock("el", socket=self.socket)
        self.eiBlock = ExplainerInitBlock("ei", socket=self.socket)

        self.eBlock = ExplainerWBlock("e", [self.elBlock, self.eiBlock], socket=self.socket)
        self.diagram.add_dependency([self.dvcBlock, self.train_block], self.eBlock)

        self.erBlock = ExplainerRunBlock("er", socket=self.socket)
        self.diagram.add_dependency(self.eiBlock, self.erBlock)

    # def drop(self):
    #     """ Drop all current data
    #     """
    #     self.diagram.drop()

    @property
    def train_block(self):
        """ Get the block that performs model training. """
        if self.mode == ClientMode.defense:
            return self.atBlock
        else:
            return self.mtBlock

    def request_block(
            self,
            block: str,
            func: str,
            params: dict = None
    ) -> object:
        """
        :param block: name of block
        :param func: block function to call
        :param params: function kwargs as a dict
        :return: jsonable result to be sent to frontend
        """
        assert func in ["modify", "submit", "unlock", "breik"]
        block = self.diagram.get(block)
        func = getattr(block, func)
        res = func(**params or {})
        return res
