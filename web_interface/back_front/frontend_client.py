import json

from aux.utils import FUNCTIONS_PARAMETERS_PATH, FRAMEWORK_PARAMETERS_PATH, MODULES_PARAMETERS_PATH, \
    EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH, \
    EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH, OPTIMIZERS_PARAMETERS_PATH
from web_interface.back_front.explainer_blocks import ExplainerRunBlock, ExplainerInitBlock, \
    ExplainerWBlock, ExplainerLoadBlock
from web_interface.back_front.model_blocks import ModelWBlock, ModelManagerBlock, ModelLoadBlock, \
    ModelConstructorBlock, ModelCustomBlock, ModelTrainerBlock
from web_interface.back_front.dataset_blocks import DatasetBlock, DatasetVarBlock
from web_interface.back_front.diagram import Diagram
from web_interface.back_front.utils import WebInterfaceError, SocketConnect


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
        'F': None, 'FW': None, 'M': None, 'EI': None, 'ER': None, 'O': None}

    @staticmethod
    def get_parameters(type):
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
                  }[type], 'r') as f:
            FrontendClient.parameters[type] = json.load(f)

        return FrontendClient.parameters[type]

    def __init__(self, sid):
        self.sid = sid  # socket ID
        self.socket = SocketConnect(sid=sid)

        # Build the diagram
        self.diagram = Diagram()

        self.dcBlock = DatasetBlock("dc", socket=self.socket)
        self.dvcBlock = DatasetVarBlock("dvc", socket=self.socket)
        self.diagram.add_dependency(self.dcBlock, self.dvcBlock)

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

        self.mtBlock = ModelTrainerBlock("mt", socket=self.socket)
        self.diagram.add_dependency(
            [self.dvcBlock, self.mmcBlock, self.mloadBlock], self.mtBlock,
            lambda args: args[0] and (args[1] or args[2]))

        self.elBlock = ExplainerLoadBlock("el", socket=self.socket)

        self.eiBlock = ExplainerInitBlock("ei", socket=self.socket)

        self.eBlock = ExplainerWBlock("e", [self.elBlock, self.eiBlock], socket=self.socket)
        self.diagram.add_dependency(self.dvcBlock, self.eBlock)
        self.diagram.add_dependency(self.mtBlock, self.eBlock)

        self.erBlock = ExplainerRunBlock("er", socket=self.socket)
        self.diagram.add_dependency(self.eiBlock, self.erBlock)

    # def drop(self):
    #     """ Drop all current data
    #     """
    #     self.diagram.drop()

    def request_block(self, block, func, params: dict = None):
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
