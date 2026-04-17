import json
import logging
from enum import Enum
from typing import Union
from multiprocessing import Process, Queue

from gnn_aid.aux.utils import (
    FUNCTIONS_PARAMETERS_PATH, FRAMEWORK_PARAMETERS_PATH, MODULES_PARAMETERS_PATH,
    EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
    EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH, OPTIMIZERS_PARAMETERS_PATH,
    POISON_ATTACK_PARAMETERS_PATH, POISON_DEFENSE_PARAMETERS_PATH, EVASION_ATTACK_PARAMETERS_PATH,
    EVASION_DEFENSE_PARAMETERS_PATH, MI_ATTACK_PARAMETERS_PATH, MI_DEFENSE_PARAMETERS_PATH)
from . import json_loads, json_dumps, ViewPoint
from .attack_defense_blocks import BeforeTrainBlock, AfterTrainBlock
from .dataset_blocks import DatasetBlock, DatasetVarBlock
from .diagram import Diagram
from .explainer_blocks import (
    ExplainerRunBlock, ExplainerInitBlock, ExplainerWBlock, ExplainerLoadBlock)
from .model_blocks import (
    ModelWBlock, ModelManagerBlock, ModelLoadBlock, ModelConstructorBlock, ModelCustomBlock,
    ModelTrainerBlock)
from .utils import WebInterfaceError, SocketConnect, get_sid_logger


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
            sid: str,
    ):
        self.mode = mode  # mode: analysis, interpretation, defense
        self.socket = socket_connect
        self.logger = get_sid_logger(sid)

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

        # --- Common
        self.view_point = None

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

    def _set_view_point(
            self,
            part: dict
    ) -> bool:
        """ Set a new viewpoint. Return True if new viewpoint is different from the current one.
        """
        view_point = ViewPoint(center=part.get('center'), depth=part.get('depth'))
        if view_point != self.view_point:
            self.view_point = view_point
            return True
        self.logger.info('setting the same viewpoint')
        return False

    def run_loop(
            self,
            response_queue: Queue,
            msg_queue: Queue,
            request_queue: Queue,
    ) -> None:
        while True:
            self.logger.info('Worker is waiting for command...')
            command = request_queue.get()

            if not isinstance(command, dict):
                raise WebInterfaceError(f"Unknown command format: {command!r}")

            type = command.get('type')
            args = command.get('args') or {}

            if type == "STOP":
                self.logger.info("Worker received STOP command; it will finish")
                break

            if type == "dataset":
                get = args.get('get')
                set = args.get('set')
                part = args.get('part')
                if part:
                    part = json_loads(part)
                    is_new = self._set_view_point(part)

                if set == "visible_part":
                    result = self.dvcBlock.set_visible_part(self.view_point)

                elif get == "data":
                    dataset_data = self.dvcBlock.visible_part.get_dataset_data(self.view_point)
                    data = dataset_data.to_json()
                    logging.info(f"Length of dataset_data: {len(data)}")
                    result = data

                elif get == "var_data":
                    if not self.dvcBlock.is_set():
                        result = ''
                    else:
                        dataset_var_data = self.dvcBlock.visible_part.get_dataset_var_data(self.view_point)
                        data = dataset_var_data.to_json()
                        logging.info(f"Length of dataset_var_data: {len(data)}")
                        result = data

                elif get == "stat":
                    stat = args.get('stat')
                    result = json_dumps(self.dcBlock.get_stat(stat))

                elif get == "index":
                    result = self.dcBlock.get_index()

                else:
                    raise WebInterfaceError(f"Unknown 'part' command {get} for dataset")

                response_queue.put(result)

            elif type == "block":
                block = args.get('block')
                func = args.get('func')
                params = args.get('params')
                if params:
                    params = json_loads(params)
                self.logger.info(f"request_block: block={block}, func={func}, params={params}")
                # TODO what if raise exception? process will stop
                self.request_block(block, func, params)
                response_queue.put('')
                self.logger.info("Worker puts result to response_queue")

            elif type == "model":
                do = args.get('do')
                get = args.get('get')

                if do:
                    self.logger.info(f"model.do: do={do}, params={args}")
                    if do == 'index':
                        type = args.get('type')
                        if type == "saved":
                            result = json_dumps(self.mloadBlock.get_index())
                        elif type == "custom":
                            result = json_dumps(self.mcustomBlock.get_index())
                    elif do in ['train', 'reset', 'run', 'save']:
                        result = self.mtBlock.do(do, args)
                    elif do in ['run with attacks']:
                        result = self.atBlock.do(do, args)
                    else:
                        raise WebInterfaceError(f"Unknown do command: '{do}'")

                if get:
                    if get == "satellites":
                        if self.mmcBlock.is_set():
                            part = args.get('part')
                            if part:
                                part = json_loads(part)
                                is_new = self._set_view_point(part)
                            dvd = self.mmcBlock.get_satellites(self.view_point)
                            data = dvd.to_json()
                            logging.info(f"Length of dataset_var_data: {len(data)}")
                            result = data
                        else:
                            result = ''

                assert result is not None
                response_queue.put(result)

            elif type == "explainer":
                do = args.get('do')

                self.logger.info(f"explainer.do: do={do}, params={args}")

                if do in ["run", "stop"]:
                    result = self.erBlock.do(do, args)

                elif do == 'index':
                    result = self.elBlock.get_index()

                # elif do == "save":
                #     return self.save_explanation()

                else:
                    raise WebInterfaceError(f"Unknown 'do' command {do} for explainer")

                response_queue.put(result)

            else:
                raise WebInterfaceError(f"Unknown command type: 'f{type}'")
