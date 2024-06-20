import json
import os

from aux.configs import ExplainerInitConfig, ExplainerModificationConfig, ExplainerRunConfig, \
    ConfigPattern
from aux.data_info import DataInfo
from aux.declaration import Declare
from aux.utils import MODELS_DIR, EXPLAINERS_INIT_PARAMETERS_PATH, \
    EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH, EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH
from base.datasets_processing import GeneralDataset
from explainers.explainers_manager import FrameworkExplainersManager
from models_builder.gnn_models import GNNModelManager
from web_interface.back_front.block import Block, WrapperBlock
from web_interface.back_front.utils import json_loads


class ExplainerWBlock(WrapperBlock):
    def __init__(self, name, blocks, *args, **kwargs):
        super().__init__(blocks, name, *args, **kwargs)

    def _init(self, gen_dataset: GeneralDataset, gmm: GNNModelManager):
        self.gen_dataset = gen_dataset
        self.gmm = gmm

    def _finalize(self):
        return True

    def _submit(self):
        pass


class ExplainerLoadBlock(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.explainer_path = None
        self.info = None
        self.gen_dataset = None
        self.gmm = None

    def _init(self, gen_dataset: GeneralDataset, gmm: GNNModelManager):
        # Define options for model manager
        self.gen_dataset = gen_dataset
        self.gmm = gmm
        return [gen_dataset.dataset.num_node_features, gen_dataset.is_multi(), self.get_index()]
        # return self.get_index()

    def _finalize(self):
        # if 1:  # TODO better check
        #     return False

        self.explainer_path = self._config
        return True

    def _submit(self):
        init_config, run_config = self._explainer_kwargs(model_path=self.gmm.model_path_info(),
                                                         explainer_path=self.explainer_path)
        modification_config = ExplainerModificationConfig(
            explainer_ver_ind=self.explainer_path["explainer_ver_ind"],
            explainer_attack_type=self.explainer_path["explainer_attack_type"])

        from explainers.explainers_manager import FrameworkExplainersManager
        self._object = FrameworkExplainersManager(
            init_config=init_config,
            modification_config=modification_config,
            dataset=self.gen_dataset,
            gnn_manager=self.gmm,
        )
        self._result = {
            "path": {k: json.loads(self.info[k][v]) if k in self.info else v
                     for k, v in self.explainer_path.items()},
            "explanation_data": self._object.load_explanation(run_config=run_config)
        }

    def get_index(self):
        """ Get all available explanations with respect to current dataset and model
        """
        path = os.path.relpath(self.gmm.model_path_info(), MODELS_DIR)
        keys_list, full_keys_list, dir_structure, _ = DataInfo.take_keys_etc_by_prefix(
            prefix=("data_root", "data_prepared", "models")
        )
        values_info = DataInfo.values_list_by_path_and_keys(path=path,
                                                            full_keys_list=full_keys_list,
                                                            dir_structure=dir_structure)
        DataInfo.refresh_explanations_dir_structure()
        index, self.info = DataInfo.explainers_parse()

        ps = index.filter(dict(zip(keys_list, values_info)))
        # return [ps.to_json(), json_dumps(self.info)] FIXME misha parsing error on front
        return [ps.to_json(), '{}']

    def _explainer_kwargs(self, model_path, explainer_path):
        init_kwargs_file, run_kwargs_file = Declare.explainer_kwargs_path_full(
            model_path=model_path, explainer_path=explainer_path)
        with open(init_kwargs_file) as f:
            init_config = ConfigPattern(**json.load(f))
        with open(run_kwargs_file) as f:
            run_config = ConfigPattern(**json.load(f))
        return init_config, run_config


class ExplainerInitBlock(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.explainer_init_config = None
        self.gen_dataset = None
        self.gmm = None

    def _init(self, gen_dataset: GeneralDataset, gmm: GNNModelManager):
        # Define options for model manager
        self.gen_dataset = gen_dataset
        self.gmm = gmm
        return FrameworkExplainersManager.available_explainers(self.gen_dataset, self.gmm)

    def _finalize(self):
        # if 1:  # TODO better check
        #     return False

        # self.explainer_init_config = ExplainerInitConfig(**self._config)
        self.explainer_init_config = ConfigPattern(
            **self._config,
            _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
            _config_class="ExplainerInitConfig")
        return True

    def _submit(self):
        # Build an explainer
        self._object = FrameworkExplainersManager(
            dataset=self.gen_dataset, gnn_manager=self.gmm,
            init_config=self.explainer_init_config,
            explainer_name=self.explainer_init_config._class_name
        )
        self._result = {"config": self.explainer_init_config}


class ExplainerRunBlock(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.explainer_run_config = None
        self.explainer_manager = None

    def _init(self, explainer_manager: FrameworkExplainersManager):
        self.explainer_manager = explainer_manager
        return [self.explainer_manager.gen_dataset.dataset.num_node_features,
                self.explainer_manager.gen_dataset.is_multi(),
                self.explainer_manager.explainer.name]

    def _finalize(self):
        # if 1:  # TODO better check
        #     return False
        raise NotImplementedError

        # self.explainer_run_config = ExplainerRunConfig(**self._config)  # FIXME add class_name
        import copy
        config = copy.deepcopy(self._config)
        config['_config_kwargs']['kwargs']["_import_path"] = EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH
        config['_config_kwargs']['kwargs']["_config_class"] = "Config"
        self.explainer_run_config = ConfigPattern(
            **config,
            _config_class="ExplainerRunConfig"
        )
        return True

    def _submit(self):
        raise NotImplementedError
        # # NOTE: multiprocess = multiprocessing + dill instead of pickle, so it can serialize our objects and
        # # send them via a Queue
        # # NOTE 2: should be imported separately from torch.multiprocessing
        # from multiprocess import Process as mpProcess, Queue as mpQueue

        # queue = mpQueue()
        # self.explainer_subprocess = mpProcess(
        #     target=run_function, args=(
        #         self.explainer_manager, 'conduct_experiment',
        #         self.explainer_run_config, queue))
        # self.explainer_subprocess.start()
        # self.socket.send("explainer", {"status": "STARTED", "mode": self.explainer_run_config["mode"]})
        # self.explainer_subprocess.join()
        #
        # # Get result if present - otherwise nothing changed
        # if not queue.empty():
        #     self.explainer_manager = queue.get_nowait()

        self.socket.send(block="er", msg=
        {"status": "STARTED", "mode": self.explainer_run_config.mode})
        self.explainer_manager.conduct_experiment(self.explainer_run_config, socket=self.socket)

    def do(self, do, params):
        if do == "run":
            import copy
            config = json_loads(params.get('explainerRunConfig'))
            config['_config_kwargs']['kwargs']["_import_path"] =\
                EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH \
                    if config['_config_kwargs']['mode'] == 'local' \
                    else EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH
            config['_config_kwargs']['kwargs']["_config_class"] = "Config"
            self.explainer_run_config = ConfigPattern(
                **config,
                _config_class="ExplainerRunConfig"
            )

            print(f"explainer_run_config: {self.explainer_run_config.to_json()}")
            self._run_explainer()
            return ''

        elif do == "stop":
            # BACK_FRONT.model_manager.stop_signal = True  # TODO misha remove stop_signal
            self._stop_explainer()
            return ''

        elif do == "save":
            return self._save_explainer()

    def _run_explainer(self):
        # self.explainer_run_config = explainer_run_config

        # # NOTE: multiprocess = multiprocessing + dill instead of pickle, so it can serialize our objects and
        # # send them via a Queue
        # # NOTE 2: should be imported separately from torch.multiprocessing
        # from multiprocess import Process as mpProcess, Queue as mpQueue
        #
        # # queue = mpQueue()
        # # self.explainer_subprocess = mpProcess(
        # #     target=run_function, args=(
        # #         self.explainer_manager, 'conduct_experiment',
        # #         self.explainer_run_config, queue))
        # # self.explainer_subprocess.start()
        # # self.socket.send("explainer", {"status": "STARTED", "mode": self.explainer_run_config["mode"]})
        # # self.explainer_subprocess.join()
        # #
        # # # Get result if present - otherwise nothing changed
        # # if not queue.empty():
        # #     self.explainer_manager = queue.get_nowait()

        self.socket.send("explainer", {
            "status": "STARTED", "mode": self.explainer_run_config.mode})
        self.explainer_manager.conduct_experiment(self.explainer_run_config, socket=self.socket)

    def _stop_explainer(self):
        raise NotImplementedError
        # FIXME not implemented
        print('stop explainer')
        if self.explainer_subprocess and self.explainer_subprocess.is_alive():
            self.explainer_subprocess.terminate()
            self.socket.send("er", {"status": "INTERRUPTED", "mode": mode})
            self.explanation_data = None

    def _save_explainer(self):
        # self.explainer.save_explanation() TODO is it necessary?
        return str(self.explainer_manager.explainer_result_file_path)
