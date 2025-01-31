import json
import os
from pathlib import Path
from typing import Union

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
from web_interface.back_front.utils import json_loads, get_config_keys


class ExplainerWBlock(WrapperBlock):
    def __init__(
            self,
            name: str,
            blocks: [Block],
            *args,
            **kwargs
    ):
        super().__init__(blocks, name, *args, **kwargs)

    def _init(
            self,
            gen_dataset: GeneralDataset,
            gmm: GNNModelManager
    ) -> None:
        self.gen_dataset = gen_dataset
        self.gmm = gmm

    def _finalize(
            self
    ) -> bool:
        return True

    def _submit(
            self
    ) -> None:
        pass


class ExplainerLoadBlock(Block):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.explainer_path = None
        self.info = None
        self.gen_dataset = None
        self.gmm = None

    def _init(
            self,
            gen_dataset: GeneralDataset,
            gmm: GNNModelManager
    ) -> list:
        # Define options for model manager
        self.gen_dataset = gen_dataset
        self.gmm = gmm
        return [gen_dataset.dataset.num_node_features, gen_dataset.is_multi(), self.get_index()]
        # return self.get_index()

    def _finalize(
            self
    ) -> bool:
        if set(get_config_keys("explanations")) != set(self._config.keys()):
            return False

        self.explainer_path = self._config
        return True

    def _submit(
            self
    ) -> None:
        init_config, run_config = self._explainer_kwargs(model_path=self.gmm.model_path_info(),
                                                         explainer_path=self.explainer_path)
        modification_config = ExplainerModificationConfig(
            explainer_ver_ind=self.explainer_path["explainer_ver_ind"],
        )

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

    def get_index(
            self
    ) -> [str, str]:
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

    def _explainer_kwargs(
            self,
            model_path: Union[str, Path],
            explainer_path: Union[str, Path]
    ):
        init_kwargs_file, run_kwargs_file = Declare.explainer_kwargs_path_full(
            model_path=model_path, explainer_path=explainer_path)
        with open(init_kwargs_file) as f:
            init_config = ConfigPattern(**json.load(f))
        with open(run_kwargs_file) as f:
            run_config = ConfigPattern(**json.load(f))
        return init_config, run_config


class ExplainerInitBlock(Block):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.explainer_init_config = None
        self.gen_dataset = None
        self.gmm = None

    def _init(
            self,
            gen_dataset: GeneralDataset,
            gmm: GNNModelManager
    ) -> list:
        # Define options for model manager
        self.gen_dataset = gen_dataset
        self.gmm = gmm
        return FrameworkExplainersManager.available_explainers(self.gen_dataset, self.gmm)

    def _finalize(
            self
    ) -> bool:
        self.explainer_init_config = ConfigPattern(
            **self._config,
            _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
            _config_class="ExplainerInitConfig")
        return True

    def _submit(
            self
    ) -> None:
        # Build an explainer
        self._object = FrameworkExplainersManager(
            dataset=self.gen_dataset, gnn_manager=self.gmm,
            init_config=self.explainer_init_config,
            explainer_name=self.explainer_init_config._class_name
        )
        self._result = {"config": self.explainer_init_config}


class ExplainerRunBlock(Block):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.explainer_run_config = None
        self.explainer_manager = None

    def _init(
            self,
            explainer_manager: FrameworkExplainersManager
    ) -> list:
        self.explainer_manager = explainer_manager
        return [self.explainer_manager.gen_dataset.dataset.num_node_features,
                self.explainer_manager.gen_dataset.is_multi(),
                self.explainer_manager.explainer.name]

    def do(
            self,
            do: str,
            params: dict
    ) -> str:
        if do == "run":
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

        # elif do == "save":
        #     return self._save_explainer()

    def _run_explainer(
            self
    ) -> None:
        self.socket.send("explainer", {
            "status": "STARTED", "mode": self.explainer_run_config.mode})
        # Saves explanation by default, save_explanation_flag=True
        self.explainer_manager.conduct_experiment(self.explainer_run_config, socket=self.socket)

    # def _save_explainer(self):
    #     # self.explainer_manager.save_explanation()
    #     return str(self.explainer_manager.explainer_result_file_path)
