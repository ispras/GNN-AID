import json
from typing import Union, Type

from gnn_aid.aux.declaration import Declare
from gnn_aid.aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, all_subclasses, ProgressBar
from gnn_aid.data_structures.configs import ExplainerInitConfig, ExplainerModificationConfig, \
    ExplainerRunConfig
from gnn_aid.data_structures.gen_config import CONFIG_OBJ, ConfigPattern
from gnn_aid.datasets.gen_dataset import GeneralDataset
from gnn_aid.models_builder.model_managers import GNNModelManager
from .explainer import Explainer
from .explainer_metrics import NodesExplainerMetric


class FrameworkExplainersManager:
    """
    A class based on ExplainerManager for working with
    interpretation methods built into the framework
    Currently supports 6 explainers
    """
    supported_explainers = [e.name for e in all_subclasses(Explainer)]

    def __init__(
            self,
            dataset: GeneralDataset,
            gnn_manager: Type,
            init_config: Union[ConfigPattern, ExplainerInitConfig] = None,
            explainer_name: str = None,
            modification_config: Union[ConfigPattern, ExplainerModificationConfig] = None,
            device: str = None
    ):
        self.files_paths = None
        if device is None:
            device = "cpu"
        self.device = device
        if init_config is None:
            if explainer_name is None:
                raise Exception("if init_config is None, explainer_name must be defined")
            init_config = ConfigPattern(
                _class_name=explainer_name,
                _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
                _config_class="ExplainerInitConfig",
                _config_kwargs={}
            )
        elif isinstance(init_config, ExplainerInitConfig):
            if explainer_name is None:
                raise Exception("if init_config is None, explainer_name must be defined")
            init_config = ConfigPattern(
                _class_name=explainer_name,
                _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
                _config_class="ExplainerInitConfig",
                _config_kwargs=init_config.to_savable_dict(),
            )
        self.init_config = init_config
        if modification_config is None:
            modification_config = ConfigPattern(
                _config_class="ExplainerModificationConfig",
                _config_kwargs={}
            )
        elif isinstance(modification_config, ExplainerModificationConfig):
            modification_config = ConfigPattern(
                _config_class="ExplainerModificationConfig",
                _config_kwargs=modification_config.to_savable_dict(),
            )
        self.modification_config = modification_config

        self.save_explanation_flag = True
        self.explainer_result_file_path = None

        self.gen_dataset = dataset
        self.gnn = gnn_manager.gnn
        self.model_manager = gnn_manager
        self.gnn_model_path = gnn_manager.model_path_info()

        # init_kwargs = self.init_config.to_dict()
        init_kwargs = getattr(self.init_config, CONFIG_OBJ).to_dict()
        # self.explainer_name = init_kwargs.pop(CONFIG_CLASS_NAME)
        if explainer_name is None:
            explainer_name = self.init_config._class_name
        elif explainer_name != self.init_config._class_name:
            raise Exception(f"explainer_name and self.init_config._class_name should be eqequal, "
                            f"but now explainer_name is {explainer_name}, "
                            f"self.init_config._class_name is {self.init_config._class_name}")
        self.explainer_name = explainer_name

        if self.explainer_name not in FrameworkExplainersManager.supported_explainers:
            raise ValueError(
                f"Explainer {self.explainer_name} is not supported. Choose one of "
                f"{FrameworkExplainersManager.supported_explainers}")

        print("Creating explainer")
        name_klass = {e.name: e for e in all_subclasses(Explainer)}
        klass = name_klass[self.explainer_name]
        self.explainer = klass(
            self.gen_dataset, model=self.gnn,
            device=self.device,
            # device=device("cpu"),
            **init_kwargs
        )

        self.explanation = None
        self.explanation_data = None
        self.running = False

    def save_explanation(
            self,
            run_config: Union[ConfigPattern, ExplainerRunConfig]
    ) -> None:
        """ Save explanation to file.
        """
        self.explanation_result_path(run_config)
        self.explainer.save(self.explainer_result_file_path)
        print("Saved explanation")

    def load_explanation(
            self,
            run_config: Union[ConfigPattern, ExplainerRunConfig]
    ) -> dict:
        if self.modification_config.explainer_ver_ind is None:
            raise RuntimeError("explainer_ver_ind should not be None")
        self.explanation_result_path(run_config)
        try:
            print(self.explainer_result_file_path)
            with open(self.explainer_result_file_path, "r") as f:
                explanation = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"No explanation exists for given path: "
                                    f"{self.explainer_result_file_path}")
        return explanation

    def explanation_result_path(
            self,
            run_config: Union[ConfigPattern, ExplainerRunConfig],
            create_dir_flag: bool = True,
    ) -> None:
        # TODO pass configs
        self.explainer_result_file_path, self.files_paths = Declare.explanation_file_path(
            models_path=self.gnn_model_path,
            explainer_name=self.explainer_name,
            explainer_ver_ind=self.modification_config.explainer_ver_ind,
            explainer_init_kwargs=self.init_config.to_savable_dict(),
            explainer_run_kwargs=run_config.to_savable_dict(),
            create_dir_flag=create_dir_flag,
        )

    def conduct_experiment(
            self,
            run_config: Union[ConfigPattern, ExplainerRunConfig],
            pbar: ProgressBar = None
    ) -> dict:
        """
        Runs the full cycle of the interpretation experiment
        """
        self.explainer.pbar = pbar or ProgressBar()
        # mode = run_config.mode
        mode = getattr(run_config, CONFIG_OBJ).mode
        params = getattr(getattr(run_config, CONFIG_OBJ).kwargs, CONFIG_OBJ).to_dict()
        # params.pop(CONFIG_CLASS_NAME)

        print("Running explainer...")
        self.explainer.run(mode, params, finalize=True)
        print("Explanation ready")
        # self.explainer._finalize()
        result = self.explainer.explanation.dictionary
        # if self._after_run_hook:
        #     self._after_run_hook(result)

        # TODO what if save_explanation_flag=False?
        if self.save_explanation_flag:
            self.save_explanation(run_config)
            path = self.model_manager.save_model_executor()
            self.gen_dataset.save_train_test_mask(path)

        return result

    def conduct_experiment_by_dataset(
            self,
            run_config: Union[ConfigPattern, ExplainerRunConfig],
            dataset: GeneralDataset,
            save_explanation_flag=False
    ) -> dict:
        init_kwargs = getattr(self.init_config, CONFIG_OBJ).to_dict()
        if self.explainer_name not in FrameworkExplainersManager.supported_explainers:
            raise ValueError(
                f"Explainer {self.explainer_name} is not supported. Choose one of "
                f"{FrameworkExplainersManager.supported_explainers}")
        print("Creating explainer")
        name_klass = {e.name: e for e in all_subclasses(Explainer)}
        klass = name_klass[self.explainer_name]
        self.explainer = klass(
            dataset, model=self.gnn,
            device=self.device,
            # device=device("cpu"),
            **init_kwargs
        )
        old_save_explanation_flag = self.save_explanation_flag
        self.save_explanation_flag = save_explanation_flag
        result = self.conduct_experiment(run_config)
        self.save_explanation_flag = old_save_explanation_flag
        return result

    def evaluate_metrics(
            self,
            node_id_to_explainer_run_config: dict[int, ConfigPattern],
            explaining_metrics_params: Union[dict, None] = None,
    ) -> dict:
        """
        Evaluates explanation metrics between given node indices
        """
        # TODO misha do we want progress bar here?
        # self.explainer.pbar = ProgressBar(
        #     "er", desc=f'{self.explainer.name} explaining metrics calculation'
        # )  # progress bar
        print("Evaluating explanation metrics...")
        if self.gen_dataset.is_multi():
            raise NotImplementedError("Explanation metrics for graph classification")
        else:

            explanation_metrics_calculator = NodesExplainerMetric(
                self,
                explaining_metrics_params
            )
            result = explanation_metrics_calculator.evaluate(node_id_to_explainer_run_config)
        print("Explanation metrics are ready")

        # TODO what if save_explanation_flag=False?
        if self.save_explanation_flag:
            # self.save_explanation_metrics(run_config)
            self.model_manager.save_model_executor()

        return result

    @staticmethod
    def available_explainers(
            gen_dataset: GeneralDataset,
            model_manager: GNNModelManager
    ) -> list:
        """ Get a list of explainers applicable for current model and dataset.
        """
        return [
            e.name for e in all_subclasses(Explainer)
            if e.check_availability(gen_dataset, model_manager)
        ]
