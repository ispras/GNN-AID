import importlib.util
import json
import random
from math import ceil
from types import FunctionType
import numpy as np
import torch
import sklearn.metrics
from torch.nn.utils import clip_grad_norm
from torch import tensor
import torch.nn.functional as F
from torch.cuda import is_available
from torch_geometric.data import DataLoader
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader

from aux.configs import ModelManagerConfig, ModelModificationConfig, ModelConfig, CONFIG_CLASS_NAME
from aux.data_info import UserCodeInfo
from aux.utils import import_by_name, all_subclasses, FRAMEWORK_PARAMETERS_PATH, model_managers_info_by_names_list, hash_data_sha256, \
    TECHNICAL_PARAMETER_KEY, IMPORT_INFO_KEY, OPTIMIZERS_PARAMETERS_PATH, FUNCTIONS_PARAMETERS_PATH
from aux.declaration import Declare
from explainers.explainer import ProgressBar
from explainers.ProtGNN.MCTS import mcts_args
from attacks.evasion_attacks import EvasionAttacker
from attacks.mi_attacks import MIAttacker
from attacks.poison_attacks import PoisonAttacker
from aux.configs import ConfigPattern, PoisonAttackConfig, CONFIG_OBJ, EvasionAttackConfig, MIAttackConfig, \
    PoisonDefenseConfig, EvasionDefenseConfig, MIDefenseConfig
from aux.utils import POISON_ATTACK_PARAMETERS_PATH, EVASION_ATTACK_PARAMETERS_PATH, MI_ATTACK_PARAMETERS_PATH, \
    POISON_DEFENSE_PARAMETERS_PATH, EVASION_DEFENSE_PARAMETERS_PATH, MI_DEFENSE_PARAMETERS_PATH
from defense.evasion_defense import EvasionDefender
from defense.mi_defense import MIDefender
from defense.poison_defense import PoisonDefender


class Metric:
    available_metrics = {
        'Accuracy': sklearn.metrics.accuracy_score,
        'F1': sklearn.metrics.f1_score,
        'BalancedAccuracy': sklearn.metrics.balanced_accuracy_score,
        'Recall': sklearn.metrics.recall_score,
        'Precision': sklearn.metrics.precision_score,
        'Jaccard': sklearn.metrics.jaccard_score,
    }

    @staticmethod
    def add_custom(name, compute_function):
        """
        Register a custom metric.
        Example for accuracy:

        >>> Metric.add_custom('accuracy', lambda y_true, y_pred, normalize=False:
        >>>     int((y_true == y_pred).sum()) / (len(y_true) if normalize else 1))

        :param name: name to refer to this metric
        :param compute_function: function which computes metric result:
         f(y_true, y_pred, **kwargs) -> value
        """
        if name in Metric.available_metrics:
            raise NameError(f"Metric '{name}' already registered, use another name")
        Metric.available_metrics[name] = compute_function

    def __init__(self, name, mask, **kwargs):
        """
        :param name: name to refer to this metric
        :param mask: 'train', 'val', 'test', or a bool valued list
        :param kwargs: params used in compute function
        """
        self.name = name
        self.mask = mask
        self.kwargs = kwargs

    def compute(self, y_true, y_pred):
        if self.name in Metric.available_metrics:
            return Metric.available_metrics[self.name](y_true, y_pred, **self.kwargs)

        raise NotImplementedError()

    @staticmethod
    def create_mask_by_target_list(y_true, target_list=None):
        if target_list is None:
            mask = [True] * len(y_true)
        else:
            mask = [False] * len(y_true)
        for i in target_list:
            if 0 <= i < len(mask):
                mask[i] = True
        return tensor(mask)
        # return mask


class GNNModelManager:
    """ class of basic functions over models:
    training, evaluation, save and load principle
    """

    def __init__(self,
                 manager_config=None,
                 modification: ModelModificationConfig = None):
        """
        :param manager_config: socket to use for sending data to frontend
        :param modification: socket to use for sending data to frontend
        """
        if manager_config is None:
            # raise RuntimeError("model manager config must be specified")
            manager_config = ConfigPattern(
                _config_class="ModelManagerConfig",
                _config_kwargs={},
            )
        elif isinstance(manager_config, ModelManagerConfig):
            manager_config = ConfigPattern(
                _config_class="ModelManagerConfig",
                _config_kwargs=manager_config.to_saveable_dict(),
            )
        # TODO Kirill, write raise Exception
        # else:
        #     raise Exception()

        # if modification is None:
        #     modification = ModelModificationConfig()

        if modification is None:
            # raise RuntimeError("model manager config must be specified")
            modification = ConfigPattern(
                _config_class="ModelModificationConfig",
                _config_kwargs={},
            )
        elif isinstance(modification, ModelModificationConfig):
            modification = ConfigPattern(
                _config_class="ModelModificationConfig",
                _config_kwargs=modification.to_dict(),
            )
        # TODO Kirill, write raise Exception
        # else:
        #     raise Exception()

        self.manager_config = manager_config
        self.modification = modification

        # QUE Kirill do we need to store it? maybe pass when need to
        self.dataset_path = None

        # FIXME Kirill, remove self.gen_dataset
        self.gen_dataset = None

        self.mi_defender = None
        self.mi_defense_name = None
        self.mi_defense_config = None
        self.evasion_defender = None
        self.evasion_defense_name = None
        self.evasion_defense_config = None
        self.poison_defense_name = None
        self.poison_defense_config = None
        self.poison_defender = None
        self.mi_attack_config = None
        self.mi_attacker = None
        self.mi_attack_name = None
        self.evasion_attack_config = None
        self.evasion_attacker = None
        self.evasion_attack_name = None
        self.poison_attack_name = None
        self.poison_attacker = None
        self.poison_attack_config = None

        self.poison_attack_flag = False
        self.evasion_attack_flag = False
        self.mi_attack_flag = False
        self.poison_defense_flag = False
        self.evasion_defense_flag = False
        self.mi_defense_flag = False

        self.gnn = None
        # We do not want store socket because it is not picklable for a subprocess
        self.socket = None
        self.stop_signal = False
        self.stats_data = None  # Stores some stats to be sent to frontend

        self.set_poison_defender()
        self.set_poison_attacker()
        self.set_mi_attacker()
        self.set_mi_defender()
        self.set_evasion_attacker()
        self.set_evasion_defender()

    def train_model(self, **kwargs):
        pass

    def train_1_step(self, gen_dataset):
        """ Perform 1 step of model training.
        """
        # raise NotImplementedError()
        pass

    def train_complete(self, gen_dataset, steps=None, **kwargs):
        """
        """
        # raise NotImplementedError()
        pass

    def train_on_batch(self, batch, **kwargs):
        pass

    def evaluate_model(self, **kwargs):
        pass

    def get_name(self):
        manager_name = self.manager_config.to_saveable_dict()
        # FIXME Kirill, make ModelManagerConfig and remove manager_name[CONFIG_CLASS_NAME]
        manager_name[CONFIG_CLASS_NAME] = self.__class__.__name__
        # for key, value in kwargs.items():
        #     manager_name[key] = value
        # manager_name = dict(sorted(manager_name.items()))
        json_str = json.dumps(manager_name, indent=2)
        return json_str

    def load_model(self, path=None, **kwargs):
        """
        Load model from torch save format
        """
        raise NotImplementedError()

    def save_model(self, path=None):
        """
        Save the model in torch format

        :param path: path to save the model. By default,
         the path is compiled based on the global class variables
        """
        raise NotImplementedError()

    def model_path_info(self):
        path, _ = Declare.models_path(self)
        return path

    def load_model_executor(self, path=None, **kwargs):
        """
        Load executor. Generates the download model path if no other path is specified.

        :param path: path to load the model. By default, the path is compiled based on the global
         class variables
        """

        if path is None:
            gnn_mm_name_hash = self.get_hash()
            model_dir_path, files_paths = Declare.declare_model_by_config(
                dataset_path=self.dataset_path,
                GNNModelManager_hash=gnn_mm_name_hash,
                model_ver_ind=kwargs.get('model_ver_ind') if 'model_ver_ind' in kwargs else
                self.modification.model_ver_ind,
                epochs=self.modification.epochs,
                gnn_name=self.gnn.get_hash()
            )
            path = model_dir_path / 'model'
        else:
            model_dir_path = path

        # TODO Kirill, check default parameters in gnn
        self.load_model(path=path, **kwargs)
        self.gnn.eval()
        return model_dir_path

    def get_hash(self):
        """
        calculates the hash on behalf of the model manager required for storage. The sha256
        algorithm is used.
        """
        gnn_MM_name = self.get_name()
        json_object = json.dumps(gnn_MM_name)
        gnn_MM_name_hash = hash_data_sha256(json_object.encode('utf-8'))
        return gnn_MM_name_hash

    def save_model_executor(self, path=None, files_paths=None):
        """
        Save executor, generates paths and prepares all information about the model
        and its parameters for saving

        :param gnn_architecture_path: path to save the architecture of the model,
         by default it forms the path itself.
        :param path: path to save the model. By default,
         the path is compiled based on the global class variables
        """
        if path is None:
            dir_path, files_paths = Declare.models_path(self)
            dir_path.mkdir(exist_ok=True, parents=True)
            path = dir_path / 'model'
        gnn_name_file = files_paths[0]
        gnn_mm_kwargs_file = files_paths[1]
        poison_attack_kwargs_file = files_paths[2]
        poison_defense_kwargs_file = files_paths[3]
        mi_defense_kwargs_file = files_paths[4]
        evasion_defense_kwargs_file = files_paths[5]
        evasion_attack_kwargs_file = files_paths[6]
        mi_attack_kwargs_file = files_paths[7]
        self.save_model(path)

        with open(gnn_name_file, "w") as f:
            f.write(self.gnn.get_name(obj_name_flag=True))
        with open(gnn_mm_kwargs_file, "w") as f:
            f.write(self.get_name())
        with open(poison_attack_kwargs_file, "w") as f:
            f.write(self.poison_attack_config.json_for_config())
        with open(poison_defense_kwargs_file, "w") as f:
            f.write(self.poison_defense_config.json_for_config())
        with open(mi_defense_kwargs_file, "w") as f:
            f.write(self.mi_defense_config.json_for_config())
        with open(evasion_defense_kwargs_file, "w") as f:
            f.write(self.evasion_defense_config.json_for_config())
        with open(evasion_attack_kwargs_file, "w") as f:
            f.write(self.evasion_attack_config.json_for_config())
        with open(mi_attack_kwargs_file, "w") as f:
            f.write(self.mi_attack_config.json_for_config())
        return path.parent

    def set_poison_attacker(self, poison_attack_config=None, poison_attack_name: str = None):
        if poison_attack_config is None:
            poison_attack_config = ConfigPattern(
                _class_name=poison_attack_name or "EmptyPoisonAttacker",
                _import_path=POISON_ATTACK_PARAMETERS_PATH,
                _config_class="PoisonAttackConfig",
                _config_kwargs={}
            )
        elif isinstance(poison_attack_config, PoisonAttackConfig):
            if poison_attack_name is None:
                raise Exception("if poison_attack_config is None, poison_attack_name must be defined")
            poison_attack_config = ConfigPattern(
                _class_name=poison_attack_name,
                _import_path=POISON_ATTACK_PARAMETERS_PATH,
                _config_class="PoisonAttackConfig",
                _config_kwargs=poison_attack_config.to_saveable_dict(),
            )
        self.poison_attack_config = poison_attack_config
        if poison_attack_name is None:
            poison_attack_name = self.poison_attack_config._class_name
        elif poison_attack_name != self.poison_attack_config._class_name:
            raise Exception(f"poison_attack_name and self.poison_attack_config._class_name should be equal, "
                            f"but now poison_attack_name is {poison_attack_name}, "
                            f"self.poisontorch.optim_attack_config._class_name is {self.poison_attack_config._class_name}")
        self.poison_attack_name = poison_attack_name
        poison_attack_kwargs = getattr(self.poison_attack_config, CONFIG_OBJ).to_dict()

        # name_klass = {e.name: e for e in PoisonAttacker.__subclasses__()}
        name_klass = {e.name: e for e in all_subclasses(PoisonAttacker)}

        klass = name_klass[self.poison_attack_name]
        self.poison_attacker = klass(
            # device=self.device,
            # device=device("cpu"),
            **poison_attack_kwargs
        )
        self.poison_attack_flag = True

    def set_evasion_attacker(self, evasion_attack_config=None, evasion_attack_name: str = None):
        if evasion_attack_config is None:
            evasion_attack_config = ConfigPattern(
                _class_name=evasion_attack_name or "EmptyEvasionAttacker",
                _import_path=EVASION_ATTACK_PARAMETERS_PATH,
                _config_class="EvasionAttackConfig",
                _config_kwargs={}
            )
        elif isinstance(evasion_attack_config, EvasionAttackConfig):
            if evasion_attack_name is None:
                raise Exception("if evasion_attack_config is None, evasion_attack_name must be defined")
            evasion_attack_config = ConfigPattern(
                _class_name=evasion_attack_name,
                _import_path=EVASION_ATTACK_PARAMETERS_PATH,
                _config_class="EvasionAttackConfig",
                _config_kwargs=evasion_attack_config.to_saveable_dict(),
            )
        self.evasion_attack_config = evasion_attack_config
        if evasion_attack_name is None:
            evasion_attack_name = self.evasion_attack_config._class_name
        elif evasion_attack_name != self.evasion_attack_config._class_name:
            raise Exception(f"evasion_attack_name and self.evasion_attack_config._class_name should be equal, "
                            f"but now evasion_attack_name is {evasion_attack_name}, "
                            f"self.evasion_attack_config._class_name is {self.evasion_attack_config._class_name}")
        self.evasion_attack_name = evasion_attack_name
        evasion_attack_kwargs = getattr(self.evasion_attack_config, CONFIG_OBJ).to_dict()

        name_klass = {e.name: e for e in EvasionAttacker.__subclasses__()}
        klass = name_klass[self.evasion_attack_name]
        self.evasion_attacker = klass(
            # device=self.device,
            # device=device("cpu"),
            **evasion_attack_kwargs
        )
        self.evasion_attack_flag = True

    def set_mi_attacker(self, mi_attack_config=None, mi_attack_name: str = None):
        if mi_attack_config is None:
            mi_attack_config = ConfigPattern(
                _class_name=mi_attack_name or "EmptyMIAttacker",
                _import_path=MI_ATTACK_PARAMETERS_PATH,
                _config_class="MIAttackConfig",
                _config_kwargs={}
            )
        elif isinstance(mi_attack_config, MIAttackConfig):
            if mi_attack_name is None:
                raise Exception("if mi_attack_config is None, mi_attack_name must be defined")
            mi_attack_config = ConfigPattern(
                _class_name=mi_attack_name,
                _import_path=MI_ATTACK_PARAMETERS_PATH,
                _config_class="MIAttackConfig",
                _config_kwargs=mi_attack_config.to_saveable_dict(),
            )
        self.mi_attack_config = mi_attack_config
        if mi_attack_name is None:
            mi_attack_name = self.mi_attack_config._class_name
        elif mi_attack_name != self.mi_attack_config._class_name:
            raise Exception(f"mi_attack_name and self.mi_attack_config._class_name should be equal, "
                            f"but now mi_attack_name is {mi_attack_name}, "
                            f"self.mi_attack_config._class_name is {self.mi_attack_config._class_name}")
        self.mi_attack_name = mi_attack_name
        mi_attack_kwargs = getattr(self.mi_attack_config, CONFIG_OBJ).to_dict()

        name_klass = {e.name: e for e in MIAttacker.__subclasses__()}
        klass = name_klass[self.mi_attack_name]
        self.mi_attacker = klass(
            # device=self.device,
            # device=device("cpu"),
            **mi_attack_kwargs
        )
        self.mi_attack_flag = True

    def set_poison_defender(self, poison_defense_config=None, poison_defense_name: str = None):
        if poison_defense_config is None:
            poison_defense_config = ConfigPattern(
                _class_name=poison_defense_name or "EmptyPoisonDefender",
                _import_path=POISON_DEFENSE_PARAMETERS_PATH,
                _config_class="PoisonDefenseConfig",
                _config_kwargs={}
            )
        elif isinstance(poison_defense_config, PoisonDefenseConfig):
            if poison_defense_name is None:
                raise Exception("if poison_defense_config is None, poison_defense_name must be defined")
            poison_defense_config = ConfigPattern(
                _class_name=poison_defense_name,
                _import_path=POISON_DEFENSE_PARAMETERS_PATH,
                _config_class="PoisonDefenseConfig",
                _config_kwargs=poison_defense_config.to_saveable_dict(),
            )
        self.poison_defense_config = poison_defense_config
        if poison_defense_name is None:
            poison_defense_name = self.poison_defense_config._class_name
        elif poison_defense_name != self.poison_defense_config._class_name:
            raise Exception(f"poison_defense_name and self.poison_defense_config._class_name should be equal, "
                            f"but now poison_defense_name is {poison_defense_name}, "
                            f"self.poison_defense_config._class_name is {self.poison_defense_config._class_name}")
        self.poison_defense_name = poison_defense_name
        poison_defense_kwargs = getattr(self.poison_defense_config, CONFIG_OBJ).to_dict()

        name_klass = {e.name: e for e in all_subclasses(PoisonDefender)}
        klass = name_klass[self.poison_defense_name]
        self.poison_defender = klass(
            # device=self.device,
            # device=device("cpu"),
            **poison_defense_kwargs
        )
        self.poison_defense_flag = True

    def set_evasion_defender(self, evasion_defense_config=None, evasion_defense_name: str = None):
        if evasion_defense_config is None:
            evasion_defense_config = ConfigPattern(
                _class_name=evasion_defense_name or "EmptyEvasionDefender",
                _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
                _config_class="EvasionDefenseConfig",
                _config_kwargs={}
            )
        elif isinstance(evasion_defense_config, EvasionDefenseConfig):
            if evasion_defense_name is None:
                raise Exception("if evasion_defense_config is None, evasion_defense_name must be defined")
            evasion_defense_config = ConfigPattern(
                _class_name=evasion_defense_name,
                _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
                _config_class="EvasionDefenseConfig",
                _config_kwargs=evasion_defense_config.to_saveable_dict(),
            )
        self.evasion_defense_config = evasion_defense_config
        if evasion_defense_name is None:
            evasion_defense_name = self.evasion_defense_config._class_name
        elif evasion_defense_name != self.evasion_defense_config._class_name:
            raise Exception(f"evasion_defense_name and self.evasion_defense_config._class_name should be equal, "
                            f"but now evasion_defense_name is {evasion_defense_name}, "
                            f"self.evasion_defense_config._class_name is {self.evasion_defense_config._class_name}")
        self.evasion_defense_name = evasion_defense_name
        evasion_defense_kwargs = getattr(self.evasion_defense_config, CONFIG_OBJ).to_dict()

        name_klass = {e.name: e for e in EvasionDefender.__subclasses__()}
        klass = name_klass[self.evasion_defense_name]
        self.evasion_defender = klass(
            # device=self.device,
            # device=device("cpu"),
            **evasion_defense_kwargs
        )
        self.evasion_defense_flag = True

    def set_mi_defender(self, mi_defense_config=None, mi_defense_name: str = None):
        """

        """
        if mi_defense_config is None:
            mi_defense_config = ConfigPattern(
                _class_name=mi_defense_name or "EmptyMIDefender",
                _import_path=MI_DEFENSE_PARAMETERS_PATH,
                _config_class="MIDefenseConfig",
                _config_kwargs={}
            )
        elif isinstance(mi_defense_config, MIDefenseConfig):
            if mi_defense_name is None:
                raise Exception("if mi_defense_config is None, mi_defense_name must be defined")
            mi_defense_config = ConfigPattern(
                _class_name=mi_defense_name,
                _import_path=MI_DEFENSE_PARAMETERS_PATH,
                _config_class="MIDefenseConfig",
                _config_kwargs=mi_defense_config.to_saveable_dict(),
            )
        self.mi_defense_config = mi_defense_config
        if mi_defense_name is None:
            mi_defense_name = self.mi_defense_config._class_name
        elif mi_defense_name != self.mi_defense_config._class_name:
            raise Exception(f"mi_defense_name and self.mi_defense_config._class_name should be equal, "
                            f"but now mi_defense_name is {mi_defense_name}, "
                            f"self.mi_defense_config._class_name is {self.mi_defense_config._class_name}")
        self.mi_defense_name = mi_defense_name
        mi_defense_kwargs = getattr(self.mi_defense_config, CONFIG_OBJ).to_dict()

        name_klass = {e.name: e for e in MIDefender.__subclasses__()}
        klass = name_klass[self.mi_defense_name]
        self.mi_defender = klass(
            # device=self.device,
            # device=device("cpu"),
            **mi_defense_kwargs
        )
        self.mi_defense_flag = True

    @staticmethod
    def available_attacker():
        pass

    @staticmethod
    def available_defender():
        pass

    @staticmethod
    def from_model_path(model_path, dataset_path, **kwargs):
        """
        Use information about model and model manager take gnn model,
        create gnn model manager object and load weights to the save model
        :param model_path: dict with information how create path to the model
        :param dataset_path: path to the dataset
        :param kwargs:
        :return: gnn model manager object and path to the model directory
        """

        model_dir_path, files_paths = Declare.declare_model_by_config(
            dataset_path=dataset_path,
            GNNModelManager_hash=str(model_path['gnn_model_manager']),
            epochs=int(model_path['epochs']) if model_path['epochs'] != 'None' else None,
            model_ver_ind=int(model_path['model_ver_ind']),
            gnn_name=model_path['gnn'],
        )

        gnn_mm_file = files_paths[1]
        gnn_file = files_paths[0]

        gnn = GNNModelManager.take_gnn_obj(gnn_file=gnn_file)

        modification_config = ModelModificationConfig(
            epochs=int(model_path['epochs']) if model_path['epochs'] != 'None' else None,
            model_ver_ind=int(model_path['model_ver_ind']),
        )

        with open(gnn_mm_file) as f:
            params = json.load(f)
            class_name = params.pop(CONFIG_CLASS_NAME)
            # class_name =
            # manager_config = ModelManagerConfig(**params)
            manager_config = ConfigPattern(**params)
        with open(FRAMEWORK_PARAMETERS_PATH, 'r') as f:
            framework_model_managers_info = json.load(f)
        if class_name in framework_model_managers_info.keys():
            klass = import_by_name(class_name, ["models_builder.gnn_models"])
            gnn_model_manager_obj = klass(
                gnn=gnn,
                manager_config=manager_config,
                modification=modification_config,
                dataset_path=dataset_path, **kwargs)
        else:
            mm_info = model_managers_info_by_names_list({class_name})
            klass = import_by_name(class_name, [mm_info[class_name][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY]])
            gnn_model_manager_obj = klass(
                gnn=gnn,
                manager_config=manager_config,
                dataset_path=dataset_path, **kwargs)

        gnn_model_manager_obj.load_model_executor()

        return gnn_model_manager_obj, model_dir_path

    def get_full_info(self):
        """
        Get available info about model for frontend
        """
        result = {}
        if hasattr(self, 'manager_config'):
            result["manager"] = self.manager_config.to_saveable_dict()
        if hasattr(self, 'modification'):
            result["modification"] = self.modification.to_saveable_dict()
        if hasattr(self, 'epochs'):
            result["epochs"] = f"Epochs={self.epochs}"
        return result

    def get_model_data(self):
        """
        :return: dict with the available functions of the model manager by the 'functions' key.
        """
        model_data = {}

        # Functions
        def get_own_functions(cls):
            return [x for x, y in cls.__dict__.items()
                    if isinstance(y, (FunctionType, classmethod, staticmethod))]

        model_data["functions"] = get_own_functions(type(self))
        return model_data

    @staticmethod
    def take_gnn_obj(gnn_file):
        with open(gnn_file) as f:
            params = json.load(f)
            class_name = params.pop(CONFIG_CLASS_NAME)
            obj_name = params.pop("obj_name")
            gnn_config = ModelConfig(**params)
        user_models_obj_dict_info = UserCodeInfo.user_models_list_ref()
        if class_name == 'FrameworkGNNConstructor':
            from models_builder.gnn_constructor import FrameworkGNNConstructor
            gnn = FrameworkGNNConstructor(gnn_config)
        else:
            if class_name not in user_models_obj_dict_info.keys():
                raise Exception(f"User class {class_name} does not defined")
            else:
                if obj_name is None:
                    try:
                        spec = importlib.util.spec_from_file_location(
                            class_name, user_models_obj_dict_info[class_name]['import_path'])
                        foo = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(foo)
                        gnn_class = getattr(foo, class_name)
                        gnn = gnn_class(gnn_config)
                    except:
                        raise Exception(f"Can't import user class {class_name}")
                else:
                    gnn = UserCodeInfo.take_user_model_obj(user_models_obj_dict_info[class_name]['import_path'],
                                                           obj_name)
        return gnn

    def before_epoch(self, gen_dataset):
        """ This hook is called before training the next training epoch
        """
        pass

    def after_epoch(self, gen_dataset):
        """ This hook is called after training the next training epoch
        """
        pass

    def before_batch(self, batch):
        """ This hook is called before training the next training batch
        """
        pass

    def after_batch(self, batch):
        """ This hook is called after training the next training batch
        """
        pass


class FrameworkGNNModelManager(GNNModelManager):
    """
    GNN model control class. Have methods: train_model, save_model, load_model
    """
    additional_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "mask_features": [],
            "optimizer": {
                "_config_class": "Config",
                "_class_name": "Adam",
                "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                "_class_import_info": ["torch.optim"],
                "_config_kwargs": {},
            },
            # FUNCTIONS_PARAMETERS_PATH,
            "loss_function": {
                "_config_class": "Config",
                "_class_name": "NLLLoss",
                "_import_path": FUNCTIONS_PARAMETERS_PATH,
                "_class_import_info": ["torch.nn"],
                "_config_kwargs": {},
            },
        }
    )
    """
    Args not listed in meta-info (thus not shown at frontend) to be added at initialization.
    
    optimizer: optimizer class name and params, all supported classes are described in optimizers_parameters.json file
    batch: int, batch to train
    loss_function: loss_function class name and params, all supported classes are described 
    in functions_parameters.json file
    clip: float, clip to train. If not None call clip_grad_norm
    mask_features: list of the names of the features to be masked. For example, 
    to prevent leakage of the response during training.
    """

    def __init__(self, gnn=None,
                 dataset_path=None,
                 **kwargs
                 ):
        """
        :param gnn: graph neural network model based on the GNNConstructor class
        :param manager_config:
        :param modification:
        :param dataset_path: int, the number of epochs the model actually was trained
        :param epochs: int, the number of epochs the model actually was trained
        :param kwargs: kwargs for GNNModelManager
        """

        # TODO Kirill, add train_test_split in default parameters gnnMM
        super().__init__(**kwargs)
        # Fulfill absent fields from default configs
        with open(FRAMEWORK_PARAMETERS_PATH, 'r') as f:
            params = json.load(f)
            class_name = type(self).__name__
            if class_name in params:
                self.manager_config = ConfigPattern(
                    _config_class="ModelManagerConfig",
                    _config_kwargs={k: v[2] for k, v in params[class_name].items()},
                ).merge(self.manager_config)

        # Add fields from additional config
        self.manager_config = self.manager_config.merge(self.additional_config)

        self.stop_signal = False  # TODO misha do we need it?
        self.gnn = gnn

        if self.modification.epochs is None:
            self.modification.epochs = 0
        self.optimizer = None
        self.loss_function = None

        self.batch = getattr(self.manager_config, CONFIG_OBJ).batch
        self.clip = getattr(self.manager_config, CONFIG_OBJ).clip
        self.mask_features = getattr(self.manager_config, CONFIG_OBJ).mask_features

        self.dataset_path = dataset_path

        if self.gnn is not None:
            self.init()

    def init(self):
        """
        Initialize optimizer and loss function.
        """
        if self.gnn is None:
            raise Exception("FrameworkGNNModelManager need GNN, now GNN is None")

        # QUE Kirill, can we make this better
        if "optimizer" in getattr(self.manager_config, CONFIG_OBJ):
            self.optimizer = getattr(self.manager_config, CONFIG_OBJ).optimizer.create_obj(params=self.gnn.parameters())
            # self.optimizer = getattr(self.manager_config, CONFIG_OBJ).optimizer.create_obj()

        if "loss_function" in getattr(self.manager_config, CONFIG_OBJ):
            self.loss_function = getattr(self.manager_config, CONFIG_OBJ).loss_function.create_obj()

    def train_complete(self, gen_dataset, steps=None, pbar=None, metrics=None, **kwargs):
        for _ in range(steps):
            self.before_epoch(gen_dataset)
            print("epoch", self.modification.epochs)
            train_loss = self.train_1_step(gen_dataset)
            self.after_epoch(gen_dataset)
            early_stopping_flag = self.early_stopping(train_loss=train_loss, gen_dataset=gen_dataset,
                                                      metrics=metrics, steps=steps)
            if self.socket:
                self.report_results(train_loss=train_loss, gen_dataset=gen_dataset,
                                    metrics=metrics)
            pbar.update(1)
            if early_stopping_flag:
                break

    def early_stopping(self, train_loss, gen_dataset, metrics, steps):
        return False

    def train_1_step(self, gen_dataset):
        task_type = gen_dataset.domain()
        if task_type == "single-graph":
            # FIXME Kirill, add data_x_copy mask
            loader = NeighborLoader(gen_dataset.dataset._data,
                                    num_neighbors=[-1], input_nodes=gen_dataset.train_mask,
                                    batch_size=self.batch, shuffle=True)
        elif task_type == "multiple-graphs":
            train_dataset = gen_dataset.dataset.index_select(gen_dataset.train_mask)
            loader = DataLoader(train_dataset, batch_size=self.batch, shuffle=True)
        # TODO Kirill, remove False when release edge recommendation task
        elif task_type == "edge" and False:
            loader = LinkNeighborLoader(gen_dataset.dataset._data,
                                        num_neighbors=[-1], input_nodes=gen_dataset.train_mask,
                                        batch_size=self.batch, shuffle=True)
        else:
            raise ValueError("Unsupported task type")
        loss = 0
        for batch in loader:
            self.before_batch(batch)
            loss += self.train_on_batch(batch, task_type)
            self.after_batch(batch)
        print("loss %.8f" % loss)
        self.modification.epochs += 1
        self.gnn.eval()
        return loss.cpu().detach().numpy().tolist()

    def train_on_batch(self, batch, task_type=None):
        if self.mi_defender:
            self.mi_defender.pre_batch()
        if self.evasion_defender:
            self.evasion_defender.pre_batch(model_manager=self, batch=batch)
        loss = None
        if hasattr(batch, "edge_weight"):
            weight = batch.edge_weight
        else:
            weight = None
        if task_type == "single-graph":
            self.optimizer.zero_grad()
            logits = self.gnn(batch.x, batch.edge_index, weight)
            loss = self.loss_function(logits, batch.y)
            if self.clip is not None:
                clip_grad_norm(self.gnn.parameters(), self.clip)
            self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
        elif task_type == "multiple-graphs":
            self.optimizer.zero_grad()
            logits = self.gnn(batch.x, batch.edge_index, batch.batch, weight)
            loss = self.loss_function(logits, batch.y)
            # loss.backward()
            # self.optimizer.step()
        # TODO Kirill, remove False when release edge recommendation task
        elif task_type == "edge" and False:
            self.optimizer.zero_grad()
            edge_index = batch.edge_index
            pos_edge_index = edge_index[:, batch.y == 1]
            neg_edge_index = edge_index[:, batch.y == 0]

            pos_out = self.gnn(batch.x, pos_edge_index, weight)
            neg_out = self.gnn(batch.x, neg_edge_index, weight)

            pos_loss = self.loss_function(pos_out, torch.ones_like(pos_out))
            neg_loss = self.loss_function(neg_out, torch.zeros_like(neg_out))

            loss = pos_loss + neg_loss
            # loss.backward()
        else:
            raise ValueError("Unsupported task type")
        if self.mi_defender:
            self.mi_defender.post_batch()
        evasion_defender_dict = None
        if self.evasion_defender:
            evasion_defender_dict = self.evasion_defender.post_batch(
                model_manager=self, batch=batch, loss=loss,
            )
        if evasion_defender_dict and "loss" in evasion_defender_dict:
            loss = evasion_defender_dict["loss"]
        loss.backward()
        self.optimizer.step()
        return loss

    def get_name(self, **kwargs):
        json_str = super().get_name()
        return json_str

    def load_model(self, path=None, **kwargs):
        """
        Load model from torch save format

        :param path: path to load the model. By default, the path is compiled based on the global
         class variables
        """
        if not is_available():
            self.gnn.load_state_dict(torch.load(path, map_location=torch.device('cpu'), ))
            # self.gnn = torch.load(path, map_location=torch.device('cpu'))
        else:
            self.gnn.load_state_dict(torch.load(path, ))
            # self.gnn = torch.load(path)
        if self.optimizer is None:
            self.init()
        return self.gnn

    def save_model(self, path=None):
        """
        Save the model in torch format

        :param path: path to save the model. By default,
         the path is compiled based on the global class variables
        """
        torch.save(self.gnn.state_dict(), path)

    def report_results(self, train_loss, gen_dataset, metrics):
        metrics_values = self.evaluate_model(gen_dataset=gen_dataset, metrics=metrics)
        self.compute_stats_data(gen_dataset, predictions=True, logits=True)
        self.send_epoch_results(
            metrics_values=metrics_values,
            stats_data={k: gen_dataset.visible_part.filter(v)
                        for k, v in self.stats_data.items()},
            weights={"weights": self.gnn.get_weights()}, loss=train_loss)

    def train_model(self, gen_dataset, save_model_flag=True, mode=None, steps=None, metrics=None,
                    socket=None):
        """
        Convenient train method.

        :param gen_dataset: dataset in torch_geometric data format for train
        :param save_model_flag: if need save model after train. Default save_model_flag=True
        :param mode: '1 step' or 'full' or None (choose automatically)
        :param steps: train specific number of epochs, if None - all of them
        :param metrics: list of metrics to measure at each step or at the end of training
        :param socket: socket to use for sending data to frontend
        """
        if self.poison_attacker:
            loc = self.poison_attacker.attack(gen_dataset=gen_dataset)
            if loc is not None:
                gen_dataset = loc

        if self.poison_defender:
            loc = self.poison_defender.defense(gen_dataset=gen_dataset)
            if loc is not None:
                gen_dataset = loc
        self.socket = socket
        pbar = ProgressBar(self.socket, "mt")

        # Assure we call here from subclass
        assert issubclass(type(self), GNNModelManager)

        assert mode in ['1_step', 'full', None]
        # TODO Kirill what is this? Outdated?
        # has_complete = self.train_complete != super(type(self), self).train_complete
        # assert has_complete
        do_1_step = True

        try:
            if do_1_step:
                assert steps > 0
                pbar.total = self.modification.epochs + steps
                pbar.n = self.modification.epochs
                pbar.update(0)
                self.train_complete(gen_dataset=gen_dataset, steps=steps,
                                    pbar=pbar, metrics=metrics)
                pbar.close()
                self.send_data("mt", {"status": "OK"})

            else:
                raise Exception

            if save_model_flag:
                return self.save_model_executor()

        except Exception as e:
            self.send_data("mt", {"status": "FAILED"})
            raise e
        finally:
            self.socket = None

    def run_model(self, gen_dataset, mask='test', out='answers'):
        """
        Run the model on a part of dataset specified with a mask.

        :param gen_dataset: wrapper over the dataset, stores the dataset and all meta-information about the dataset
        :param mask: 'train', 'val', 'test', 'all' -- part of the dataset on which the output will be obtained
        :param out: 'answers', 'predictions', 'logits' -- what output format will be calculated,
         availability depends on which methods have been overridden in the parent class
        :return:
        """
        try:
            mask = {
                'train': gen_dataset.train_mask,
                'val': gen_dataset.val_mask,
                'test': gen_dataset.test_mask,
                'all': tensor([True] * len(gen_dataset.labels)),
            }[mask]
        except KeyError:
            assert isinstance(mask, torch.Tensor)

        run_func = {
            'answers': self.gnn.get_answer,
            'predictions': self.gnn.get_predictions,
            'logits': self.gnn.__call__,
        }[out]
        self.gnn.eval()
        with torch.no_grad():  # Turn off gradients computation
            if gen_dataset.is_multi():
                dataset = gen_dataset.dataset
                part_loader = DataLoader(
                    dataset.index_select(mask), batch_size=self.batch, shuffle=False)
                full_out = torch.Tensor()
                # y_true = torch.Tensor()
                if hasattr(self, 'optimizer'):
                    self.optimizer.zero_grad()
                for data in part_loader:
                    # logits_batch = self.gnn(data.x, data.edge_index, data.batch)
                    # pred_batch = logits_batch.argmax(dim=1)
                    out = run_func(data.x, data.edge_index, data.batch)
                    full_out = torch.cat((full_out, out))
                    # y_true = torch.cat((y_true, data.y))
            else:  # single-graph
                data = gen_dataset.dataset._data  # FIXME what if no data? use .get(0) ?
                ver_ind = [n for n, x in enumerate(mask) if x]
                mask_size = len(ver_ind)

                number_of_batches = ceil(mask_size / self.batch)
                # data_x_elem_len = data.x.size()[1]
                full_out = torch.Tensor()
                # features_mask_tensor = torch.full(size=data.x.size(), fill_value=True)

                for batch_ind in range(number_of_batches):
                    data_x_copy = torch.clone(data.x)
                    mask_copy = [False] * data.x.size()[0]

                    # features_mask_tensor_copy = torch.clone(features_mask_tensor)

                    for elem_ind in ver_ind[
                                    batch_ind * self.batch: (batch_ind + 1) * self.batch]:
                        if hasattr(self, 'mask_features'):
                            for feature in self.mask_features:
                                # features_mask_tensor_copy[elem_ind][gen_dataset.info.node_attr_slices[feature][0]:
                                #                                     gen_dataset.info.node_attr_slices[feature][1]] = False
                                data_x_copy[elem_ind][gen_dataset.info.node_attr_slices[feature][0]:
                                                      gen_dataset.info.node_attr_slices[feature][1]] = 0
                        # if self.gnn_mm.train_mask_flag:
                        #     data_x_copy[elem_ind] = torch.zeros(data_x_elem_len)
                        # y_true = torch.masked.masked_tensor(data.y, mask_tensor)
                        mask_copy[elem_ind] = True

                    # mask_x_tensot = torch.masked.masked_tensor(data.x, features_mask_tensor_copy)

                    # FIXME Kirill what to do if no optimizer, train_mask_flag, batch?
                    if hasattr(self, 'optimizer'):
                        self.optimizer.zero_grad()
                    # logits_batch = self.gnn(data_x_copy, data.edge_index)
                    # pred_batch = logits_batch.argmax(dim=1)
                    out = run_func(data_x_copy, data.edge_index)
                    full_out = torch.cat((full_out, out[mask_copy]))
                    # y_true = torch.cat((y_true, data.y[mask_copy]))

        return full_out

    def evaluate_model(self, gen_dataset, metrics):
        """
        Compute metrics for a model result on a part of dataset specified by the metric mask.

        :param gen_dataset: wrapper over the dataset, stores the dataset and all meta-information about the dataset
        :param metrics: list of metrics to compute. metric based on class Metric
        :return: dict {metric -> value}
        """
        mask_metrics = {}
        for metric in metrics:
            mask = metric.mask
            if mask not in mask_metrics:
                mask_metrics[mask] = []
            mask_metrics[mask].append(metric)

        metrics_values = {}
        for mask, ms in mask_metrics.items():
            try:
                mask_tensor = {
                    'train': gen_dataset.train_mask.tolist(),
                    'val': gen_dataset.val_mask.tolist(),
                    'test': gen_dataset.test_mask.tolist(),
                    'all': [True] * len(gen_dataset.labels),
                }[mask]
            except KeyError:
                assert isinstance(mask, torch.Tensor)
                mask_tensor = mask
            if self.evasion_attacker:
                self.evasion_attacker.attack(model_manager=self, gen_dataset=gen_dataset, mask_tensor=mask_tensor)
            metrics_values[mask] = {}
            y_pred = self.run_model(gen_dataset, mask=mask)
            y_true = gen_dataset.labels[mask_tensor]

            for metric in ms:
                metrics_values[mask][metric.name] = metric.compute(y_pred, y_true)
                # metrics_values[mask][metric.name] = MetricManager.compute(metric, y_pred, y_true)
        if self.mi_attacker:
            self.mi_attacker.attack()
        return metrics_values

    def compute_stats_data(self, gen_dataset, predictions=False, logits=False):
        """
        :param gen_dataset: wrapper over the dataset, stores the dataset
         and all meta-information about the dataset
        :param predictions: boolean flag that indicates the need to enter model predictions
         in the statistics for the front
        :param logits: boolean flag that indicates the need to enter model logits
         in the statistics for the front
        :return: dict with model weights. Also function can add in dict model predictions
         and logits
        """
        self.stats_data = {}

        # Stats: weights, logits, predictions
        if predictions:  # and hasattr(self.gnn, 'get_predictions'):
            predictions = self.run_model(gen_dataset, mask='all', out='predictions')
            self.stats_data["predictions"] = predictions.detach().cpu().tolist()
        if logits:  # and hasattr(self.gnn, 'forward'):
            logits = self.run_model(gen_dataset, mask='all', out='logits')
            self.stats_data["embeddings"] = logits.detach().cpu().tolist()

    def send_data(self, block, msg, tag='model', obligate=True, socket=None):
        """
        Send data to the frontend.

        :param socket:
        :param tag:
        :param block:
        :param msg: message as a json-convertible dict
        :param obligate: if you send a lot of updates of the same stuff, e.g. weights at each
         training step, set obligate=False to save traffic and actually send only the last one on
         the queue.
        :return: bool flag
        """
        socket = socket or self.socket
        if socket is None:
            return False
        socket.send(block=block, msg=msg, tag=tag, obligate=obligate)
        return True

    def send_epoch_results(self, metrics_values=None, stats_data=None, weights=None, loss=None, obligate=False,
                           socket=None):
        """
        Send updates to the frontend after a training epoch: epoch, metrics, logits, loss.

        :param weights:
        :param metrics_values: quality metrics (accuracy, F1)
        :param stats_data: model statistics (logits, predictions)
        :param loss: train loss
        """
        socket = socket or self.socket
        # Metrics values, epoch, loss
        if metrics_values:
            metrics_data = {"epochs": self.modification.epochs}
            if loss:
                metrics_data["loss"] = loss
            metrics_data["metrics_values"] = metrics_values
            self.send_data("mt", {"metrics": metrics_data}, tag='model_metrics', socket=socket)
        if weights:
            self.send_data("mt", weights, tag='model_weights', obligate=obligate, socket=socket)
        if stats_data:
            self.send_data("mt", stats_data, tag='model_stats', obligate=obligate, socket=socket)

    def load_train_test_split(self, gen_dataset):
        path = self.model_path_info()
        path = path / 'train_test_split'
        gen_dataset.train_mask, gen_dataset.val_mask, gen_dataset.test_mask, _ = torch.load(path)[:]
        return gen_dataset

class ProtGNNModelManager(FrameworkGNNModelManager):
    # additional_config = ModelManagerConfig(
    #     loss_function={CONFIG_CLASS_NAME: "CrossEntropyLoss"},
    #     mask_features=[],
    # )
    additional_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "mask_features": [],
            "optimizer": {
                "_config_class": "Config",
                "_class_name": "Adam",
                "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                "_class_import_info": ["torch.optim"],
                "_config_kwargs": {},
            },
            #FUNCTIONS_PARAMETERS_PATH,
            "loss_function": {
                "_config_class": "Config",
                "_class_name": "CrossEntropyLoss",
                "_import_path": FUNCTIONS_PARAMETERS_PATH,
                "_class_import_info": ["torch.nn"],
                "_config_kwargs": {},
            },
        }
    )

    def __init__(self, gnn=None, dataset_path=None, **kwargs):
        super().__init__(gnn=gnn, dataset_path=dataset_path, **kwargs)

        # Get prot layer and its params
        self.prot_layer = getattr(self.gnn, self.gnn.prot_layer_name)
        _config_obj = getattr(self.manager_config, CONFIG_OBJ)
        self.clst = _config_obj.clst
        self.sep = _config_obj.sep
        #lr = _config_obj.lr
        self.early_stopping_marker = _config_obj.early_stopping
        self.proj_epochs = _config_obj.proj_epochs
        self.warm_epoch = _config_obj.warm_epoch
        self.save_epoch = _config_obj.save_epoch
        self.save_thrsh = _config_obj.save_thrsh
        # TODO implement other MCTS args too
        # TODO MCTS args via static ?
        mcts_args.min_atoms = _config_obj.mcts_min_atoms
        mcts_args.max_atoms = _config_obj.mcts_max_atoms
        self.prot_thrsh = _config_obj.prot_thrsh
        self.early_stop_count = 0
        self.gnn.best_prots = self.prot_layer.prototype_graphs
        self.best_acc = 0.0

    def save_model(self, path=None):
        """
        Save the model in torch format

        :param path: path to save the model. By default,
         the path is compiled based on the global class variables
        """
        torch.save({"model_state_dict": self.gnn.state_dict(),
                    "best_prots": self.gnn.best_prots,
                    }, path)

    def load_model(self, path=None, **kwargs):
        """
        Load model from torch save format

        :param path: path to load the model. By default, the path is compiled based on the global
         class variables
        """
        if not is_available():
            checkpoint = torch.load(path, map_location=torch.device('cpu'), )
        else:
            checkpoint = torch.load(path)
        self.gnn.load_state_dict(checkpoint["model_state_dict"])
        self.gnn.best_prots = checkpoint["best_prots"]
        if self.optimizer is None:
            self.init()
        return self.gnn

    # TODO Kirill train_on_batch to be divided into two parts
    def train_on_batch(self, batch, task_type=None):
        if self.mi_defender:
            self.mi_defender.pre_batch()
        if self.evasion_defender:
            self.evasion_defender.pre_batch(model_manager=self, batch=batch)
        loss = None
        if task_type == "single-graph":
            # TODO Prot with single-graph
            self.optimizer.zero_grad()
            logits = self.gnn(batch.x, batch.edge_index)
            min_distances = self.gnn.min_distances

            # cluster loss
            self.prot_layer.prototype_class_identity = self.prot_layer.prototype_class_identity
            prototypes_of_correct_class = torch.t(
                self.prot_layer.prototype_class_identity[:, batch.y].bool())
            cluster_cost = torch.mean(
                torch.min(min_distances[prototypes_of_correct_class]
                          .reshape(-1, self.prot_layer.num_prototypes_per_class), dim=1)[0])

            # seperation loss
            separation_cost = -torch.mean(
                torch.min(min_distances[~prototypes_of_correct_class].reshape(-1, (
                        self.prot_layer.output_dim - 1) * self.prot_layer.num_prototypes_per_class),
                          dim=1)[0])

            # sparsity loss
            l1_mask = 1 - torch.t(self.prot_layer.prototype_class_identity)
            l1 = (self.prot_layer.last_layer.weight * l1_mask).norm(p=1)

            # diversity loss
            ld = 0
            # TODO expreriments required. With zero coeff - meaningless
            # for k in range(prot_layer.output_dim):
            #     p = prot_layer.prototype_vectors[
            #         k * prot_layer.num_prototypes_per_class:
            #         (k + 1) * prot_layer.num_prototypes_per_class]
            #     p = F.normalize(p, p=2, dim=1)
            #     matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]) - 0.3
            #     matrix2 = torch.zeros(matrix1.shape)
            #     ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))

            loss = self.loss_function(logits, batch.y)
            loss += self.clst * cluster_cost + self.sep * separation_cost + 5e-4 * l1 + 0.00 * ld
            if self.clip is not None:
                clip_grad_norm(self.gnn.parameters(), self.clip)
            self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
        elif task_type == "multiple-graphs":
            self.optimizer.zero_grad()
            logits = self.gnn(batch.x, batch.edge_index, batch.batch)
            loss = self.loss_function(logits, batch.y)
            # loss.backward()
            # self.optimizer.step()
        # TODO Kirill, remove False when release edge recommendation task
        elif task_type == "edge" and False:
            self.optimizer.zero_grad()
            edge_index = batch.edge_index
            pos_edge_index = edge_index[:, batch.y == 1]
            neg_edge_index = edge_index[:, batch.y == 0]

            pos_out = self.gnn(batch.x, pos_edge_index)
            neg_out = self.gnn(batch.x, neg_edge_index)

            pos_loss = self.loss_function(pos_out, torch.ones_like(pos_out))
            neg_loss = self.loss_function(neg_out, torch.zeros_like(neg_out))

            loss = pos_loss + neg_loss
            # loss.backward()
        else:
            raise ValueError("Unsupported task type")
        if self.mi_defender:
            self.mi_defender.post_batch()
        evasion_defender_dict = None
        if self.evasion_defender:
            evasion_defender_dict = self.evasion_defender.post_batch(
                model_manager=self, batch=batch, loss=loss,
            )
        if evasion_defender_dict and "loss" in evasion_defender_dict:
            loss = evasion_defender_dict["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.gnn.parameters(), clip_value=2.0)
        self.optimizer.step()
        return loss

    def before_epoch(self, gen_dataset):
        cur_step = self.modification.epochs
        train_ind = [n for n, x in enumerate(gen_dataset.train_mask) if x]
        # Prototype projection
        if cur_step > self.proj_epochs and cur_step % self.proj_epochs == 0:
            self.prot_layer.projection(self.gnn, gen_dataset.dataset, train_ind, gen_dataset.dataset.data, thrsh=self.prot_thrsh)
        self.gnn.train()
        for p in self.gnn.parameters():
            p.requires_grad = True
        self.prot_layer.prototype_vectors.requires_grad = True
        if cur_step < self.warm_epoch:
            for p in self.prot_layer.last_layer.parameters():
                p.requires_grad = False
        else:
            for p in self.prot_layer.last_layer.parameters():
                p.requires_grad = True

    def after_epoch(self, gen_dataset):
        # TODO compare is_best with different metrics to be implemented

        # check if best model
        metrics_values = self.evaluate_model(
            gen_dataset, metrics=[Metric("Accuracy", mask='val'),
                                  Metric("Precision", mask='val'),
                                  Metric("Recall", mask='val')])
        self.cur_acc = metrics_values['val']["Accuracy"]
        self.is_best = (self.cur_acc - self.best_acc >= 0.01)

        if self.is_best:
            self.best_acc = self.cur_acc
            self.early_stop_count = 0
            self.gnn.best_prots = self.prot_layer.prototype_graphs


    def early_stopping(self, train_loss, gen_dataset, metrics, steps):
        step = self.modification.epochs
        if self.is_best:
            self.early_stop_count = 0
        else:
            self.early_stop_count += 1
        last_projection = (step % self.proj_epochs == 0 and step + self.proj_epochs >= steps)

        return self.early_stop_count >= self.early_stopping_marker or last_projection