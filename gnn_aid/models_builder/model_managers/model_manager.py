import importlib.util
import json
from pathlib import Path
from types import FunctionType
from typing import List, Union, Type, Callable, Tuple, Any

from gnn_aid.aux import Declare, UserCodeInfo
from gnn_aid.aux.utils import hash_data_sha256, POISON_ATTACK_PARAMETERS_PATH, all_subclasses, \
    EVASION_ATTACK_PARAMETERS_PATH, MI_ATTACK_PARAMETERS_PATH, POISON_DEFENSE_PARAMETERS_PATH, \
    EVASION_DEFENSE_PARAMETERS_PATH, MI_DEFENSE_PARAMETERS_PATH, FRAMEWORK_PARAMETERS_PATH, \
    import_by_name, model_managers_info_by_names_list, TECHNICAL_PARAMETER_KEY, IMPORT_INFO_KEY
from gnn_aid.data_structures import ModelManagerConfig, ModelModificationConfig, PoisonAttackConfig, \
    EvasionAttackConfig, MIAttackConfig, PoisonDefenseConfig, EvasionDefenseConfig, MIDefenseConfig, \
    ModelConfig, Task
from gnn_aid.data_structures.gen_config import CONFIG_OBJ, CONFIG_CLASS_NAME, ConfigPattern
from gnn_aid.datasets import GeneralDataset
from gnn_aid.models_builder import FrameworkGNNConstructor, GNNConstructor


class GNNModelManager:
    """
    Base class providing core model operations: training, evaluation, save, and load.

    Manages model state including attack/defense configurations, training hooks,
    and serialization to/from disk.
    """

    def __init__(
            self,
            manager_config: ModelManagerConfig = None,
            modification: ModelModificationConfig = None
    ):
        """
        Args:
            manager_config (ModelManagerConfig): Configuration for the model manager.
                Wrapped in a ConfigPattern if given as a plain ModelManagerConfig. Default value: `None`.
            modification (ModelModificationConfig): Configuration for model versioning and epochs.
                Wrapped in a ConfigPattern if given as a plain ModelModificationConfig. Default value: `None`.
        """
        if manager_config is None:
            manager_config = ConfigPattern(
                _config_class="ModelManagerConfig",
                _config_kwargs={},
            )
        elif isinstance(manager_config, ModelManagerConfig):
            manager_config = ConfigPattern(
                _config_class="ModelManagerConfig",
                _config_kwargs=manager_config.to_savable_dict(),
            )

        if modification is None:
            modification = ConfigPattern(
                _config_class="ModelModificationConfig",
                _config_kwargs={},
            )
        elif isinstance(modification, ModelModificationConfig):
            modification = ConfigPattern(
                _config_class="ModelModificationConfig",
                _config_kwargs=modification.to_dict(),
            )

        self.manager_config = manager_config
        self.modification = modification

        # QUE Kirill do we need to store it? maybe pass when need to
        self.dataset_path = None
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
        self.stats_data: dict = None  # Stores some stats to be sent to frontend

        self.set_poison_defender()
        self.set_poison_attacker()
        self.set_mi_attacker()
        self.set_mi_defender()
        self.set_evasion_attacker()
        self.set_evasion_defender()

        # Hooks for frontend client
        self._after_epoch_hook: Callable = None

    def set_hook(
            self,
            hook: Callable,
            where: str
    ) -> None:
        """ Register a callback hook for a specific training event.
        """
        if where == 'after_epoch':
            self._after_epoch_hook = hook
        else:
            raise ValueError(f"Hook {where} is not supported")

    def train_model(
            self,
            **kwargs
    ):
        pass

    def train_1_step(
            self,
            gen_dataset: GeneralDataset
    ):
        pass

    def train_complete(
            self,
            gen_dataset: GeneralDataset,
            steps: int = None,
            **kwargs
    ) -> None:
        pass

    def train_on_batch(
            self,
            batch,
            task_type: Task,
            **kwargs
    ):
        pass

    def evaluate_model(
            self,
            **kwargs
    ):
        pass

    def get_name(
            self
    ) -> str:
        """
        Returns:
            JSON string identifying this model manager by class name and config.
        """
        manager_name = self.manager_config.to_savable_dict()
        # FIXME Kirill, make ModelManagerConfig and remove manager_name[CONFIG_CLASS_NAME]
        manager_name[CONFIG_CLASS_NAME] = self.__class__.__name__
        json_str = json.dumps(manager_name, indent=2)
        return json_str

    def load_model(
            self,
            path: Union[str, Path] = None,
            **kwargs
    ) -> Type:
        raise NotImplementedError()

    def save_model(
            self,
            path: Union[str, Path] = None
    ) -> None:
        raise NotImplementedError()

    def model_path_info(
            self
    ) -> Union[str, Path]:
        path, _ = Declare.models_path(self)
        return path

    def load_model_executor(
            self,
            path: Union[str, Path, None] = None,
            **kwargs
    ) -> Union[str, Path]:
        """
        Load the model, generating the path automatically if none is provided.

        Args:
            path (Union[str, Path, None]): Path to load the model from. If None, the path is
                derived from the global config variables. Default value: `None`.
            **kwargs: Additional arguments, e.g. model_ver_ind override.

        Returns:
            Path to the model directory.
        """
        if path is None:
            gnn_mm_name_hash = self.get_hash()
            model_dir_path, _ = Declare.declare_model_by_config(
                dataset_path=self.dataset_path,
                GNNModelManager_hash=gnn_mm_name_hash,
                model_ver_ind=kwargs.get('model_ver_ind') if 'model_ver_ind' in kwargs else
                self.modification.model_ver_ind,
                epochs=self.modification.epochs,
                gnn_name=self.gnn.get_hash(),
                mi_defense_hash=self.mi_defense_config.hash_for_config(),
                evasion_defense_hash=self.evasion_defense_config.hash_for_config(),
                poison_defense_hash=self.poison_defense_config.hash_for_config(),
                mi_attack_hash=self.mi_attack_config.hash_for_config(),
                evasion_attack_hash=self.evasion_attack_config.hash_for_config(),
                poison_attack_hash=self.poison_attack_config.hash_for_config(),
            )
            path = model_dir_path / 'model'
        else:
            model_dir_path = path

        # TODO Kirill, check default parameters in gnn
        self.load_model(path=path, **kwargs)
        self.gnn.eval()
        return model_dir_path

    def get_hash(
            self
    ) -> str:
        """
        Compute the SHA-256 hash of the model manager name string used for storage paths.

        Returns:
            Hex digest of the SHA-256 hash.
        """
        gnn_MM_name = self.get_name()
        json_object = json.dumps(gnn_MM_name)
        gnn_MM_name_hash = hash_data_sha256(json_object.encode('utf-8'))
        return gnn_MM_name_hash

    def save_model_executor(
            self,
            path: Union[str, Path, None] = None,
            files_paths: List[Union[str, Path]] = None
    ) -> Path:
        """
        Save the model and all associated parameter files.

        Args:
            path (Union[str, Path, None]): Directory to save the model into. If None, the path
                is derived from the global config variables. Default value: `None`.
            files_paths (List[Union[str, Path]]): List of 10 file paths for the parameter files.
                Required when path is not None.

        Returns:
            Path to the model directory.
        """
        if path is None:
            dir_path, files_paths = Declare.models_path(self)
            dir_path.mkdir(exist_ok=True, parents=True)
            path = dir_path / 'model'
        else:
            assert files_paths is not None
        assert len(files_paths) == 10
        gnn_name_file = files_paths[0]
        gnn_mm_kwargs_file = files_paths[1]

        poison_attack_kwargs_file = files_paths[2]
        poison_attack_diff_file = files_paths[3]
        poison_defense_kwargs_file = files_paths[4]
        poison_defense_diff_file = files_paths[5]
        mi_defense_kwargs_file = files_paths[6]
        evasion_defense_kwargs_file = files_paths[7]
        evasion_attack_kwargs_file = files_paths[8]
        mi_attack_kwargs_file = files_paths[9]
        self.save_model(path)

        with open(gnn_name_file, "w") as f:
            f.write(self.gnn.get_name(obj_name_flag=True))
        with open(gnn_mm_kwargs_file, "w") as f:
            f.write(self.get_name())
        with open(poison_attack_kwargs_file, "w") as f:
            f.write(self.poison_attack_config.json_for_config())
        if self.poison_attack_flag and self.poison_attacker.attack_diff is not None:
            with open(poison_attack_diff_file, 'w') as file:
                json.dump(self.poison_attacker.attack_diff.to_json(), file, indent=2)
        with open(poison_defense_kwargs_file, "w") as f:
            f.write(self.poison_defense_config.json_for_config())
        if self.poison_defense_flag and self.poison_defender.defense_diff is not None:
            with open(poison_defense_diff_file, 'w') as file:
                json.dump(self.poison_defender.defense_diff.to_json(), file, indent=2)
        with open(mi_defense_kwargs_file, "w") as f:
            f.write(self.mi_defense_config.json_for_config())
        with open(evasion_defense_kwargs_file, "w") as f:
            f.write(self.evasion_defense_config.json_for_config())
        with open(evasion_attack_kwargs_file, "w") as f:
            f.write(self.evasion_attack_config.json_for_config())
        with open(mi_attack_kwargs_file, "w") as f:
            f.write(self.mi_attack_config.json_for_config())
        return path.parent

    def set_poison_attacker(
            self,
            poison_attack_config: Union[ConfigPattern, PoisonAttackConfig] = None,
            poison_attack_name: str = None
    ) -> None:
        """
        Configure and instantiate the poison attacker.

        Args:
            poison_attack_config (Union[ConfigPattern, PoisonAttackConfig]): Attack config.
                Defaults to EmptyPoisonAttacker if None. Default value: `None`.
            poison_attack_name (str): Attacker class name. Required if config is a plain
                PoisonAttackConfig. Default value: `None`.
        """
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
                _config_kwargs=poison_attack_config.to_savable_dict(),
            )
        self.poison_attack_config = poison_attack_config
        if poison_attack_name is None:
            poison_attack_name = self.poison_attack_config._class_name
        elif poison_attack_name != self.poison_attack_config._class_name:
            raise Exception(
                f"poison_attack_name and self.poison_attack_config._class_name should be equal, "
                f"but now poison_attack_name is {poison_attack_name}, "
                f"self.poison_attack_config._class_name is {self.poison_attack_config._class_name}"
            )
        self.poison_attack_name = poison_attack_name
        poison_attack_kwargs = getattr(self.poison_attack_config, CONFIG_OBJ).to_dict()

        from gnn_aid.attacks.poison_attacks import PoisonAttacker
        name_klass = {e.name: e for e in all_subclasses(PoisonAttacker)}

        klass = name_klass[self.poison_attack_name]
        self.poison_attacker = klass(**poison_attack_kwargs)
        self.poison_attack_flag = True

    def set_evasion_attacker(
            self,
            evasion_attack_config: Union[ConfigPattern, EvasionAttackConfig] = None,
            evasion_attack_name: str = None
    ) -> None:
        """
        Configure and instantiate the evasion attacker.

        Args:
            evasion_attack_config (Union[ConfigPattern, EvasionAttackConfig]): Attack config.
                Defaults to EmptyEvasionAttacker if None. Default value: `None`.
            evasion_attack_name (str): Attacker class name. Required if config is a plain
                EvasionAttackConfig. Default value: `None`.
        """
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
                _config_kwargs=evasion_attack_config.to_savable_dict(),
            )
        self.evasion_attack_config = evasion_attack_config
        if evasion_attack_name is None:
            evasion_attack_name = self.evasion_attack_config._class_name
        elif evasion_attack_name != self.evasion_attack_config._class_name:
            raise Exception(
                f"evasion_attack_name and self.evasion_attack_config._class_name should be equal, "
                f"but now evasion_attack_name is {evasion_attack_name}, "
                f"self.evasion_attack_config._class_name is {self.evasion_attack_config._class_name}"
            )
        self.evasion_attack_name = evasion_attack_name
        evasion_attack_kwargs = getattr(self.evasion_attack_config, CONFIG_OBJ).to_dict()

        from gnn_aid.attacks.evasion_attacks import EvasionAttacker
        name_klass = {e.name: e for e in EvasionAttacker.__subclasses__()}
        klass = name_klass[self.evasion_attack_name]
        self.evasion_attacker = klass(**evasion_attack_kwargs)
        self.evasion_attack_flag = True

    def set_mi_attacker(
            self,
            mi_attack_config: Union[ConfigPattern, MIAttackConfig] = None,
            mi_attack_name: str = None
    ) -> None:
        """
        Configure and instantiate the membership inference attacker.

        Args:
            mi_attack_config (Union[ConfigPattern, MIAttackConfig]): Attack config.
                Defaults to EmptyMIAttacker if None. Default value: `None`.
            mi_attack_name (str): Attacker class name. Required if config is a plain
                MIAttackConfig. Default value: `None`.
        """
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
                _config_kwargs=mi_attack_config.to_savable_dict(),
            )
        self.mi_attack_config = mi_attack_config
        if mi_attack_name is None:
            mi_attack_name = self.mi_attack_config._class_name
        elif mi_attack_name != self.mi_attack_config._class_name:
            raise Exception(
                f"mi_attack_name and self.mi_attack_config._class_name should be equal, "
                f"but now mi_attack_name is {mi_attack_name}, "
                f"self.mi_attack_config._class_name is {self.mi_attack_config._class_name}"
            )
        self.mi_attack_name = mi_attack_name
        mi_attack_kwargs = getattr(self.mi_attack_config, CONFIG_OBJ).to_dict()

        from gnn_aid.attacks.mi_attacks import MIAttacker
        name_klass = {e.name: e for e in MIAttacker.__subclasses__()}
        klass = name_klass[self.mi_attack_name]
        self.mi_attacker = klass(**mi_attack_kwargs)
        self.mi_attack_flag = True

    def set_poison_defender(
            self,
            poison_defense_config: Union[ConfigPattern, PoisonDefenseConfig] = None,
            poison_defense_name: str = None
    ) -> None:
        """
        Configure and instantiate the poison defender.

        Args:
            poison_defense_config (Union[ConfigPattern, PoisonDefenseConfig]): Defense config.
                Defaults to EmptyPoisonDefender if None. Default value: `None`.
            poison_defense_name (str): Defender class name. Required if config is a plain
                PoisonDefenseConfig. Default value: `None`.
        """
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
                _config_kwargs=poison_defense_config.to_savable_dict(),
            )
        self.poison_defense_config = poison_defense_config
        if poison_defense_name is None:
            poison_defense_name = self.poison_defense_config._class_name
        elif poison_defense_name != self.poison_defense_config._class_name:
            raise Exception(
                f"poison_defense_name and self.poison_defense_config._class_name should be equal, "
                f"but now poison_defense_name is {poison_defense_name}, "
                f"self.poison_defense_config._class_name is {self.poison_defense_config._class_name}"
            )
        self.poison_defense_name = poison_defense_name
        poison_defense_kwargs = getattr(self.poison_defense_config, CONFIG_OBJ).to_dict()

        from gnn_aid.defenses.poison_defense import PoisonDefender
        name_klass = {e.name: e for e in all_subclasses(PoisonDefender)}
        klass = name_klass[self.poison_defense_name]
        self.poison_defender = klass(**poison_defense_kwargs)
        self.poison_defense_flag = True

    def set_evasion_defender(
            self,
            evasion_defense_config: Union[ConfigPattern, EvasionDefenseConfig] = None,
            evasion_defense_name: str = None
    ) -> None:
        """
        Configure and instantiate the evasion defender.

        Args:
            evasion_defense_config (Union[ConfigPattern, EvasionDefenseConfig]): Defense config.
                Defaults to EmptyEvasionDefender if None. Default value: `None`.
            evasion_defense_name (str): Defender class name. Required if config is a plain
                EvasionDefenseConfig. Default value: `None`.
        """
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
                _config_kwargs=evasion_defense_config.to_savable_dict(),
            )
        self.evasion_defense_config = evasion_defense_config
        if evasion_defense_name is None:
            evasion_defense_name = self.evasion_defense_config._class_name
        elif evasion_defense_name != self.evasion_defense_config._class_name:
            raise Exception(
                f"evasion_defense_name and self.evasion_defense_config._class_name should be equal, "
                f"but now evasion_defense_name is {evasion_defense_name}, "
                f"self.evasion_defense_config._class_name is {self.evasion_defense_config._class_name}"
            )
        self.evasion_defense_name = evasion_defense_name
        evasion_defense_kwargs = getattr(self.evasion_defense_config, CONFIG_OBJ).to_dict()

        from gnn_aid.defenses.evasion_defense import EvasionDefender
        name_klass = {e.name: e for e in EvasionDefender.__subclasses__()}
        klass = name_klass[self.evasion_defense_name]
        self.evasion_defender = klass(**evasion_defense_kwargs)
        self.evasion_defense_flag = True

    def set_mi_defender(
            self,
            mi_defense_config: Union[ConfigPattern, MIDefenseConfig] = None,
            mi_defense_name: str = None
    ) -> None:
        """
        Configure and instantiate the membership inference defender.

        Args:
            mi_defense_config (Union[ConfigPattern, MIDefenseConfig]): Defense config.
                Defaults to EmptyMIDefender if None. Default value: `None`.
            mi_defense_name (str): Defender class name. Required if config is a plain
                MIDefenseConfig. Default value: `None`.
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
                _config_kwargs=mi_defense_config.to_savable_dict(),
            )
        self.mi_defense_config = mi_defense_config
        if mi_defense_name is None:
            mi_defense_name = self.mi_defense_config._class_name
        elif mi_defense_name != self.mi_defense_config._class_name:
            raise Exception(
                f"mi_defense_name and self.mi_defense_config._class_name should be equal, "
                f"but now mi_defense_name is {mi_defense_name}, "
                f"self.mi_defense_config._class_name is {self.mi_defense_config._class_name}"
            )
        self.mi_defense_name = mi_defense_name
        mi_defense_kwargs = getattr(self.mi_defense_config, CONFIG_OBJ).to_dict()

        from gnn_aid.defenses.mi_defense import MIDefender
        name_klass = {e.name: e for e in MIDefender.__subclasses__()}
        klass = name_klass[self.mi_defense_name]
        self.mi_defender = klass(**mi_defense_kwargs)
        self.mi_defense_flag = True

    @staticmethod
    def available_attacker(
    ):
        pass

    @staticmethod
    def available_defender(
    ):
        pass

    @staticmethod
    def from_model_path(
            model_path: dict,
            dataset_path: Union[str, Path],
            **kwargs
    ) -> Tuple[Type, Path]:
        """
        Reconstruct a model manager from saved path info and load model weights.

        Args:
            model_path (dict): Dict with keys identifying the model path components.
            dataset_path (Union[str, Path]): Path to the dataset.
            **kwargs: Additional arguments forwarded to the model manager constructor.

        Returns:
            Tuple of the loaded GNNModelManager instance and the model directory path.
        """
        model_dir_path, files_paths = Declare.declare_model_by_config(
            dataset_path=dataset_path,
            GNNModelManager_hash=str(model_path['gnn_model_manager']),
            epochs=int(model_path['epochs']) if model_path['epochs'] != 'None' else None,
            model_ver_ind=int(model_path['model_ver_ind']),
            gnn_name=model_path['gnn'],
            poison_attack_hash=model_path['poison_attacker'],
            poison_defense_hash=model_path['poison_defender'],
            evasion_defense_hash=model_path['evasion_defender'],
            mi_defense_hash=model_path['mi_defender'],
            evasion_attack_hash=model_path['evasion_attacker'],
            mi_attack_hash=model_path['mi_attacker'],
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
            manager_config = ConfigPattern(**params)
        with open(FRAMEWORK_PARAMETERS_PATH, 'r') as f:
            framework_model_managers_info = json.load(f)
        if class_name in framework_model_managers_info.keys():
            klass = import_by_name(class_name, ["gnn_aid.models_builder.model_managers.framework_mm"])
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

    def get_full_info(
            self
    ) -> dict:
        """
        Get available info about this model for the frontend.

        Returns:
            Dict with keys 'manager', 'modification', and/or 'epochs'.
        """
        result = {}
        if hasattr(self, 'manager_config'):
            result["manager"] = self.manager_config.to_savable_dict()
        if hasattr(self, 'modification'):
            result["modification"] = self.modification.to_savable_dict()
        if hasattr(self, 'epochs'):
            result["epochs"] = f"Epochs={self.epochs}"
        return result

    def get_model_data(
            self
    ) -> dict:
        """
        Return a dict with the model manager's own methods listed under the 'functions' key.

        Returns:
            Dict with key 'functions' mapping to a list of method names.
        """
        model_data = {}

        def get_own_functions(cls):
            return [x for x, y in cls.__dict__.items()
                    if isinstance(y, (FunctionType, classmethod, staticmethod))]

        model_data["functions"] = get_own_functions(type(self))
        return model_data

    @staticmethod
    def take_gnn_obj(
            gnn_file: Union[str, Path]
    ) -> GNNConstructor:
        """
        Load and return a GNN object from a saved JSON descriptor file.

        Args:
            gnn_file (Union[str, Path]): Path to the JSON file describing the GNN class and config.

        Returns:
            Instantiated GNN object (FrameworkGNNConstructor or user-defined class).
        """
        with open(gnn_file) as f:
            params = json.load(f)
            class_name = params.pop(CONFIG_CLASS_NAME)
            obj_name = params.pop("obj_name")
            gnn_config = ModelConfig(**params)
        user_models_obj_dict_info = UserCodeInfo.user_models_list_ref()
        if class_name == 'FrameworkGNNConstructor':
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
                    except Exception as e:
                        raise Exception(f"Can't import user class {class_name}") from e
                else:
                    gnn = UserCodeInfo.take_user_model_obj(user_models_obj_dict_info[class_name]['import_path'],
                                                           obj_name)
        return gnn

    def before_epoch(
            self,
            gen_dataset: GeneralDataset
    ):
        """ This hook is called before training the next training epoch
        """
        pass

    def after_epoch(
            self,
            gen_dataset: GeneralDataset,
            **hook_kwargs
    ):
        """ This hook is called after training the next training epoch
        """
        if self._after_epoch_hook:
            self._after_epoch_hook(**hook_kwargs)

    def before_batch(
            self,
            batch
    ):
        """ This hook is called before training the next training batch
        """
        pass

    def after_batch(
            self,
            batch
    ):
        """ This hook is called after training the next training batch
        """
        pass
