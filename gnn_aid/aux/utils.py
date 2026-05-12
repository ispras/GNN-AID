import hashlib
import json
import shutil
import warnings
from enum import EnumMeta
from pathlib import Path
from pydoc import locate
from time import time
from typing import Union, Type, Any, Tuple, Callable

import numpy as np
import torch
from torch import Tensor
from torch_sparse import SparseTensor
from tqdm import tqdm

root_dir = Path(__file__).parent.parent.parent.resolve()  # directory of source root

GRAPHS_DIR = root_dir / 'data'
DATASETS_DIR = root_dir / 'datasets'
MODELS_DIR = root_dir / 'models'
EXPLANATIONS_DIR = root_dir / 'explanations'
DATA_INFO_DIR = root_dir / 'data_info'
METAINFO_DIR = root_dir / "metainfo"
SAVE_DIR_STRUCTURE_PATH = METAINFO_DIR / "save_dir_structure.json"
TORCH_GEOM_GRAPHS_PATH = METAINFO_DIR / "torch_geom_index.json"
EXPLAINERS_INIT_PARAMETERS_PATH = METAINFO_DIR / "explainers_init_parameters.json"
EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH = METAINFO_DIR / "explainers_local_run_parameters.json"
EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH = METAINFO_DIR / "explainers_global_run_parameters.json"

POISON_ATTACK_PARAMETERS_PATH = METAINFO_DIR / "poison_attack_parameters.json"
POISON_DEFENSE_PARAMETERS_PATH = METAINFO_DIR / "poison_defense_parameters.json"
EVASION_ATTACK_PARAMETERS_PATH = METAINFO_DIR / "evasion_attack_parameters.json"
EVASION_DEFENSE_PARAMETERS_PATH = METAINFO_DIR / "evasion_defense_parameters.json"
MI_ATTACK_PARAMETERS_PATH = METAINFO_DIR / "mi_attack_parameters.json"
MI_DEFENSE_PARAMETERS_PATH = METAINFO_DIR / "mi_defense_parameters.json"

MODULES_PARAMETERS_PATH = METAINFO_DIR / "modules_parameters.json"
FUNCTIONS_PARAMETERS_PATH = METAINFO_DIR / "functions_parameters.json"
FRAMEWORK_PARAMETERS_PATH = METAINFO_DIR / "framework_parameters.json"
OPTIMIZERS_PARAMETERS_PATH = METAINFO_DIR / "optimizers_parameters.json"
CUSTOM_LAYERS_INFO_PATH = METAINFO_DIR / "information_check_correctness_models.json"
USER_MODELS_DIR = root_dir / "user_models_obj"
USER_MODEL_MANAGER_DIR = root_dir / "user_models_managers"
USER_MODEL_MANAGER_INFO = USER_MODEL_MANAGER_DIR / "user_model_managers_info.json"
USER_DATASET_DIR = root_dir / "user_datasets"

IMPORT_INFO_KEY = "import_info"
TECHNICAL_PARAMETER_KEY = "_technical_parameter"


def hash_data_sha256(
        data: bytes
) -> str:
    return hashlib.sha256(data).hexdigest()


def import_by_name(
        name: str,
        packs: list = None
) -> Type[Any]:
    """Import a class by name, searching in the given packages.

    Args:
        name (str): Class name, fully qualified or relative.
        packs (list): List of package paths to search in. Default value: `None`.

    Returns:
        Type[Any]: The imported class.

    Raises:
        ImportError: If the class cannot be found in any of the given packages.
    """
    if packs is None:
        return locate(name)
    else:
        for pack in packs:
            klass = locate(f"{pack}.{name}")
            if klass is not None:
                return klass
            raise ImportError(f"Cannot import class '{name}' from module {pack}.")
    raise ImportError(f"Cannot import class '{name}' from modules {packs}.")


def model_managers_info_by_names_list(
        model_managers_names: set,
) -> dict:
    """Return info dicts for the given model manager class names.

    Looks up each name in the framework parameters file first, then in the
    user model managers info file.

    Args:
        model_managers_names (set): Set of model manager class names (user and framework).

    Returns:
        dict: Mapping from model manager name to its parameter info dict.

    Raises:
        Exception: If a name is not found in either source.
    """
    model_managers_info = {}
    with open(FRAMEWORK_PARAMETERS_PATH) as f:
        framework_model_managers = json.load(f)
    with open(USER_MODEL_MANAGER_INFO) as f:
        user_model_managers = json.load(f)
    for model_manager_name in model_managers_names:
        if model_manager_name in framework_model_managers:
            model_managers_info[model_manager_name] = framework_model_managers[model_manager_name]
        elif model_manager_name in user_model_managers:
            model_managers_info[model_manager_name] = user_model_managers[model_manager_name]
        else:
            raise Exception(f"Model manager {model_manager_name} is not defined among the built-in, "
                            f"not among the custom model managers."
                            f"To make {model_manager_name} available for use, enter information about its parameters "
                            f"in the file user_model_managers_info.json")
    return model_managers_info


def setting_class_default_parameters(
        class_name: str,
        class_kwargs: dict,
        default_parameters_file_path: Union[str, Path]
) -> Tuple[dict, dict]:
    """Fill missing class kwargs with defaults from a JSON parameter file.

    Args:
        class_name (str): Class name; must match a key in the parameter file.
        class_kwargs (dict): Parameters to supplement with defaults.
        default_parameters_file_path (Union[str, Path]): Path to the JSON file
            containing default parameters for class_name.

    Returns:
        Tuple[dict, dict]: A pair ``(class_kwargs_for_save, class_kwargs_for_init)``
            where values are cast to the appropriate types. ``class_kwargs_for_save``
            preserves the flat parameter layout; ``class_kwargs_for_init`` applies
            any parameter grouping defined in the file.

    Raises:
        Exception: If class_name is not in the parameter file, or an unsupported
            grouping format is encountered.
    """
    with open(default_parameters_file_path) as f:
        class_kwargs_default = json.load(f)
        if class_name not in class_kwargs_default.keys():
            raise Exception(f"{class_name} is not currently supported")
        class_kwargs_default = class_kwargs_default[class_name]
    for key, val in class_kwargs.items():
        if key == TECHNICAL_PARAMETER_KEY or key not in class_kwargs_default.keys():
            warnings.warn(f"WARNING: Parameter {key} cannot be set for {class_name} "
                          f"in def setting_class_default_parameters")
            continue
        key_1 = class_kwargs_default[key][1]
        if key_1 == 'int_or_tuple' or key_1 == 'int_or_tuple_or_mask':
            try:
                class_kwargs[key] = int(val)
            except TypeError:  # tuple
                class_kwargs[key] = tuple(val)
            except ValueError:  # string
                class_kwargs[key] = val
        elif val is None or key_1 == 'string'\
                or (key_1 == 'dynamic' and isinstance(val, str))\
                or np.isinf(val):
            class_kwargs[key] = val
        else:
            class_kwargs[key] = locate(key_1)(val)
    for key, val in class_kwargs_default.items():
        if key != TECHNICAL_PARAMETER_KEY and key not in class_kwargs.keys():
            if val[2] is None or val[1] == 'string' or val[2] == np.inf:
                class_kwargs[key] = val[2]
            elif val[1] == 'int_or_tuple' or val[1] == 'int_or_tuple_or_mask':
                try:
                    class_kwargs[key] = int(val[2])
                except TypeError:  # tuple
                    class_kwargs[key] = tuple(val[2])
                except ValueError:  # string
                    class_kwargs[key] = val[2]
            else:
                class_kwargs[key] = locate(val[1])(val[2])

    class_kwargs_for_save = class_kwargs.copy()
    PARAMETERS_GROUPING = "parameters_grouping"

    if TECHNICAL_PARAMETER_KEY in class_kwargs_default and \
            PARAMETERS_GROUPING in class_kwargs_default[TECHNICAL_PARAMETER_KEY] and \
            len(class_kwargs_default[TECHNICAL_PARAMETER_KEY][PARAMETERS_GROUPING]) > 0:
        for elem in class_kwargs_default[TECHNICAL_PARAMETER_KEY][PARAMETERS_GROUPING]:
            if elem[0] == 'tuple':
                parameters_grouping = tuple()
                for parameter_name_loop_elem in elem[1]:
                    parameters_grouping = parameters_grouping + (
                        class_kwargs.pop(parameter_name_loop_elem),)
                class_kwargs[elem[2]] = parameters_grouping
            elif elem[0] == 'list':
                parameters_grouping = []
                for parameter_name_loop_elem in elem[1]:
                    parameters_grouping.append(class_kwargs.pop(parameter_name_loop_elem))
                class_kwargs[elem[2]] = parameters_grouping
            else:
                raise Exception(
                    f"Grouping parameters in the format {elem[0]} is not currently supported")

    class_kwargs_for_init = class_kwargs.copy()

    return class_kwargs_for_save, class_kwargs_for_init


def deep_update(
        target: dict,
        overrides: dict
) -> dict:
    """Recursively update a dictionary with values from overrides.

    Nested dictionaries are merged instead of overwritten.

    Args:
        target (dict): Dictionary to update in place.
        overrides (dict): Values to merge into target.

    Returns:
        dict: The updated target dictionary.
    """
    for key, value in overrides.items():
        if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
        ):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def all_subclasses(
        cls: Type[Any]
) -> set:
    """ Return all subclasses of cls recursively.
    """
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


def move_to_same_device(
        *tensors,
        device: torch.device = None,
) -> tuple:
    """
    Move all tensor arguments to the same device.
    Non-tensor arguments are passed through unchanged.

    Args:
        *tensors: Arbitrary objects; torch.Tensor instances are moved to device.
        device (torch.device): Target device. Defaults to CUDA if available, else CPU.

    Returns:
        tuple: Arguments with tensors moved to the target device.
    """
    def is_movable_tensor(x):
        return isinstance(x, torch.Tensor) and hasattr(x, 'to')
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    moved_args = tuple(
        t.to(device) if is_movable_tensor(t) else t
        for t in tensors
    )
    return moved_args


class tmp_dir():
    """
    Context manager that creates a temporary directory and removes it on exit.
    """

    def __init__(
            self,
            path: Path,
    ):
        """
        Args:
            path (Path): Base path; the temporary directory is created alongside it.
        """
        self.path = path
        self.tmp_dir = self.path.parent / (self.path.name + str(time()))

    def __enter__(
            self
    ) -> Path:
        """ Create the temporary directory and return its path.
        """
        self.tmp_dir.mkdir(parents=True)
        return self.tmp_dir

    def __exit__(
            self,
            exception_type,
            exception_value,
            exception_traceback,
    ) -> None:
        """ Remove the temporary directory.
        """
        try:
            shutil.rmtree(self.tmp_dir)
        except FileNotFoundError:
            pass


def short_str(
        obj: Any,
        max_len: int = 120,
) -> str:
    """
    Return a string representation of obj truncated to max_len characters.

    Args:
        obj (Any): Object to represent as a string.
        max_len (int): Maximum length of the result. Default value: `120`.

    Returns:
        str: Truncated string representation.
    """
    res = str(obj)
    if len(res) > max_len:
        res = res[:max_len - 5] + "..." + res[-2:]
    return res


def edge_index_to_edge_list(
        edge_index: Union[list, Tensor],
        directed: bool = True
) -> list:
    """
    Convert a COO edge index to a list of (src, dst) pairs.

    Args:
        edge_index (Union[list, tensor]): A [2, E] tensor or a list of such tensors.
        directed (bool): If False, only edges where src <= dst are kept. Default value: `True`.

    Returns:
        list: List of (src, dst) integer pairs, or a list of such lists.
    """
    if isinstance(edge_index, list):
        return [edge_index_to_edge_list(x) for x in edge_index]

    assert edge_index.shape[0] == 2
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    if directed:
        return edges
    else:
        # Leave i <= j edges only
        result = []
        for i, j in edges:
            if i > j: continue
            result.append([i, j])
        return result


def shape(
        x: Union[Tensor, SparseTensor]
) -> list:
    """
    Return the shape of a dense or sparse tensor as a list.

    Args:
        x (Union[Tensor, SparseTensor]): Input tensor.

    Returns:
        list: Shape as a list of integers.

    Raises:
        NotImplementedError: If x is neither a Tensor nor a SparseTensor.
    """
    if isinstance(x, Tensor):
        _shape = list(x.shape)
    elif isinstance(x, SparseTensor):
        _shape = x.sizes()
    else:
        raise NotImplementedError

    return _shape


class MetaEnum(EnumMeta):
    """
    EnumMeta subclass that supports the ``in`` operator for value membership tests.
    """

    def __contains__(cls, item):
        """ Return True if item is a valid value of the enum.
        """
        try:
            cls(item)
        except ValueError:
            return False
        return True


class ProgressBar(tqdm):
    """
    tqdm progress bar with optional frontend hooks for init, reset, and update events.
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(ProgressBar, self).__init__(*args, **kwargs)
        self.kwargs = {}

        # Hooks from frontend client
        self._on_init_hook: Callable = None
        self._on_reset_hook: Callable = None
        self._on_update_hook: Callable = None

    def set_hook(
            self,
            hook: Callable,
            where: str
    ) -> None:
        """
        Register a callback for a progress bar event.

        Args:
            hook (Callable): Callback to invoke on the event.
            where (str): Event name; one of ``'on_init'``, ``'on_reset'``, ``'on_update'``.

        Raises:
            ValueError: If where is not a recognized event name.
        """
        if where == 'on_init':
            self._on_init_hook = hook
        elif where == 'on_reset':
            self._on_reset_hook = hook
        elif where == 'on_update':
            self._on_update_hook = hook
        else:
            raise ValueError(f"Hook {where} is not supported")

    def init(
            self,
            **kwargs
    ):
        """ Initialize the progress bar state and fire the on_init hook.
        """
        # TODO do we save kwargs?
        self.kwargs.update(**kwargs)

        if self._on_init_hook:
            self._on_init_hook()

    def reset(
            self,
            total: Union[float, None] = None,
            **kwargs
    ):
        """
        Reset the progress bar and fire the on_reset hook.

        Args:
            total (Union[float, None]): New total for the progress bar. Default value: `None`.
            **kwargs: Additional keyword arguments stored in self.kwargs.
        """
        super().reset(total=total)
        self.kwargs = kwargs
        if self._on_reset_hook:
            self._on_reset_hook()

    def update(
            self,
            n: int = 1
    ):
        """ Advance the progress bar by n steps and fire the on_update hook.
        """
        res = super().update(n=n)
        if self._on_update_hook:
            self._on_update_hook()
        return res
