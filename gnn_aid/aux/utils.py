import hashlib
import json
import warnings
from enum import EnumMeta
from pathlib import Path
from pydoc import locate
from typing import Union, Type, Any, Tuple

import numpy as np
import torch
from torch import tensor, Tensor
from torch_sparse import SparseTensor

root_dir = Path(__file__).parent.parent.parent.resolve()  # directory of source root
root_dir_len = len(root_dir.parts)

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
        data
) -> str:
    return hashlib.sha256(data).hexdigest()


def import_by_name(
        name: str,
        packs: list = None
) -> Type[Any]:
    """
    Import name from packages, return class
    :param name: class name, full or relative
    :param packs: list of packages to search in
    :return: <class>
    """
    from pydoc import locate
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
        model_managers_names: set
) -> dict:
    """
    :param model_managers_names: set with model managers class names (user and framework)
    :return: dict with info about model managers 
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
    """
    :param class_name: class name, should be same in default_parameters_file
    :param class_kwargs: dict with parameters, which needs to be supplemented with default parameters
    :param default_parameters_file_path: path to the file with default parameters of the class_name object
    :return: new dict with all class kwargs
    """
    with open(default_parameters_file_path) as f:
        class_kwargs_default = json.load(f)
        if class_name not in class_kwargs_default.keys():
            raise Exception(f"{class_name} is not currently supported")
        class_kwargs_default = class_kwargs_default[class_name]
    for key, val in class_kwargs.items():
        if key == TECHNICAL_PARAMETER_KEY or key not in class_kwargs_default.keys():
            # raise Exception(
            #     f"Parameter {key} cannot be set for {class_name}")
            warnings.warn(f"WARNING: Parameter {key} cannot be set for {class_name} "
                          f"in def setting_class_default_parameters")
            continue
        key_1 = class_kwargs_default[key][1]
        if key_1 == 'int_or_tuple':
            try:
                class_kwargs[key] = int(val)
            except TypeError:
                class_kwargs[key] = tuple(val)
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
            elif val[1] == 'int_or_tuple':
                try:
                    class_kwargs[key] = int(val[2])
                except TypeError:
                    class_kwargs[key] = tuple(val[2])
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
    """
    Recursively update a dictionary with values from overrides.
    Nested dictionaries are merged instead of overwritten.
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
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


def move_to_same_device(
        *tensors,
        device: torch.device = None
):
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
    Temporary create a directory near the given path. Remove it on exit.
    """

    def __init__(
            self,
            path: Path
    ):
        self.path = path
        from time import time
        self.tmp_dir = self.path.parent / (self.path.name + str(time()))

    def __enter__(
            self
    ) -> Path:
        self.tmp_dir.mkdir(parents=True)
        return self.tmp_dir

    def __exit__(
            self,
            exception_type,
            exception_value,
            exception_traceback
    ) -> None:
        import shutil
        try:
            shutil.rmtree(self.tmp_dir)
        except FileNotFoundError:
            pass


def short_str(obj, max_len=120):
    res = str(obj)
    if len(res) > max_len:
        res = res[:max_len - 5] + "..." + res[-2:]
    return res


def edge_index_to_edge_list(
        edge_index: Union[list, tensor],
        directed: bool = True
) -> list:
    if isinstance(edge_index, list):
        return [edge_index_to_edge_list(x) for x in edge_index]

    assert edge_index.shape[0] == 2
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    if directed:
        return edges
    else:
        # Удалим дубликаты: (i, j) и (j, i) → оставить только (min(i, j), max(i, j))
        edge_set = set()
        for i, j in edges:
            edge_set.add(tuple(sorted((i, j))))
        return list(edge_set)


def shape(
        x: Union[Tensor, SparseTensor]
) -> list:
    if isinstance(x, Tensor):
        _shape = list(x.shape)
    elif isinstance(x, SparseTensor):
        _shape = x.sizes()
    else:
        raise NotImplementedError

    return _shape


class MetaEnum(EnumMeta):
    """
    A helper class which allows us to do "x in XEnum" if XEnum extends this class
    """
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True
