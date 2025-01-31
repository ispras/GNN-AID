import json
import logging
import re
from json import JSONEncoder
import copy
import inspect
from pathlib import Path
from typing import Union, Any, Type, Tuple, List

from aux.utils import setting_class_default_parameters, EXPLAINERS_INIT_PARAMETERS_PATH, \
    EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH, EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH, \
    OPTIMIZERS_PARAMETERS_PATH, FUNCTIONS_PARAMETERS_PATH, FRAMEWORK_PARAMETERS_PATH, import_by_name, hash_data_sha256

CONFIG_SAVE_KWARGS_KEY = '__save_kwargs_to_be_used_for_saving'
# CONFIG_PARAMS_PATH_KEY = '__default_parameters_file_path'
CONFIG_OBJ = "config_obj"
CONFIG_CLASS_NAME = 'class_name'
DATA_CHANGE_FLAG = "__data_change_flag"


# TECHNICAL_KEYS_SET_FOR_CONFIGS = {CONFIG_PARAMS_PATH_KEY, CONFIG_CLASS_NAME,
#                                   CONFIG_SAVE_KWARGS_KEY, DATA_CHANGE_FLAG}


# Patch of json.dumps() - classes which implement to_json() can be jsonified
def _default(
        self,
        obj: Any
):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default

_key_path = {
    "optimizer": {
        "_config_class": "Config",
        "_import_path": OPTIMIZERS_PARAMETERS_PATH,
        "_class_import_info": ["torch.optim"],
    },
    "loss_function": {
        "_config_class": "Config",
        "_import_path": FUNCTIONS_PARAMETERS_PATH,
        "_class_import_info": ["torch.nn"],
    },
}


class GeneralConfig:
    # TODO Kirill rename, docs
    _mutable = False
    _TECHNICAL_KEYS = {"_class_name", "_class_import_info", "_import_path", "_config_class",
                       "_config_kwargs"}
    _CONFIG_KEYS = "_config_keys"

    def __init__(
            self,
            **kwargs
    ):
        self._config_keys = set()

        for key, value in kwargs.items():
            if isinstance(value, dict) and len(value.values()) != 0 and set(value.keys()).issubset(
                    self._TECHNICAL_KEYS):
                if len(value.values()) != len(self._TECHNICAL_KEYS):
                    value = GeneralConfig.set_defaults_config_pattern_info(key=key, value=value)
                assert len(value.values()) == len(self._TECHNICAL_KEYS)
                value = ConfigPattern(**value)
            if key != self._CONFIG_KEYS and key not in self._TECHNICAL_KEYS:
                self._config_keys.add(key)
            setattr(self, key, value)

    def __setattr__(
            self,
            key: str,
            value: Any
    ) -> None:
        frame = inspect.currentframe()
        try:
            locals_info = frame.f_back.f_locals
            if locals_info.get('self', None) is self:
                # print("Called from this class!")
                if self._mutable or key == self._CONFIG_KEYS or key in getattr(self,
                                                                               self._CONFIG_KEYS) or key in self._TECHNICAL_KEYS:
                    self.__dict__[key] = value
                else:
                    raise TypeError
            else:
                # print("Called from outside of class!")
                if (self._mutable or key in getattr(self,
                                                    self._CONFIG_KEYS)) and key != self._CONFIG_KEYS and key not in self._TECHNICAL_KEYS:
                    self.__dict__[key] = value
                else:
                    raise TypeError
        except TypeError:
            raise TypeError("Config cannot be changed outside of init()!")
        # except AttributeError as ae:
        #     print(ae)
        except Exception as e:
            if self._mutable or key == self._CONFIG_KEYS or key in getattr(self, self._CONFIG_KEYS):
                self.__dict__[key] = value
            else:
                raise TypeError("Config cannot be changed outside of init()!")
        finally:
            del frame

    def json_for_config(
            self
    ) -> str:
        config_kwargs = self.to_saveable_dict().copy()
        config_kwargs = dict(sorted(config_kwargs.items()))
        json_object = json.dumps(config_kwargs, indent=2)
        return json_object

    def hash_for_config(
            self
    ) -> str:
        return hash_data_sha256(self.json_for_config().encode('utf-8'))

    def to_saveable_dict(
            self,
            compact: bool = False,
            **kwargs
    ) -> dict:
        def sorted_dict(d):
            res = {}
            for key in sorted(d):
                value = d[key]
                if isinstance(value, dict):
                    value = sorted_dict(value)
                res[key] = value
            return res

        dct = {}
        for key in sorted(kwargs):
            # FIXME misha check this can be removed
            # if key in [CONFIG_PARAMS_PATH_KEY, CONFIG_SAVE_KWARGS_KEY]:
            #     continue
            value = kwargs[key]
            if isinstance(value, (Config, ConfigPattern)):
                value = value.to_saveable_dict()
            elif isinstance(value, dict):
                # value = json.dumps(sorted_dict(value), separators=separators, indent=indent)
                value = sorted_dict(value)
            else:
                # value = json.dumps(value)
                value = value
            dct[key] = value

        if compact:
            for key, value in dct.items():
                if isinstance(value, dict):
                    dct[key] = json.dumps(value, separators=(',', ':'), indent=None)
                else:
                    # FIXME Misha, what do we do if value is special list or tuple in general
                    dct[key] = str(value)
        return dct

    def to_dict(
            self
    ) -> dict:
        """ Represent config as a dictionary, as well as all included configs.
        Dict is a copy of all values.
        """
        res = {}
        for k, v in self.__dict__.items():
            if k not in self._config_keys:
                continue
            # FIXME copy of dict, config
            if isinstance(v, Config):
                v = v.to_dict()
            res[k] = copy.copy(v)
        return res

    @staticmethod
    def set_defaults_config_pattern_info(
            key: str,
            value: dict
    ) -> dict:
        if "_config_kwargs" not in value:
            raise Exception("_config_kwargs can't set automatically")
        if key in _key_path:
            # TODO Kirill, make this better use info about intersection between keys
            value.update(_key_path[key])
        # QUE Kirill, maybe need fix
        if "_class_name" not in value:
            value.update({"_class_name": None})
        if "_import_path" not in value:
            value.update({"_import_path": None})
        if "_class_import_info" not in value:
            value.update({"_class_import_info": None})
        # elif "_class_name" not in value and "_import_path" not in value and "_class_import_info" not in value:
        #     value.update({"_class_name": None, "_import_path": None, "_class_import_info": None})
        # else:
        #     raise Exception(f"It is impossible to provide default information for the key {key}; "
        #                     f"{key} is not currently supported")
        return value

    def to_json(
            self
    ) -> dict:
        """ Special method which allows to use json.dumps() on Config object """
        return self.to_dict()


class ConfigPattern(
    GeneralConfig
):
    def __init__(
            self,
            _config_class: str,
            _config_kwargs: Union[dict, None],
            _class_name: Union[str, None] = None,
            _import_path: Union[str, Path, None] = None,
            _class_import_info: str = None):
        if _import_path is not None:
            _import_path = str(_import_path)
        super().__init__(_class_name=_class_name, _import_path=_import_path,
                         _class_import_info=_class_import_info, _config_class=_config_class,
                         _config_kwargs=_config_kwargs, config_obj=None)
        save_kwargs = None
        if self._class_name is not None:
            # if _class_import_info is None:
            #     raise Exception("_class_name is not None, but _class_import_info is None. "
            #                     "If _class_name is define, _class_import_info must be define too")
            if self._import_path is None:
                raise Exception("_class_name is not None, but _import_path is None. "
                                "If _class_name is define, _import_path must be define too")
            self._config_kwargs, save_kwargs = self._set_defaults()
        setattr(self, CONFIG_OBJ, self.make_config_by_pattern(save_kwargs))

    def __getattribute__(
            self,
            item: Union[str, Type]
    ) -> Any:
        if item == "__dict__" or item == "__class__":
            return object.__getattribute__(self, item)

        if item is CONFIG_OBJ:
            return object.__getattribute__(self, item)
        elif CONFIG_OBJ in self.__dict__ and getattr(
                self, CONFIG_OBJ) is not None and item in getattr(self, CONFIG_OBJ):
            return getattr(self, CONFIG_OBJ).__getattribute__(item)
        else:
            try:
                attr = object.__getattribute__(self, item)
            except AttributeError:
                attr = getattr(self, CONFIG_OBJ).__getattribute__(item)
            return attr

    def __setattr__(
            self,
            key: str,
            value: Any
    ) -> None:
        if (hasattr(self, CONFIG_OBJ) and hasattr(getattr(self, CONFIG_OBJ), self._CONFIG_KEYS) and
                key in getattr(getattr(self, CONFIG_OBJ), self._CONFIG_KEYS)):
            getattr(self, CONFIG_OBJ).__setattr__(key, value)
        else:
            super().__setattr__(key, value)

    def _set_defaults(
            self
    ) -> [dict, dict]:
        default_parameters_file_path = self._import_path
        kwargs = self._config_kwargs

        # Pop the first key-value supposing it is a class name
        # QUE Kirill, fix CONFIG_SAVE_KWARGS_KEY problem, add in _TECHNICAL_KEYS or remove (maybe we can set new confid kwargs after init)
        save_kwargs, init_kwargs = setting_class_default_parameters(
            class_name=self._class_name,
            class_kwargs=kwargs,
            default_parameters_file_path=default_parameters_file_path
        )
        return init_kwargs, save_kwargs

    def make_config_by_pattern(
            self,
            save_kwargs: dict
    ) -> Type:
        config_class = import_by_name(self._config_class, ["aux.configs"])
        config_obj = config_class(save_kwargs=save_kwargs, **self._config_kwargs)
        return config_obj

    def create_obj(self, **kwargs):
        if self._class_name is not None or self._class_import_info is not None:
            obj_class = import_by_name(self._class_name, self._class_import_info)
        else:
            raise Exception(f"_class_name is None, so def make_obj can not be call")
        config_obj = getattr(self, CONFIG_OBJ).to_dict()
        try:
            obj = obj_class(**kwargs, **config_obj)
        except TypeError as te:
            logging.warning(f"class {self._class_name} can not create obj by config_obj, "
                            f"add missing kwargs when call def create_obj")
            raise TypeError(te)
        except Exception as e:
            print(e)
        return obj

    def merge(
            self,
            config: Type
    ):
        self_config_obj = getattr(self, CONFIG_OBJ)
        if config.__class__.__name__ == "ConfigPattern":
            config_obj = getattr(config, CONFIG_OBJ)
            setattr(self, CONFIG_OBJ, self_config_obj.merge(config_obj))
        else:
            setattr(self, CONFIG_OBJ, self_config_obj.merge(config))
        return self

    def to_saveable_dict(
            self,
            compact: bool = False,
            need_full: bool = True,
            **kwargs
    ) -> dict:
        """
        Create dict which values are strings without spaces and are guaranteed to be sorted by key
        including inner dicts and configs.
        Result dict is a deep copy and can be modified.

        :param compact: if compact=True, outer dict values are strings without spaces
        :return: dict
        """
        if CONFIG_SAVE_KWARGS_KEY in self.__dict__ and self.__dict__[
            CONFIG_SAVE_KWARGS_KEY] is not None:
            kw = self.__dict__[CONFIG_SAVE_KWARGS_KEY]
        else:
            kw = dict(filter(lambda x: x[0] in self._TECHNICAL_KEYS, self.__dict__.items()))
            kw["_config_kwargs"] = getattr(self, CONFIG_OBJ).to_saveable_dict(compact=compact)
        # BUG Kirill, fix for modification
        if not need_full:
            kw = kw["_config_kwargs"]
        dct = super().to_saveable_dict(compact=compact, **kw)
        return dct


class Config(
    GeneralConfig
):
    """ Contains a set of named parameters.
    Immutable - values can be set in constructor only.
    Parameters can be dicts or Configs themselves.
    Supports setting of default parameters stored in json file, which can specify complex
    operations over values (e.g. technical_parameter).
    After setting defaults, saveable representation of parameters is created.
    """

    # _mutable = False
    # _CONFIG_KEYS = "_config_keys"

    def __init__(
            self,
            save_kwargs: Union[dict, None] = None,
            **kwargs
    ):
        self.__dict__[CONFIG_SAVE_KWARGS_KEY] = save_kwargs
        super().__init__(**kwargs)

    def __str__(
            self
    ) -> str:
        return str(dict(filter(lambda x: x[0] in self._config_keys, self.__dict__.items())))

    def __iter__(
            self
    ):
        for key, value in self.__dict__.items():
            if key in self._config_keys:
                yield key, value

    def __getitem__(
            self,
            item: str
    ) -> Any:
        if item in self._config_keys:
            return self.__dict__[item]

    def __contains__(
            self,
            item: str
    ) -> bool:
        return item in self._config_keys

    def __eq__(
            self,
            other: Type
    ) -> bool:
        if type(other) != type(self):
            return False
        return all(getattr(self, a) == getattr(other, a) for a in self._config_keys)

    def copy(
            self
    ) -> object:
        res = type(self)()
        # res.__dict__ = self.__dict__.copy()
        for k, v in self.__dict__.items():
            # FIXME copy of dict, config
            if k in self._config_keys:
                res.__dict__[k] = copy.copy(v)
        return res

    def merge(
            self,
            config: Union[dict, object]
    ):
        """ Create a new config with params obtained by updating self params with a given ones.
        Given config is a dict or a Config.
        """
        assert isinstance(config, (dict, type(self)))

        if isinstance(config, type(self)):
            config = config.to_dict()
            # config = dict(filter(lambda x: x[0] in config._config_keys, config.__dict__.items()))

        # kwargs = dict(filter(lambda x: x[0] in self._config_keys, self.__dict__.items()))
        kwargs = self.to_dict()
        kwargs.update(config.copy())
        return type(self)(**kwargs)

    def to_saveable_dict(
            self,
            compact: bool = False,
            **kwargs
    ) -> dict:
        """
        Create dict which values are strings without spaces and are guaranteed to be sorted by key
        including inner dicts and configs.
        Result dict is a deep copy and can be modified.

        :param compact: if compact=True, outer dict values are strings without spaces
        :return: dict
        """
        if self.__dict__[CONFIG_SAVE_KWARGS_KEY] is not None:
            kw = self.__dict__[CONFIG_SAVE_KWARGS_KEY]
        else:
            kw = dict(filter(lambda x: x[0] in self._config_keys, self.__dict__.items()))
        dct = super().to_saveable_dict(compact=compact, **kw)
        return dct


class DatasetConfig(
    Config
):
    """
    Contains a set of distinguishing characteristics to identify the dataset or family of datasets.
    Determines the path to the file with raw data in the inner storage.
    """

    def __init__(
            self,
            domain: str = None,
            group: str = None,
            graph: str = None
    ):
        """
        """
        super().__init__(domain=domain, group=group, graph=graph)

    def full_name(
            self
    ) -> tuple:
        """ Return all fields as a tuple. """
        return tuple([self.domain, self.group, self.graph])

    @staticmethod
    def from_full_name(
            full_name: tuple
    ) -> object:
        """ Build DatasetConfig from a name tuple. """
        res = DatasetConfig(
            domain=full_name[0], group=full_name[1], graph=full_name[2])
        return res


class DatasetVarConfig(Config):
    """
    Contains description of how to obtain tensors for the dataset, having DatasetConfig.
    Specifies the path to the file with tensors in the inner storage.
    """

    def __init__(
            self,
            features: dict = None,
            labeling: str = None,
            dataset_ver_ind: int = None
    ):
        """ """
        super().__init__(
            features=features, labeling=labeling,
            dataset_ver_ind=dataset_ver_ind)


class ModelStructureConfig(
    Config
):
    """
    Contains a full description of a model structure.
    Represents a list of layers.
    Access by key and iterating behave like it is list.

    General principle for describing one layer of the network:
    >>> structure=[
    >>>     {
    >>>         'label': 'n' or 'g',
    >>>         'layer': {
    >>>             ...
    >>>         },
    >>>         'batchNorm': {
    >>>             ...
    >>>         },
    >>>         'activation': {
    >>>             ...
    >>>         },
    >>>         'dropout': {
    >>>             ...
    >>>         },
    >>>         'connections': [
    >>>             {
    >>>                 ...
    >>>             },
    >>>             ...
    >>>         ]
    >>>     },
    >>>     {
    >>>         new block
    >>>     },
    >>> ]

    For connections now support variant connection between layers
    with labels: n -> n, n -> g, g -> g
    Example connections:
    >>> 'connections': [
    >>>             {
    >>>                 'into_layer': 3,
    >>>                 'connection_kwargs': {
    >>>                     'pool': {
    >>>                         'pool_type': 'global_add_pool',
    >>>                     },
    >>>                     'aggregation_type': 'cat',
    >>>                 },
    >>>             },
    >>>         ],
    into_layer: layer (block) index, numeration start from 0
    For aggregation_type now support only cat
    pool_type has taken from torch_geometric.nn, pooling

    In the case of using GINConv, it is necessary to write the internal structure nn.Sequential().
    For this case, a universal block logic is provided in the following format:
    >>> 'layer': {
    >>>             'layer_name': 'GINConv',
    >>>             'layer_kwargs': None,
    >>>             'gin_seq': [
    >>>                 {
    >>>                     'layer': {
    >>>                         'layer_name': 'Linear',
    >>>                         ...
    >>>                         },
    >>>                     },
    >>>                     'batchNorm': {
    >>>                         ...
    >>>                     },
    >>>                     'activation': {
    >>>                         ...
    >>>                     },
    >>>                 },
    >>>                 {
    >>>                     new block
    >>>                 },
    >>>             ],
    >>>             ...
    >>>         },
    Examples:
    Example of conv layer:
    >>> {
    >>>     'label': 'n',
    >>>     'layer': {
    >>>         'layer_name': 'GATConv',
    >>>         'layer_kwargs': {
    >>>             'in_channels': dataset.num_node_features,
    >>>             'out_channels': 16,
    >>>             'heads': 3,
    >>>         },
    >>>     },
    >>>     'batchNorm': {
    >>>         'batchNorm_name': 'BatchNorm1d',
    >>>         'batchNorm_kwargs': {
    >>>             'num_features': 48,
    >>>             'eps': 1e-05,
    >>>         }
    >>>     },
    >>>     'activation': {
    >>>         'activation_name': 'ReLU',
    >>>         'activation_kwargs': None,
    >>>     },
    >>>     'dropout': {
    >>>         'dropout_name': 'Dropout',
    >>>         'dropout_kwargs': {
    >>>             'p': 0.5,
    >>>         }
    >>>     },
    >>> }
    Example of gin layer:
    >>> {
    >>>     'label': 'n',
    >>>     'layer': {
    >>>         'layer_name': 'GINConv',
    >>>         'layer_kwargs': None,
    >>>         'gin_seq': [
    >>>             {
    >>>                 'layer': {
    >>>                     'layer_name': 'Linear',
    >>>                     'layer_kwargs': {
    >>>                         'in_features': dataset.num_node_features,
    >>>                         'out_features': 16,
    >>>                     },
    >>>                 },
    >>>                 'batchNorm': {
    >>>                     'batchNorm_name': 'BatchNorm1d',
    >>>                     'batchNorm_kwargs': {
    >>>                         'num_features': 16,
    >>>                         'eps': 1e-05,
    >>>                     }
    >>>                 },
    >>>                 'activation': {
    >>>                     'activation_name': 'ReLU',
    >>>                     'activation_kwargs': None,
    >>>                 },
    >>>             },
    >>>             {
    >>>                 'layer': {
    >>>                     'layer_name': 'Linear',
    >>>                     'layer_kwargs': {
    >>>                         'in_features': 16,
    >>>                         'out_features': 16,
    >>>                     },
    >>>                 },
    >>>                 'activation': {
    >>>                     'activation_name': 'ReLU',
    >>>                     'activation_kwargs': None,
    >>>                 },
    >>>             },
    >>>         ],
    >>>     },
    >>>     'connections': [
    >>>         {
    >>>             'into_layer': 3,
    >>>             'connection_kwargs': {
    >>>                 'pool': {
    >>>                     'pool_type': 'global_add_pool',
    >>>                 },
    >>>                 'aggregation_type': 'cat',
    >>>             },
    >>>         },
    >>>     ],
    >>> }
    Example of linear layer:
    >>> {
    >>>     'label': 'n',
    >>>     'layer': {
    >>>         'layer_name': 'Linear',
    >>>         'layer_kwargs': {
    >>>             'in_features': 48,
    >>>             'out_features': dataset.num_classes,
    >>>         },
    >>>     },
    >>>     'activation': {
    >>>         'activation_name': 'LogSoftmax',
    >>>         'activation_kwargs': None,
    >>>     },
    >>> }
    """

    def __init__(
            self,
            layers=None
    ):
        """ """
        super().__init__(layers=layers)

    def __str__(
            self
    ) -> str:
        return json.dumps(self, indent=2)

    def __iter__(
            self
    ) -> None:
        for layer in self.layers:
            yield layer

    def __getitem__(
            self,
            item: int
    ) -> Any:
        assert isinstance(item, int)
        return self.layers[item]

    def __len__(
            self
    ) -> int:
        return len(self.layers)


class ModelConfig(
    Config
):
    """ Config for GNN model. Can contain structure (for framework models) and/or additional
    parameters (for custom models).
    """

    def __init__(
            self,
            structure: Union[dict, ModelStructureConfig] = None,
            **kwargs
    ):
        if structure is not None and not isinstance(structure, Config):
            assert isinstance(structure, dict)
            structure = ModelStructureConfig(**structure)
        super().__init__(structure=structure, **kwargs)


class ModelManagerConfig(
    Config
):
    """
    Full description of model manager parameters.
    """

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)


class ModelModificationConfig(
    Config
):
    """
    Variability of a model given its structure and manager.
    Represents model attack type and the instance version.
    """
    _mutable = True

    def __init__(
            self,
            model_ver_ind: [int, None] = None,
            epochs=None,
            **kwargs
    ):
        """
        :param model_ver_ind: model index when saving. If None, then takes the nearest unoccupied
         index starting from 0 in ascending increments of 1
        """
        super().__init__(model_ver_ind=model_ver_ind,
                         epochs=epochs, **kwargs)
        self.__dict__[DATA_CHANGE_FLAG] = False

    def __setattr__(
            self,
            key: str,
            value: Any
    ) -> None:
        # Any change of ModelModificationConfig should change flag
        self.__dict__[DATA_CHANGE_FLAG] = True
        super().__setattr__(key, value)

    def data_change_flag(
            self
    ) -> bool:
        loc = self.__dict__[DATA_CHANGE_FLAG]
        self.__dict__[DATA_CHANGE_FLAG] = False
        return loc


class EvasionAttackConfig(
    Config
):
    _mutable = True

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(
            **kwargs,
        )


class EvasionDefenseConfig(
    Config
):
    _mutable = True

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(
            **kwargs,
        )


class PoisonAttackConfig(
    Config
):
    _mutable = True

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(
            **kwargs,
        )


class PoisonDefenseConfig(
    Config
):
    _mutable = True

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(
            **kwargs,
        )


class MIAttackConfig(
    Config
):
    _mutable = True

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(
            **kwargs,
        )


class MIDefenseConfig(
    Config
):
    _mutable = True

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(
            **kwargs,
        )


class ExplainerInitConfig(
    Config
):
    """
    """

    def __init__(self,
                 # class_name: str,
                 **kwargs):
        super().__init__(
            # class_name=class_name,
            **kwargs,
            # **{CONFIG_PARAMS_PATH_KEY: EXPLAINERS_INIT_PARAMETERS_PATH}
        )


class ExplainerRunConfig(
    Config
):
    """
    """

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )


class ExplainerModificationConfig(
    Config
):
    """
    """

    # _mutable = True

    def __init__(
            self,
            explainer_ver_ind: [int, None] = None,
            **kwargs
    ):
        super().__init__(
            explainer_ver_ind=explainer_ver_ind,
            **kwargs
        )


if __name__ == '__main__':
    # d = MappingProxyType({})
    # cfg = Config(a='1', b=2)
    # cfg._config_keys = set()
    optimizer_info = ConfigPattern(
        _config_class="Config",
        # _class_name="Adadelta",
        _class_name="Adam",
        _class_import_info=["torch.optim"],
        _import_path=OPTIMIZERS_PARAMETERS_PATH,
        _config_kwargs={},
    )
    print(optimizer_info)
    # print(optimizer_info.create_obj())
    print()
