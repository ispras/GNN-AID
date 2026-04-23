import copy
import inspect
import json
import logging
from json import JSONEncoder
from pathlib import Path
from typing import Any, Self, Union, Type, Tuple

from gnn_aid.aux.utils import OPTIMIZERS_PARAMETERS_PATH, FUNCTIONS_PARAMETERS_PATH, \
    hash_data_sha256, deep_update, setting_class_default_parameters, import_by_name

CONFIG_SAVE_KWARGS_KEY = '__save_kwargs_to_be_used_for_saving'
CONFIG_OBJ = "config_obj"
CONFIG_CLASS_NAME = 'class_name'
DATA_CHANGE_FLAG = "__data_change_flag"


def _default(
        self: Any,
        obj: Any
):
    """ Patch of json.dumps() - classes which implement to_json() can be jsonified
    """
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
    """
    Base config class that manages named parameters with controlled mutability.

    # TODO Kirill rename
    """
    _mutable = False
    _TECHNICAL_KEYS = {"_class_name", "_class_import_info", "_import_path", "_config_class",
                       "_config_kwargs"}
    _CONFIG_KEYS = "_config_keys"

    def __init__(
            self,
            **kwargs
    ):
        """
        Initialize config with keyword arguments as named parameters.

        Each value that is a dict whose keys are a subset of _TECHNICAL_KEYS is
        automatically converted to a ConfigPattern instance.

        Args:
            **kwargs: Named parameters to store in this config.
        """
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
        """
        Enforce immutability: allow writes only from within __init__ or for allowed keys.

        Args:
            key (str): Attribute name to set.
            value (Any): Value to assign.

        Raises:
            TypeError: If the write is not permitted by mutability rules.
        """
        frame = inspect.currentframe()
        try:
            locals_info = frame.f_back.f_locals
            if locals_info.get('self', None) is self:
                if self._mutable or key == self._CONFIG_KEYS or key in getattr(self,
                                                                               self._CONFIG_KEYS) or key in self._TECHNICAL_KEYS:
                    self.__dict__[key] = value
                else:
                    raise TypeError
            else:
                if (self._mutable or key in getattr(self,
                                                    self._CONFIG_KEYS)) and key != self._CONFIG_KEYS and key not in self._TECHNICAL_KEYS:
                    self.__dict__[key] = value
                else:
                    raise TypeError
        except TypeError:
            raise TypeError("Config cannot be changed outside of init()!")
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
        """ Serialize config to a sorted, deterministic JSON string.
        """
        config_kwargs = self.to_savable_dict().copy()
        config_kwargs = dict(sorted(config_kwargs.items()))
        json_object = json.dumps(config_kwargs, indent=2)
        return json_object

    def hash_for_config(
            self
    ) -> str:
        return hash_data_sha256(self.json_for_config().encode('utf-8'))

    def to_savable_dict(
            self,
            compact: bool = False,
            **kwargs
    ) -> dict:
        """
        Build a sorted, deep-copy dict of config values ready for serialization.

        Nested configs are recursively expanded. With compact=True, all outer values
        are converted to strings without spaces for use in filesystem paths.

        Args:
            compact (bool): If True, outer dict values are strings without spaces. Default value: `False`.
            **kwargs: Key-value pairs to serialize; overrides the default set of config keys.

        Returns:
            Sorted serializable dict of config values.
        """
        from gnn_aid.data_structures.configs import Config
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
            value = kwargs[key]
            if isinstance(value, (Config, ConfigPattern)):
                value = value.to_savable_dict()
            elif isinstance(value, dict):
                value = sorted_dict(value)
            else:
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
        """
        Return a copy of the config as a plain dictionary, including nested configs.
        """
        from gnn_aid.data_structures.configs import Config
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
        """
        Fill in missing technical keys for a ConfigPattern descriptor dict.

        Args:
            key (str): Config key name; used to look up defaults in _key_path.
            value (dict): Partial ConfigPattern descriptor; must contain '_config_kwargs'.

        Returns:
            Completed descriptor dict with all required technical keys set.

        Raises:
            Exception: If '_config_kwargs' is missing from value.
        """
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
        return value

    def to_json(
            self
    ) -> dict:
        return self.to_dict()

    def clone_with(
            self,
            overrides: dict
    ) -> Self:
        """
        Return a deep copy of this config with specified fields overridden.

        Useful when working with mostly immutable config objects that need to be
        reused with minor changes, without mutating the original.

        Args:
            overrides (dict): Fields to override; keys must match those returned by to_dict().

        Returns:
            New instance of the same class with merged configuration.
        """
        config_data = copy.deepcopy(
            self.to_savable_dict()
        )
        config_data = deep_update(
            target=config_data,
            overrides=overrides
        )
        return type(self)(**config_data)


class ConfigPattern(
    GeneralConfig
):
    """
    A lazy config wrapper that holds a class name, import path, and constructor kwargs,
    and creates a Config instance on demand via make_config_by_pattern.
    """

    def __init__(
            self,
            _config_class: str,
            _config_kwargs: Union[dict, None],
            _class_name: Union[str, None] = None,
            _import_path: Union[str, Path, None] = None,
            _class_import_info: Union[str, list[str]] = None
    ):
        """
        Args:
            _config_class (str): Name of the Config subclass to instantiate.
            _config_kwargs (Union[dict, None]): Constructor kwargs for the config class.
            _class_name (Union[str, None]): Class name to import; triggers default-filling if set.
            _import_path (Union[str, Path, None]): Path to the JSON file with default parameters.
            _class_import_info (Union[str, list[str]]): Package paths for importing _class_name.
        """
        if _import_path is not None:
            _import_path = str(_import_path)
        super().__init__(_class_name=_class_name, _import_path=_import_path,
                         _class_import_info=_class_import_info, _config_class=_config_class,
                         _config_kwargs=_config_kwargs, config_obj=None)
        save_kwargs = None
        if self._class_name is not None:
            if self._import_path is None:
                raise Exception("_class_name is not None, but _import_path is None. "
                                "If _class_name is define, _import_path must be define too")
            self._config_kwargs, save_kwargs = self._set_defaults()
        setattr(self, CONFIG_OBJ, self.make_config_by_pattern(save_kwargs))

    def __getattribute__(
            self,
            item: Union[str, Type]
    ) -> Any:
        """
        Delegate attribute access to the wrapped config_obj when appropriate.

        Args:
            item (Union[str, Type]): Attribute name to look up.

        Returns:
            Value from the wrapped config_obj if it contains item, otherwise from self.
        """
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
        """
        Delegate writes to config_obj keys; fall back to GeneralConfig for others.

        Args:
            key (str): Attribute name to set.
            value (Any): Value to assign.
        """
        if (hasattr(self, CONFIG_OBJ) and hasattr(getattr(self, CONFIG_OBJ), self._CONFIG_KEYS) and
                key in getattr(getattr(self, CONFIG_OBJ), self._CONFIG_KEYS)):
            getattr(self, CONFIG_OBJ).__setattr__(key, value)
        else:
            super().__setattr__(key, value)

    def _set_defaults(
            self
    ) -> Tuple[dict, dict]:
        """
        Apply default parameter values from the import-path JSON file.

        Returns:
            Tuple of (init_kwargs, save_kwargs) with defaults filled in.
        """
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
        config_class = import_by_name(self._config_class, ["gnn_aid.data_structures.configs"])
        config_obj = config_class(save_kwargs=save_kwargs, **self._config_kwargs)
        return config_obj

    def create_obj(
            self,
            **kwargs
    ) -> Any:
        """
        Instantiate the target class using config values and additional kwargs.

        Args:
            **kwargs: Extra keyword arguments forwarded to the class constructor,
                supplementing those from config_obj.

        Returns:
            New instance of the class named by _class_name.

        Raises:
            Exception: If _class_name is None.
            TypeError: If the class cannot be constructed with the given kwargs.
        """
        obj = None
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
            config: Union[Self, GeneralConfig]
    ) -> Self:
        """
        Merge another config into this ConfigPattern's wrapped config object.

        Args:
            config (Union[Self, GeneralConfig]): Config to merge; may be a ConfigPattern
                or a GeneralConfig.

        Returns:
            Self with the wrapped config updated.
        """
        self_config_obj = getattr(self, CONFIG_OBJ)
        if config.__class__.__name__ == "ConfigPattern":
            config_obj = getattr(config, CONFIG_OBJ)
            setattr(self, CONFIG_OBJ, self_config_obj.merge(config_obj))
        else:
            setattr(self, CONFIG_OBJ, self_config_obj.merge(config))
        return self

    def to_savable_dict(
            self,
            compact: bool = False,
            need_full: bool = True,
            **kwargs
    ) -> dict:
        """
        Build a sorted, deep-copy dict of all config values ready for serialization.

        Args:
            compact (bool): If True, outer dict values are strings without spaces. Default value: `False`.
            need_full (bool): If False, return only the inner _config_kwargs dict. Default value: `True`.

        Returns:
            Sorted serializable dict; if need_full is False, only the inner kwargs dict.
        """
        if CONFIG_SAVE_KWARGS_KEY in self.__dict__ and self.__dict__[
            CONFIG_SAVE_KWARGS_KEY] is not None:
            kw = self.__dict__[CONFIG_SAVE_KWARGS_KEY]
        else:
            kw = dict(filter(lambda x: x[0] in self._TECHNICAL_KEYS, self.__dict__.items()))
            kw["_config_kwargs"] = getattr(self, CONFIG_OBJ).to_savable_dict(compact=compact)
        # BUG Kirill, fix for modification
        if not need_full:
            kw = kw["_config_kwargs"]
        dct = super().to_savable_dict(compact=compact, **kw)
        return dct
