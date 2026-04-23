import copy
import json
from enum import Enum
from typing import Union, Any, Type, Tuple

from gnn_aid.aux.utils import (
    OPTIMIZERS_PARAMETERS_PATH, MetaEnum)
from gnn_aid.data_structures.gen_config import CONFIG_SAVE_KWARGS_KEY, DATA_CHANGE_FLAG, \
    GeneralConfig, ConfigPattern


class Task(
    str,
    Enum,
    metaclass=MetaEnum
):
    """
    Enum of supported machine learning tasks.
    """
    NODE_CLASSIFICATION = "node-classification"
    NODE_REGRESSION = "node-regression"
    GRAPH_CLASSIFICATION = "graph-classification"
    GRAPH_REGRESSION = "graph-regression"
    EDGE_PREDICTION = "edge-prediction"
    EDGE_CLASSIFICATION = "edge-classification"
    EDGE_REGRESSION = "edge-regression"

    def __str__(self):
        return self.value

    def is_node_level(self) -> bool:
        return self in [Task.NODE_CLASSIFICATION, Task.NODE_REGRESSION]

    def is_edge_level(self) -> bool:
        return self in [Task.EDGE_CLASSIFICATION, Task.EDGE_REGRESSION, Task.EDGE_PREDICTION]

    def is_graph_level(self) -> bool:
        return self in [Task.GRAPH_CLASSIFICATION, Task.GRAPH_REGRESSION]

    def is_classification(self) -> bool:
        return self in [Task.NODE_CLASSIFICATION, Task.EDGE_CLASSIFICATION, Task.GRAPH_CLASSIFICATION]

    def is_regression(self) -> bool:
        return self in [Task.NODE_REGRESSION, Task.EDGE_REGRESSION, Task.GRAPH_REGRESSION]


class Config(
    GeneralConfig
):
    """
    A set of named, immutable parameters.

    Values can only be set in the constructor. Parameters may be plain values,
    dicts, or nested Config instances. Supports loading default parameters from a
    JSON file and creating a saveable representation of those parameters.
    """

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
            other: Type[Any]
    ) -> bool:
        """
        Compare two configs for equality by type and all config key values.

        Args:
            other (Type[Any]): Object to compare against.

        Returns:
            True if other is the same type and has identical values for all config keys.
        """
        if type(other) is not type(self):
            # TODO Kirill check
            return False
        return all(getattr(self, a) == getattr(other, a) for a in self._config_keys)

    def copy(
            self
    ) -> 'GeneralConfig':
        """
        Return a shallow copy of this config with all config-key values copied.

        Returns:
            New instance of the same class with copied config-key values.
        """
        res = type(self)()
        for k, v in self.__dict__.items():
            # FIXME copy of dict, config
            if k in self._config_keys:
                res.__dict__[k] = copy.copy(v)
        return res

    def merge(
            self,
            config: Union[dict, object]
    ) -> 'Config':
        """
        Create a new config by updating this config's params with the given ones.

        Args:
            config (Union[dict, object]): Override values; must be a dict or the same Config type.

        Returns:
            New config instance of the same type with merged parameters.
        """
        assert isinstance(config, (dict, type(self)))

        if isinstance(config, type(self)):
            config = config.to_dict()

        kwargs = self.to_dict()
        kwargs.update(config.copy())
        return type(self)(**kwargs)

    def to_savable_dict(
            self,
            compact: bool = False,
            **kwargs
    ) -> dict:
        """
        Build a sorted, deep-copy dict of config values ready for serialization.

        Args:
            compact (bool): If True, outer dict values are strings without spaces. Default value: `False`.

        Returns:
            Sorted serializable dict of config values.
        """
        if self.__dict__[CONFIG_SAVE_KWARGS_KEY] is not None:
            kw = self.__dict__[CONFIG_SAVE_KWARGS_KEY]
        else:
            kw = dict(filter(lambda x: x[0] in self._config_keys, self.__dict__.items()))
        dct = super().to_savable_dict(compact=compact, **kw)
        return dct


class DatasetConfig(
    Config
):
    """
    A set of distinguishing characteristics that identify a dataset or family of datasets.
    Determines the path to the raw data file in the inner storage.
    """

    def __init__(
            self,
            full_name: Tuple[str, ...] = None
    ):
        assert len(full_name) >= 2
        super().__init__(full_name=full_name)

    @property
    def full_name(self):
        return self["full_name"]

    def path(
            self
    ) -> str:
        import os
        return os.sep.join(self.full_name)


class FeatureConfig(Config):
    """
    Instructions for forming node, edge, and graph features from attributes and structure.
    """
    one_hot = "one_hot"
    degree = "degree"
    clustering = "clustering"
    ten_ones = "10-ones"

    def __init__(
            self,
            node_struct: Union[str, list, dict] = None,
            node_attr: Union[str, list, dict] = None,
            edge_attr: Union[str, list, dict] = None,
            graph_attr: Union[str, list, dict] = None,
            **kwargs
    ):
        super().__init__(node_struct=node_struct,
                         node_attr=node_attr, edge_attr=edge_attr, graph_attr=graph_attr, **kwargs)

    @property
    def node_struct(
            self
    ) -> Union[str, list, dict]:
        return self["node_struct"]

    @property
    def node_attr(
            self
    ) -> Union[str, list, dict]:
        return self["node_attr"]

    @property
    def edge_attr(
            self
    ) -> Union[str, list, dict]:
        return self["edge_attr"]

    @property
    def graph_attr(
            self
    ) -> Union[str, list, dict]:
        return self["graph_attr"]

    def __len__(
            self
    ) -> int:
        """ Sum of feature elements. NOTE: this is not the length of the final feature vector.
        """
        res = 0
        for key in ["node_struct", "node_attr", "edge_attr", "graph_attr"]:
            item = self[key]
            if item is None:
                continue
            if isinstance(item, str):
                res += 1
            elif isinstance(item, list):
                res += len(item)
            else:
                res += len(item)
        return res


class DatasetVarConfig(Config):
    """
    Description of how to produce tensors for a dataset given a DatasetConfig.
    Specifies the path to the tensor file in the inner storage.
    """

    def __init__(
            self,
            task: Task = None,
            labeling: Union[str, dict] = None,
            features: FeatureConfig = None,
            dataset_ver_ind: int = None,
            **kwargs
    ):
        super().__init__(
            task=task, labeling=labeling, features=features, dataset_ver_ind=dataset_ver_ind, **kwargs)

    @property
    def task(
            self
    ) -> Task:
        return self["task"]

    @property
    def labeling(
            self
    ) -> Union[str, dict]:
        return self["labeling"]

    @property
    def features(
            self
    ) -> FeatureConfig:
        return self["features"]

    @property
    def dataset_ver_ind(
            self
    ) -> int:
        return self["dataset_ver_ind"]


class ModelStructureConfig(
    Config
):
    """
    Full description of a GNN model structure as a list of layer blocks.

    Each block can specify a graph layer, batch normalization, activation, dropout,
    and skip connections to other layers. Access by index or iteration behaves like a list.

    General principle for describing one layer of the network:

    .. code-block:: python

        structure=[
            {
                'label': 'n' or 'g',
                'layer': { ... },
                'batchNorm': { ... },
                'activation': { ... },
                'dropout': { ... },
                'connections': [ { ... }, ... ]
            },
        ]

    For connections now support variant connection between layers
    with labels: n -> n, n -> g, g -> g.
    Example connections:

    .. code-block:: python

        'connections': [
            {
                'into_layer': 3,
                'connection_kwargs': {
                    'pool': { 'pool_type': 'global_add_pool' },
                    'aggregation_type': 'cat',
                },
            },
        ]

    into_layer: layer (block) index, numeration starts from 0.
    For aggregation_type only 'cat' is currently supported.
    pool_type is taken from torch_geometric.nn pooling.

    In the case of GINConv, provide an internal nn.Sequential structure via 'gin_seq'.
    For a complete example see the project documentation.
    """
    layers: Any

    def __init__(
            self,
            layers=None
    ):
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
    """
    Config for a GNN model. Can contain structure (for framework models) and/or
    additional parameters (for custom models).
    """

    def __init__(
            self,
            structure: Union[dict, ModelStructureConfig] = None,
            **kwargs
    ):
        """
        Args:
            structure (Union[dict, ModelStructureConfig]): Model layer structure;
                dicts are automatically converted to ModelStructureConfig.
        """
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
    Represents the model attack type and the instance version index.
    """
    _mutable = True

    def __init__(
            self,
            model_ver_ind: int | None = None,
            epochs=None,
            **kwargs
    ):
        """
        Args:
            model_ver_ind (int | None): Model index when saving. If None, takes the
                nearest free index starting from 0. Default value: `None`.
            epochs: Number of training epochs.
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
        """ Read and reset the data-change flag; returns True if any field was modified since last check.
        """
        loc = self.__dict__[DATA_CHANGE_FLAG]
        self.__dict__[DATA_CHANGE_FLAG] = False
        return loc


class EvasionAttackConfig(
    Config
):
    """
    Configuration for an evasion attack. Mutable to allow runtime updates.
    """
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
    """
    Configuration for an evasion defense. Mutable to allow runtime updates.
    """
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
    """
    Configuration for a poison attack. Mutable to allow runtime updates.
    """
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
    """
    Configuration for a poison defense. Mutable to allow runtime updates.
    """
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
    """
    Configuration for a membership inference attack. Mutable to allow runtime updates.
    """
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
    """
    Configuration for a membership inference defense. Mutable to allow runtime updates.
    """
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
    Configuration for explainer initialization parameters.
    """

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(
            **kwargs,
        )


class ExplainerRunConfig(
    Config
):
    """
    Configuration for explainer run parameters.
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
    Variability of an explainer run, including its version index.
    """

    def __init__(
            self,
            explainer_ver_ind: Union[int, None] = None,
            **kwargs
    ):
        super().__init__(
            explainer_ver_ind=explainer_ver_ind,
            **kwargs
        )


if __name__ == '__main__':
    optimizer_info = ConfigPattern(
        _config_class="Config",
        _class_name="Adam",
        _class_import_info=["torch.optim"],
        _import_path=OPTIMIZERS_PARAMETERS_PATH,
        _config_kwargs={},
    )
    print(optimizer_info)
    print()
