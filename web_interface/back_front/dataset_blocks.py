import json
from typing import Tuple

from gnn_aid.aux.data_info import DataInfo
from gnn_aid.aux.utils import TORCH_GEOM_GRAPHS_PATH
from gnn_aid.data_structures import Task
from gnn_aid.data_structures.configs import DatasetConfig, DatasetVarConfig, FeatureConfig
from gnn_aid.datasets.datasets_manager import DatasetManager
from gnn_aid.datasets.gen_dataset import GeneralDataset
from . import DatasetData, DatasetVarData, VisiblePart, ViewPoint
from .block import Block
from .utils import json_dumps, get_config_keys


class DatasetBlock(Block):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.dataset_config = None
        self.gen_dataset: GeneralDataset = None

        from gnn_aid.aux.prefix_storage import TuplePrefixStorage
        self._index = None
        with open(TORCH_GEOM_GRAPHS_PATH, 'r') as f:
            self._torch_geom_index = TuplePrefixStorage.from_json(f.read())

    def _init(
            self
    ) -> None:
        pass

    def _finalize(
            self
    ) -> bool:
        if set(get_config_keys("data_root")) != set(self._config.keys()):
            return False

        self.dataset_config = DatasetConfig(**self._config)

        return True

    def _submit(
            self
    ) -> None:
        self.gen_dataset = DatasetManager.get_by_config(self.dataset_config)
        self._object = self.gen_dataset

    def get_stat(
            self,
            stat
    ) -> object:
        return self.gen_dataset.get_stat(stat)

    def get_index(
            self
    ) -> str:
        DataInfo.refresh_data_dir_structure()
        self._index = DataInfo.data_parse()

        # Add torch_geom
        from gnn_aid.datasets.ptg_datasets import LibPTGDataset
        # assert len(index.keys) == 3
        for key, value in self._torch_geom_index:
            if key[0] == "Heterogeneous":
                continue  # Not implemented
            if value is not None and key not in self._index:
                continue  # Requires creation from backend
            try:
                self._index.add([LibPTGDataset.data_folder] + key, "Not loaded yet")
            except KeyError: pass

        return json_dumps([self._index.to_json(), json_dumps('')])

    # def get_dataset_data(
    #         self,
    #         view_point: ViewPoint
    # ) -> DatasetData:
    #     return self.visible_part.get_dataset_data(view_point)
    #     # return self._object.visible_part.get_dataset_data(part=part)


class DatasetVarBlock(Block):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tag = 'dvc'

        self.gen_dataset: GeneralDataset = None  # FIXME misha duplication!!!
        self.dataset_var_config = None

    def _init(
            self,
            gen_dataset: GeneralDataset
            # dataset_and_vp: Tuple[GeneralDataset, VisiblePart]
    ) -> dict:
        self.gen_dataset = gen_dataset
        self.visible_part = None
        return self.gen_dataset.info.to_dict()

    def _finalize(
            self
    ) -> bool:
        # if set(get_config_keys("data_prepared")) != set(self._config.keys()):
        #     return False

        kwargs = self._config.copy()
        features = FeatureConfig(**kwargs.pop('features'))
        kwargs['features'] = features
        task = Task(kwargs.pop('task'))
        kwargs['task'] = task
        self.dataset_var_config = DatasetVarConfig(**kwargs)
        # print(self.dataset_var_config.to_dict())
        return True

    def _submit(
            self
    ) -> None:
        self.gen_dataset.build(self.dataset_var_config)
        assert self.visible_part is not None
        self._object = self.visible_part
        # NOTE: we need to compute var_data to be able to get is_one_hot_able()
        self.visible_part.get_dataset_var_data()
        one_hot_able = is_one_hot_able(self.gen_dataset) if self.gen_dataset.is_multi() else True
        self._result = [self.dataset_var_config.task, self.dataset_var_config.labeling, one_hot_able]

    def set_visible_part(
            self,
            view_point: ViewPoint
    ) -> str:
        self.visible_part = VisiblePart(view_point, self.gen_dataset)
        # self._object.set_visible_part(part=part)
        return ''

    # def get_dataset_var_data(
    #         self,
    #         view_point: ViewPoint
    # ) -> DatasetVarData:
    #     return self.visible_part.get_dataset_var_data(view_point)


def is_one_hot_able(dataset: GeneralDataset) -> bool:
    """ Return whether features are 1-hot encodings. If yes, nodes can be colored.
    """
    assert dataset.dataset_var_config

    if dataset.is_multi():
        res = False
        feature_config: FeatureConfig = dataset.dataset_var_config.features
        if len(feature_config) == 1:
            # 1-hot over nodes and no attributes is OK
            if feature_config.node_struct == [FeatureConfig.one_hot]:
                return True

            # Only 1 categorical attr is OK
            if feature_config.node_attr:
                # Try to get attribute name
                try:
                    item = feature_config.node_attr
                    if isinstance(item, list):
                        assert len(item) == 1
                        item = item[0]
                    elif isinstance(item, dict):
                        item = list(item.keys())[0]
                    assert isinstance(item, str)
                except (AssertionError, KeyError):
                    return False

                attr = item

                if attr in dataset.info.node_attributes['names']:
                    ix = dataset.info.node_attributes['names'].index(attr)
                    type = dataset.info.node_attributes['types'][ix]
                    if type == 'categorical':
                        return True
                    if type in ['vector', 'other']:
                        # Check honestly each feature vector
                        feats = dataset.node_features
                        res = all(all(all(x == 1 or x == 0 for x in f) for f in feat)
                                  for feat in feats) and\
                              all(all(sum(f) == 1 for f in feat) for feat in feats)
                        return res
    else:
        # We do not need it
        raise NotImplementedError

    return False
