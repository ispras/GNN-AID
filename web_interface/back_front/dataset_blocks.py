import json

from aux.data_info import DataInfo
from aux.utils import TORCH_GEOM_GRAPHS_PATH
from base.datasets_processing import DatasetManager, GeneralDataset, VisiblePart
from aux.configs import DatasetConfig, DatasetVarConfig
from web_interface.back_front.block import Block
from web_interface.back_front.utils import json_dumps, get_config_keys


class DatasetBlock(Block):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.dataset_config = None

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
        self._object = DatasetManager.get_by_config(self.dataset_config)

    def get_stat(
            self,
            stat
    ) -> object:
        return self._object.get_stat(stat)

    def get_index(
            self
    ) -> str:
        DataInfo.refresh_data_dir_structure()
        index = DataInfo.data_parse()

        # Add torch_geom FIXME tmp
        with open(TORCH_GEOM_GRAPHS_PATH, 'r') as f:
            configuration = json.load(f)
            assert len(index.keys) == 3
            for i in configuration['content']:
                for j in configuration['content'][i]:
                    for k in configuration['content'][i][j]:
                        try:
                            index.add((i, j, k))
                        except ValueError: pass

        return json_dumps([index.to_json(), json_dumps('')])

    def set_visible_part(
            self,
            part: dict = None
    ) -> str:
        self._object.set_visible_part(part=part)
        return ''

    def get_dataset_data(
            self,
            part: dict = None
    ) -> dict:
        return self._object.get_dataset_data(part=part)


class DatasetVarBlock(Block):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tag = 'dvc'

        self.gen_dataset: GeneralDataset = None  # FIXME duplication!!!
        self.dataset_var_config = None

    def _init(
            self,
            dataset: GeneralDataset
    ) -> dict:
        self.gen_dataset = dataset
        return self.gen_dataset.info.to_dict()

    def _finalize(
            self
    ) -> bool:
        if set(get_config_keys("data_prepared")) != set(self._config.keys()):
            return False

        self.dataset_var_config = DatasetVarConfig(**self._config)
        return True

    def _submit(
            self
    ) -> None:
        self.gen_dataset.build(self.dataset_var_config)
        self._object = self.gen_dataset
        # NOTE: we need to compute var_data to be able to get is_one_hot_able()
        self.gen_dataset.get_dataset_var_data()
        self._result = [self.dataset_var_config.labeling, self.gen_dataset.is_one_hot_able()]

    def get_dataset_var_data(
            self,
            part: dict = None
    ) -> dict:
        return self._object.get_dataset_var_data(part=part)
