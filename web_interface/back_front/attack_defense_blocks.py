import json
import os
from pathlib import Path
from typing import Union

import torch
from torch_geometric.data import Dataset

from aux.configs import ModelStructureConfig, ModelConfig, ModelModificationConfig
from aux.data_info import UserCodeInfo, DataInfo
from aux.declaration import Declare
from aux.prefix_storage import PrefixStorage
from aux.utils import import_by_name, model_managers_info_by_names_list, GRAPHS_DIR, \
    TECHNICAL_PARAMETER_KEY, \
    IMPORT_INFO_KEY
from base.datasets_processing import GeneralDataset, VisiblePart
from models_builder.gnn_constructor import FrameworkGNNConstructor, GNNConstructor
from models_builder.gnn_models import ModelManagerConfig, GNNModelManager, Metric
from web_interface.back_front.block import Block, WrapperBlock
from web_interface.back_front.utils import WebInterfaceError, json_dumps, get_config_keys, \
    SocketConnect


class BeforeTrainBlock(Block):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.model_path = None
        self.gen_dataset = None

    def _init(
            self,
            gen_dataset: GeneralDataset
    ) -> list[str]:
        self.gen_dataset = gen_dataset
        return self.get_index()

    def _finalize(
            self
    ) -> bool:
        if set(get_config_keys("models")) != set(self._config.keys()):
            return False

        self.model_path = self._config
        return True

    def _submit(
            self
    ) -> None:
        from models_builder.gnn_models import GNNModelManager
        self.model_manager, train_test_split_path = GNNModelManager.from_model_path(
            model_path=self.model_path, dataset_path=self.gen_dataset.results_dir)
        self._load_train_test_mask(train_test_split_path / 'train_test_split')

        self._object = self.model_manager
        self._result = self._object.get_full_info()
        self._result.update(self._object.gnn.get_full_info(tensor_size_limit=TENSOR_SIZE_LIMIT))

    def get_index(
            self
    ) -> list[str]:
        """ Get all available models with respect to current dataset
        """
        DataInfo.refresh_models_dir_structure()
        index, info = DataInfo.models_parse()
        path, files_paths = Declare.dataset_prepared_dir(self.gen_dataset.dataset_config,
                                                         self.gen_dataset.dataset_var_config)
        path = os.path.relpath(path, GRAPHS_DIR)
        keys_list, full_keys_list, dir_structure, _ = DataInfo.take_keys_etc_by_prefix(
            prefix=("data_root", "data_prepared")
        )
        values_info = DataInfo.values_list_by_path_and_keys(
            path=path, full_keys_list=full_keys_list, dir_structure=dir_structure)
        ps = index.filter(dict(zip(keys_list, values_info)))
        return [ps.to_json(), json_dumps(info)]

    def _load_train_test_mask(
            self,
            path: Union[Path, str]
    ) -> None:
        """ Load train/test mask associated to the model and send to frontend """
        # FIXME self.manager_config.train_test_split
        self.gen_dataset.train_mask, self.gen_dataset.val_mask, \
        self.gen_dataset.test_mask, train_test_split = torch.load(path)[:]
        send_train_test_mask(self.gen_dataset, self.socket)

