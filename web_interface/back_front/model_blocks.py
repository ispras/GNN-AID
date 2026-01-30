import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Union

import torch
from torch_geometric.data import Dataset

from gnn_aid.aux.data_info import UserCodeInfo, DataInfo
from gnn_aid.aux.declaration import Declare
from gnn_aid.aux.prefix_storage import FixedKeysPrefixStorage
from gnn_aid.aux.utils import (
    import_by_name, model_managers_info_by_names_list,
    TECHNICAL_PARAMETER_KEY, IMPORT_INFO_KEY, DATASETS_DIR, ProgressBar)
from gnn_aid.data_structures import Task
from gnn_aid.data_structures.configs import (
    ModelStructureConfig, ModelConfig, ModelModificationConfig, ModelManagerConfig)
# from gnn_aid.datasets.visible_part import VisiblePart, ViewPoint
from gnn_aid.models_builder.gnn_constructor import FrameworkGNNConstructor, GNNConstructor
from gnn_aid.models_builder.models_utils import Metric
from gnn_aid.models_builder.model_managers import GNNModelManager
from . import VisiblePart, ViewPoint, DatasetVarData
from .block import Block, WrapperBlock
from .utils import WebInterfaceError, json_dumps, get_config_keys, send_epoch_results
from .visible_part import add_into_dvd

TENSOR_SIZE_LIMIT = 1024  # Max size of weights tensor we sent to frontend


class ModelWBlock(WrapperBlock):
    def __init__(
            self,
            name: str,
            blocks: [Block],
            *args,
            **kwargs
    ):
        super().__init__(blocks, name, *args, **kwargs)

    def _init(
            self,
            visible_part: VisiblePart
    ) -> list[int]:
        gen_dataset = visible_part.gen_dataset
        return [gen_dataset.num_node_features, gen_dataset.num_classes]

    def _finalize(
            self
    ) -> bool:
        return True

    def _submit(
            self
    ) -> None:
        pass


class ModelLoadBlock(Block):
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
            visible_part: VisiblePart
    ) -> list[str]:
        self.visible_part = visible_part
        self.gen_dataset = visible_part.gen_dataset
        return self.get_index()

    def _finalize(
            self
    ) -> bool:
        # fixme
        if set(get_config_keys("models")) != set(self._config.keys()):
            return False

        self.model_path = self._config
        return True

    def _submit(
            self
    ) -> None:
        from gnn_aid.models_builder.model_managers.model_manager import GNNModelManager
        self.model_manager, train_test_split_path = GNNModelManager.from_model_path(
            model_path=self.model_path, dataset_path=self.gen_dataset.prepared_dir)
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
        path, files_paths = Declare.dataset_prepared_dir(
            self.gen_dataset.dataset_config,
            self.gen_dataset.dataset_var_config
        )
        path = os.path.relpath(path, DATASETS_DIR)
        keys_list, full_keys_list, dir_structure, _ = DataInfo.take_keys_etc_by_prefix(
            prefix=("datasets",)
        )
        values_info = DataInfo.values_list_by_path_and_keys(
            path=path, full_keys_list=full_keys_list, dir_structure=dir_structure)
        ps = index.filter(values_info)
        # ps = index.filter(dict(zip(keys_list, values_info)))
        return [ps.to_json(), json_dumps(info)]

    def _load_train_test_mask(
            self,
            path: Union[Path, str]
    ) -> None:
        """ Load train/test mask associated to the model and send to frontend """
        # FIXME self.manager_config.train_test_split
        self.gen_dataset.train_mask, self.gen_dataset.val_mask, self.gen_dataset.test_mask, train_test_split = torch.load(path)[:]
        dvd = get_train_test_mask(self.gen_dataset, self.visible_part)
        self.socket.send(block='mload', msg=dvd)


class ModelConstructorBlock(Block):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.model_config = None

    def _init(
            self,
            visible_part: VisiblePart
    ) -> list:
        gen_dataset = visible_part.gen_dataset
        return [gen_dataset.num_node_features, gen_dataset.num_classes,
                gen_dataset.is_multi(), gen_dataset.dataset_var_config.task]

    def _finalize(
            self
    ) -> bool:
        # TODO better check
        if not ('layers' in self._config and isinstance(self._config['layers'], list)):
            return False

        self.model_config = ModelConfig(structure=ModelStructureConfig(**self._config))
        return True

    def _submit(
            self
    ) -> None:
        self._object = FrameworkGNNConstructor(self.model_config)
        self._result = self._object.get_full_info(tensor_size_limit=TENSOR_SIZE_LIMIT)


class ModelCustomBlock(Block):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.gen_dataset = None
        self.model_name: dict = None

    def _init(
            self,
            visible_part: VisiblePart
    ) -> list[str]:
        self.gen_dataset = visible_part.gen_dataset
        return self.get_index()

    def _finalize(
            self
    ) -> bool:
        if not (len(self._config.keys()) == 2):  # TODO better check
            return False

        self.model_name = self._config
        return True

    def _submit(
            self
    ) -> None:
        # FIXME misha this is bad way
        user_models_obj_dict_info = UserCodeInfo.user_models_list_ref()
        cm_path = None

        for klass, content in user_models_obj_dict_info.items():
            for name in content["obj_names"]:
                if klass == self.model_name["class"] and name == self.model_name["model"]:
                    cm_path = content["import_path"]
                    break

        assert cm_path

        self._object = UserCodeInfo.take_user_model_obj(cm_path, self.model_name["model"])
        self._result = self._object.get_full_info(tensor_size_limit=TENSOR_SIZE_LIMIT)

    def get_index(
            self
    ) -> list[str]:
        """ Get all available models with respect to current dataset
        """
        user_models_obj_dict_info = UserCodeInfo.user_models_list_ref()
        ps = FixedKeysPrefixStorage(["class", "model"])
        for key, content in user_models_obj_dict_info.items():
            for value in content["obj_names"]:
                ps.add([key, value])
        index = ps

        # FIXME apply dataset filter
        # cfg = self.gen_dataset.dataset_config.to_saveable_dict()
        # cfg.update(self.gen_dataset.dataset_var_config.to_saveable_dict())
        # ps = index.filter(cfg)
        return [ps.to_json(), json_dumps(None)]


class ModelManagerBlock(Block):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.model_manager_config = None
        self.klass = None

    def _init(
            self,
            visible_part: VisiblePart,
            gnn: GNNConstructor
    ) -> Union[dict, list]:
        # Define options for model manager
        self.visible_part = visible_part
        self.gen_dataset = visible_part.gen_dataset
        self.gnn = gnn

        mm_set = self.gnn.suitable_model_managers()
        # mm_set.add("_DummyModelManager")
        if len(mm_set) == 0:  # FIXME is it ok for custom model?
            mm_set.add("FrameworkGNNModelManager")
        mm_info = model_managers_info_by_names_list(mm_set)
        return [self.gen_dataset.dataset_var_config.task, mm_info]

    def _finalize(
            self
    ) -> bool:
        self.klass = self._config.pop("class")
        self.model_manager_config = ModelManagerConfig(**self._config)
        return True

    def _submit(
            self
    ) -> None:
        create_train_test_mask = True
        assert self.gnn is not None

        # Import correct class
        from web_interface.back_front.frontend_client import FrontendClient
        if self.klass in FrontendClient.get_parameters("FW"):
            mm_class = import_by_name(self.klass, ["gnn_aid.models_builder.model_managers.framework_mm"])

        else:  # Custom MM
            mm_info = model_managers_info_by_names_list({self.klass})
            mm_class = import_by_name(
                self.klass, [mm_info[self.klass][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY]])

        # Build model manager
        self._object = mm_class(
            gnn=self.gnn,
            manager_config=self.model_manager_config,
            dataset_path=self.gen_dataset.prepared_dir,
            modification=ModelModificationConfig(
                model_ver_ind=0,
                # FIXME Kirill front attack
                epochs=0,
            )
        )

        self._result = self._object.get_full_info()

        # Create and send train_test_mask
        if create_train_test_mask:
            self.gen_dataset.train_test_split(*self.model_manager_config.train_test_split)
            dvd = get_train_test_mask(self.gen_dataset, self.visible_part)
            self.socket.send(block='mmc', msg=dvd)

    def get_satellites(
            self,
            view_point: ViewPoint
    ) -> DatasetVarData:
        """ Get model dependent satellites data: train-test mask, embeds, preds
        """
        self.visible_part.update_view_point(view_point)

        task = self.gen_dataset.dataset_var_config.task
        dvd = get_train_test_mask(self.gen_dataset, self.visible_part)

        if self._object.stats_data is not None:
            stats_data = {k: self.visible_part.filter(v, task)
                          for k, v in self._object.stats_data.items()}
            dvd = add_into_dvd(self.gen_dataset, stats_data, dvd)
        return dvd


# TODO move to front client?
def get_train_test_mask(
        gen_dataset,
        visible_part: VisiblePart = None
) -> Union[None, DatasetVarData]:
    """ Get train/val/test mask for the dataset and send to frontend.
    """
    # Encode mask as train=1, val=2, test=3
    train_test_mask = [0] * len(gen_dataset.train_mask)
    for n in range(len(train_test_mask)):
        if gen_dataset.train_mask[n]:
            train_test_mask[n] = 1
        elif gen_dataset.test_mask[n]:
            train_test_mask[n] = 2
        elif gen_dataset.val_mask[n]:
            train_test_mask[n] = 3

    # Filter
    msg = {"train-test-mask": visible_part.filter(train_test_mask)}
    return add_into_dvd(gen_dataset, msg)


class ModelTrainerBlock(Block):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.gen_dataset: GeneralDataset = None
        self.model_manager = None
        self.metrics = None

        # Copy of the dataset before attacks applied.
        self._gen_dataset_backup: GeneralDataset = None

    def _init(
            self,
            visible_part: VisiblePart,
            gmm: GNNModelManager
    ) -> Union[str, dict]:
        self.visible_part = visible_part
        self.gen_dataset = visible_part.gen_dataset
        self.model_manager = gmm

        # Inject hooks
        self.model_manager.set_hook(self._after_epoch_hook, 'after_epoch')

        # fixme misha do we need it?
        # return self.model_manager.get_model_data()
        return self.gen_dataset.dataset_var_config.task

    def _report(self):
        """ Called when model training epoch changes: update or reset
        """
        msg = {}
        msg.update(self.pbar.kwargs)
        msg.update({
            "progress": {
                "text": f'{self.pbar.n} of {self.pbar.total}',
                "load": self.pbar.n / self.pbar.total if self.pbar.total > 0 else 1
            }})
        self.socket.send(block='mt', msg=msg, tag='mt' + '_progress', obligate=True)

    def _after_epoch_hook(
            self,
            train_loss,
    ):
        metrics_values = self.model_manager.evaluate_model(
            gen_dataset=self.gen_dataset, metrics=self.metrics)
        self.model_manager.compute_stats_data(
            self.gen_dataset, predictions=True, logits=True)
        stats_data = {k: self.visible_part.filter(
            v, self.gen_dataset.dataset_var_config.task)
                         for k, v in self.model_manager.stats_data.items()}

        # Reformat to DatasetVarData
        dvd = add_into_dvd(self.gen_dataset, stats_data)

        send_epoch_results(
            epochs=self.model_manager.modification.epochs,
            metrics_values=metrics_values,
            stats_data=dvd,
            weights={"weights": self.model_manager.gnn.get_weights()},
            loss=train_loss,
            socket=self.socket)
        self.pbar.update(1)

    def _finalize(
            self
    ) -> bool:
        # TODO for ProtGNN model must be trained

        return True

    def _submit(
            self
    ) -> None:
        self.metrics = [Metric(**m) for m in self._config.get('metrics')]
        self._object = [self.model_manager, self.metrics]
        self._save_model()

        # Make a dataset backup
        if self._gen_dataset_backup is None:
            # Make a dataset backup
            # FIXME This is a bad way - for large datasets very bad. It is a temporary solution
            self._gen_dataset_backup = deepcopy(self.gen_dataset)

    def _unlock(
            self
    ) -> None:
        # Retract changes - reset dataset as before evasion attacks
        # FIXME This is a bad way - for large datasets very bad. It is a temporary solution
        self.gen_dataset = deepcopy(self._gen_dataset_backup)

    def do(
            self,
            do,
            params
    ) -> str:
        if do == "run":
            self.metrics = [Metric(**m) for m in json.loads(params.get('metrics'))]
            self._adjust_metrics()
            self._run_model()
            return ''

        elif do == "reset":
            self._reset_model()
            return ''

        elif do == "train":
            mode = params.get('mode')
            steps = json.loads(params.get('steps'))
            self.metrics = [Metric(**m) for m in json.loads(params.get('metrics'))]
            self._adjust_metrics()

            from threading import Thread
            Thread(target=self._train_model, args=(mode, steps)).start()
            # NOTE: we need to return context instantly to avoid main process waiting and blocking
            # emitting messages to frontend
            return ''

        elif do == "save":
            return self._save_model()

        # elif do == "load":
        #     model_path = json.loads(params.get('modelPath'))
        #     print(f"model_path: {model_path}")
        #     model_manager = self.load_model(model_path)
        #     data = json.dumps([
        #         model_manager.model_full_config().to_dict(),
        #         model_manager.get_model_data()])
        #     logging.info(f"Length of model_data: {len(data)}")
        #     return data
        #
        else:
            raise WebInterfaceError(f"Unknown 'do' command {do} for model")

    def _reset_model(
            self
    ) -> None:
        self.model_manager.gnn.reset_parameters()
        self.model_manager.modification.epochs = 0
        self.gen_dataset.train_test_split(*self.model_manager.manager_config.train_test_split)
        dvd = get_train_test_mask(self.gen_dataset, self.visible_part)
        self.socket.send(block='mt', msg=dvd)
        self._run_model()

    def _run_model(
            self
    ) -> None:
        """ Runs model to compute predictions and logits """
        # TODO add set of nodes
        assert self.model_manager
        # from gnn_aid.models_builder.models_utils import Metric
        metrics_values = self.model_manager.evaluate_model(
            self.gen_dataset, metrics=self.metrics)
        self.model_manager.compute_stats_data(self.gen_dataset, predictions=True, logits=True)

        # fixme here we can want to obtain predicts for nodes and graphs both
        stats_data = {k: self.visible_part.filter(v)
                      for k, v in self.model_manager.stats_data.items()}
        # Reformat to DatasetVarData
        dvd = add_into_dvd(self.gen_dataset, stats_data)

        send_epoch_results(
            metrics_values=metrics_values, stats_data=dvd, socket=self.socket)

    def _train_model(
            self,
            mode: Union[str, None],
            steps: Union[int, None]
    ) -> None:

        self.pbar = ProgressBar()
        self.pbar.set_hook(self._report, 'on_reset')
        self.pbar.set_hook(self._report, 'on_update')

        self.pbar.total = self.model_manager.modification.epochs + steps
        self.pbar.n = self.model_manager.modification.epochs
        self.pbar.update(0)

        apply_posisoning_ad = True if self.model_manager.modification.epochs == 0 else False
        try:
            self.model_manager.train_model(
                gen_dataset=self.gen_dataset, save_model_flag=False,
                mode=mode, steps=steps, metrics=self.metrics,
            apply_posisoning_ad=apply_posisoning_ad)

            self.pbar.close()
            self.socket.send("mt", {"status": "OK", "info": "training-finished"})

        except Exception as e:
            self.socket.send("mt", {"status": "FAILED"})
            raise e

    def _save_model(
            self
    ) -> str:
        path = self.model_manager.save_model_executor()
        self.gen_dataset.save_train_test_mask(path)
        DataInfo.refresh_models_dir_structure()
        # TODO send dir_structure info to front
        return str(path)

    def _adjust_metrics(
            self
    ) -> None:
        """
        Adjust metrics parameters if dataset has many classes, e.g. binary -> macro averaging.
        Helper function until frontend supports metric kwargs.
        """
        if self.gen_dataset.num_classes > 2:  # Binary -> macro averaging
            for m in self.metrics:
                if m._name in ['F1', 'Recall', 'Precision', 'Jaccard']:
                    avg = m.kwargs.get('average', 'binary')
                    if avg == 'binary':
                        m.kwargs['average'] = 'macro'

        if self.gen_dataset.dataset_var_config.task == Task.EDGE_PREDICTION:
            for m in self.metrics:
                if '@' in m._name:
                    name, k = m._name.split('@')
                    m._name = name + '@k'
                    m.kwargs['k'] = int(k)
