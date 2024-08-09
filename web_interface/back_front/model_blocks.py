import json
import os

import torch
from torch_geometric.data import Dataset

from aux.configs import ModelStructureConfig, ModelConfig, ModelModificationConfig
from aux.data_info import UserCodeInfo, DataInfo
from aux.declaration import Declare
from aux.prefix_storage import PrefixStorage
from aux.utils import import_by_name, model_managers_info_by_names_list, GRAPHS_DIR, TECHNICAL_PARAMETER_KEY, \
    IMPORT_INFO_KEY
from base.datasets_processing import GeneralDataset, VisiblePart
from models_builder.gnn_constructor import FrameworkGNNConstructor
from models_builder.gnn_models import ModelManagerConfig, GNNModelManager, Metric
from web_interface.back_front.block import Block, WrapperBlock
from web_interface.back_front.utils import WebInterfaceError, json_dumps


class ModelWBlock(WrapperBlock):
    def __init__(self, name, blocks, *args, **kwargs):
        super().__init__(blocks, name, *args, **kwargs)

    def _init(self, ptg_dataset: Dataset):
        return [ptg_dataset.num_node_features, ptg_dataset.num_classes]

    def _finalize(self):
        return True

    def _submit(self):
        pass


class ModelLoadBlock(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_path = None
        self.gen_dataset = None

    def _init(self, gen_dataset: GeneralDataset):
        self.gen_dataset = gen_dataset
        return self.get_index()

    def _finalize(self):
        if not (len(self._config.keys()) == 5):  # TODO better check
            return False

        self.model_path = self._config
        return True

    def _submit(self):
        from models_builder.gnn_models import GNNModelManager
        self.model_manager, train_test_split_path = GNNModelManager.from_model_path(
            model_path=self.model_path, dataset_path=self.gen_dataset.results_dir)
        self._load_train_test_mask(train_test_split_path / 'train_test_split')

        self._object = self.model_manager
        self._result = self._object.get_full_info()
        self._result.update(self._object.gnn.get_full_info())

    def get_index(self):
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
        values_info = DataInfo.values_list_by_path_and_keys(path=path,
                                                            full_keys_list=full_keys_list,
                                                            dir_structure=dir_structure)
        ps = index.filter(dict(zip(keys_list, values_info)))
        return [ps.to_json(), json_dumps(info)]

    def _load_train_test_mask(self, path):
        """ Load train/test mask associated to the model and send to frontend """
        # FIXME self.manager_config.train_test_split
        self.gen_dataset.train_mask, self.gen_dataset.val_mask, \
        self.gen_dataset.test_mask, train_test_split = torch.load(path)[:]
        send_train_test_mask(self.gen_dataset, self.socket)


class ModelConstructorBlock(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_config = None

    def _init(self, gen_dataset: GeneralDataset):
        ptg_dataset = gen_dataset.dataset
        return [ptg_dataset.num_node_features, ptg_dataset.num_classes, gen_dataset.is_multi()]

    def _finalize(self):
        # TODO better check
        if not ('layers' in self._config and isinstance(self._config['layers'], list)):
            return False

        self.model_config = ModelConfig(structure=ModelStructureConfig(**self._config))
        return True

    def _submit(self):
        self._object = FrameworkGNNConstructor(self.model_config)
        self._result = self._object.get_full_info()


class ModelCustomBlock(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gen_dataset = None
        self.model_name: dict = None

    def _init(self, gen_dataset: GeneralDataset):
        self.gen_dataset = gen_dataset
        return self.get_index()

    def _finalize(self):
        if not (len(self._config.keys()) == 2):  # TODO better check
            return False

        self.model_name = self._config
        return True

    def _submit(self):
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
        self._result = self._object.get_full_info()

    def get_index(self):
        """ Get all available models with respect to current dataset
        """
        user_models_obj_dict_info = UserCodeInfo.user_models_list_ref()
        ps = PrefixStorage(["class", "model"])
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_manager_config = None
        self.klass = None

    def _init(self, gen_dataset: GeneralDataset, gnn):
        # Define options for model manager
        self.gen_dataset = gen_dataset
        self.gnn = gnn

        mm_set = self.gnn.suitable_model_managers()
        # mm_set.add("_DummyModelManager")
        if len(mm_set) == 0:  # FIXME is it ik for custom model?
            mm_set.add("FrameworkGNNModelManager")
        mm_info = model_managers_info_by_names_list(mm_set)
        return mm_info

    def _finalize(self):
        # if 1:  # TODO better check
        #     return False

        self.klass = self._config.pop("class")
        self.model_manager_config = ModelManagerConfig(**self._config)
        return True

    def _submit(self):
        create_train_test_mask = True
        assert self.gnn is not None

        # Import correct class
        from web_interface.main_multi import FrontendClient
        if self.klass in FrontendClient.get_parameters("FW"):
            mm_class = import_by_name(self.klass, ["models_builder.gnn_models"])

        else:  # Custom MM
            mm_info = model_managers_info_by_names_list({self.klass})
            mm_class = import_by_name(
                self.klass, [mm_info[self.klass][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY]])

        # Build model manager
        self._object = mm_class(
            gnn=self.gnn,
            manager_config=self.model_manager_config,
            dataset_path=self.gen_dataset.results_dir,
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
            send_train_test_mask(self.gen_dataset, self.socket)

    def get_satellites(self, part=None):
        """ Resend model dependent satellites data: train-test mask, embeds, preds
        """
        visible_part = self.gen_dataset.visible_part if part is None else\
            VisiblePart(self.gen_dataset, **part)

        res = {}
        res.update(send_train_test_mask(self.gen_dataset, None, visible_part))
        # TODO duplicste code
        if self._object.stats_data is not None:
            stats_data = {k: visible_part.filter(v)
                          for k, v in self._object.stats_data.items()}
            res.update(stats_data)
        return res


def send_train_test_mask(gen_dataset, socket, visible_part=None):
    """ Compute train/test mask for the dataset and send to frontend.
    """
    if visible_part is None:
        visible_part = gen_dataset.visible_part

    train_test_mask = [0] * len(gen_dataset.train_mask)
    for n in range(len(train_test_mask)):
        if gen_dataset.train_mask[n]:
            train_test_mask[n] = 1
        elif gen_dataset.test_mask[n]:
            train_test_mask[n] = 2
        elif gen_dataset.val_mask[n]:
            train_test_mask[n] = 3
    msg = {"train-test-mask": visible_part.filter(train_test_mask)}
    if socket:
        socket.send(block='mmc', msg=msg)
    else:
        return msg


class ModelTrainerBlock(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gen_dataset = None
        self.model_manager = None

    def _init(self, gen_dataset: GeneralDataset, gmm: GNNModelManager):
        self.gen_dataset = gen_dataset
        self.model_manager = gmm

        return self.model_manager.get_model_data()

    def _finalize(self):
        # TODO for ProtGNN model must be trained

        return True

    def _submit(self):
        self._object = self.model_manager

    def do(self, do, params):
        if do == "run":
            metrics = [Metric(**m) for m in json.loads(params.get('metrics'))]
            self._run_model(metrics)
            return ''

        elif do == "reset":
            self._reset_model()
            return ''

        elif do == "train":
            mode = params.get('mode')
            steps = json.loads(params.get('steps'))
            metrics = [Metric(**m) for m in json.loads(params.get('metrics'))]
            self._train_model(mode=mode, steps=steps, metrics=metrics)
            return ''

        elif do == "stop":
            # BACK_FRONT.model_manager.stop_signal = True  # TODO remove stop_signal
            self.stop_model()
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

    def _reset_model(self):
        self.model_manager.gnn.reset_parameters()
        self.model_manager.modification.epochs = 0
        self.gen_dataset.train_test_split(*self.model_manager.manager_config.train_test_split)
        send_train_test_mask(self.gen_dataset, self.socket)
        self._run_model([Metric("Accuracy", mask='train'), Metric("Accuracy", mask='test')])

    def _run_model(self, metrics):
        """ Runs model to compute predictions and logits """
        # TODO add set of nodes
        assert self.model_manager
        from models_builder.gnn_models import Metric
        self._check_metrics(metrics)
        metrics_values = self.model_manager.evaluate_model(
            self.gen_dataset, metrics=metrics)
        self.model_manager.compute_stats_data(self.gen_dataset, predictions=True, logits=True)

        stats_data = {k: self.gen_dataset.visible_part.filter(v)
                      for k, v in self.model_manager.stats_data.items()}
        self.model_manager.send_epoch_results(
            metrics_values=metrics_values, stats_data=stats_data, socket=self.socket)

    def _train_model(self, mode, steps, metrics):
        # # Remove unpickable socket
        # self.model_manager.gnn_mm.socket = None
        #
        # assert self.model_training_subprocess is None or not self.model_training_subprocess.is_alive()
        #
        # queue = tQueue()
        # self.model_training_subprocess = tProcess(
        #     target=run_function, args=(
        #         self.model_manager, 'train_model', {
        #             "gen_dataset": self.gen_dataset, "save_model_flag": False, "mode": mode,
        #             "steps": steps, "metrics": metrics},
        #         queue))
        #
        # self.model_training_subprocess.start()
        # self.socket.send("model", {"status": "STARTED"})
        # self.model_training_subprocess.join()
        #
        # # Get result if present - otherwise nothing changed
        # self.model_manager = queue.get_nowait()
        # # Put unpickable socket back
        # self.model_manager.gnn_mm.socket = self.socket

        self._check_metrics(metrics)
        self.model_manager.train_model(
            gen_dataset=self.gen_dataset, save_model_flag=False,
            mode=mode, steps=steps, metrics=metrics, socket=self.socket)

    def _save_model(self):
        path = self.model_manager.save_model_executor()
        self.gen_dataset.save_train_test_mask(path)
        DataInfo.refresh_models_dir_structure()
        # TODO send dir_structure info to front
        return str(path)

    def _check_metrics(self, metrics):
        """ Adjust metrics parameters if dataset has many classes, e.g. binary -> macro averaging
        """
        classes = self.gen_dataset.num_classes
        if classes > 2:
            for m in metrics:
                if m.name in ['F1', 'Recall', 'Precision', 'Jaccard']:
                    avg = m.kwargs.get('average', 'binary')
                    if avg == 'binary':
                        m.kwargs['average'] = 'macro'
