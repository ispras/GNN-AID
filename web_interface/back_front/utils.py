import json
import logging
from collections import deque
from datetime import timedelta
from pathlib import Path
from threading import Thread
from time import sleep
from typing import Any, Union

import numpy as np

from gnn_aid.aux.utils import SAVE_DIR_STRUCTURE_PATH

root_dir = Path(__file__).parent.parent.parent.resolve()  # directory of source root
WEB_DIR = root_dir / "web_interface"
STATIC_DIR = WEB_DIR / "static"  # js, css code
TEMPLATES_DIR = WEB_DIR / "templates"  # html templates
LOG_DIR = WEB_DIR / "logs"  # server logs
CLIENTS_STORAGE_ROOT = LOG_DIR.parent / "client_storage"
CLIENT_STORAGE_TTL = timedelta(days=30)  # client data will be removed this period after last access
CLEANUP_INTERVAL_SEC = 24 * 60 * 60  # how often to check for client storage cleanup
CLIENT_META_FILENAME = ".meta.json"
DIR_PATCH_MODULES = [
    'gnn_aid.aux.utils',
    'gnn_aid.aux.data_info',
    'gnn_aid.aux.declaration',
    'web_interface.back_front.model_blocks',
    'web_interface.back_front.explainer_blocks',
]


class WebInterfaceError(Exception):
    def __init__(
            self,
            *args
    ):
        self.message = args[0] if args else None

    def __str__(
            self
    ):
        if self.message:
            return f"WebInterfaceError: {self.message}"
        else:
            return "WebInterfaceError has been raised!"


class Queue(deque):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(Queue, self).__init__(*args, **kwargs)
        self.last_obl = True

    def push(
            self,
            obj: object,
            id: int,
            obligate: bool
    ) -> None:
        # If last is not obligate - replace it
        if len(self) > 0 and self.last_obl is False:
            self.pop()
        super(Queue, self).append((obj, id))
        self.last_obl = obligate

    def get_first_id(
            self
    ) -> int:
        if len(self) > 0:
            obj, id = self.popleft()
            self.appendleft((obj, id))
            return id
        else:
            return np.inf


class SocketConnect:
    """ Sends messages to JS socket from a python process
    """
    _framework = '?'
    # max_packet_size = 1024**2  # 1MB limit by default

    def __init__(
            self,
    ):
        self.queue = deque()  # general queue
        self.tag_queue = {}  # {tag -> Queue}
        self.obj_id = 0  # Messages ids counter
        self.sleep_time = 0.5
        self.active = False  # True when sending cycle is running

    def send(
            self,
            block: str,
            msg: Union[dict, 'DatasetVarData'],
            func: str = None,
            tag: str = 'all',
            obligate: bool = True
    ):
        """
        Send info message to frontend.

        :param block: destination block, e.g. "" (to console), "model", "explainer"
        :param msg: dict
        :param tag: keep messages in a separate queue with this tag, all but last unobligate
         messages will be squashed
        :param obligate: if not obligate, this message would be replaced by a new one if the queue
         is not empty
        """
        data = {"block": block or "", "msg": msg}
        # data = {"block": block or "", "msg": json_dumps(msg)}
        if func is not None:
            data["func"] = func

        self.queue.append((data, tag, self.obj_id))
        if tag not in self.tag_queue:
            self.tag_queue[tag] = Queue()
        self.tag_queue[tag].push(data, self.obj_id, obligate)  # FIXME tmp
        self.obj_id += 1

        if not self.active:
            print('Thread.start')
            Thread(target=self._cycle, args=()).start()

    def _send(
            self
    ) -> None:
        """ Send leftmost actual data element from the queue. """
        data = None
        # Find actual data elem
        while len(self.queue) > 0:
            data, tag, id = self.queue.popleft()
            # Check if actual
            if self.tag_queue[tag].get_first_id() <= id:
                break

        if data is None:
            return

        size = len(str(data))
        if size > 25e7:
            raise RuntimeError(f"Too big package size: {size} bytes")
        self._send_data(data)
        self.sleep_time = 0.5 * size / 25e6 * 10
        print('sent data', id, tag, 'of len=', size, 'sleep', self.sleep_time)

    def _cycle(
            self
    ) -> None:
        """ Send messages from the queue until it is empty. """
        self.active = True
        while True:
            if len(self.queue) == 0:
                self.active = False
                break
            self._send()
            sleep(self.sleep_time)

    def _send_data(self, data):
        """ Send data to web socket. """
        # To be implemented in subclass
        raise NotImplementedError


def json_dumps(
        object,
        **kwargs
) -> str:
    """ Dump an object to JSON properly handling values "-Infinity", "Infinity", and "NaN"
    """
    kwargs['ensure_ascii'] = False
    if hasattr(object, 'to_json'):
        string = object.to_json(**kwargs)
    else:
        string = json.dumps(object, **kwargs)
    return string \
        .replace('NaN', '"NaN"') \
        .replace('-Infinity', '"-Infinity"') \
        .replace('Infinity', '"Infinity"')


def json_loads(
        string: str
) -> Any:
    """ Parse JSON string properly handling values "-Infinity", "Infinity", and "NaN"
    """
    c = {"-Infinity": -np.inf, "Infinity": np.inf, "NaN": np.nan}

    def parser(
            arg
    ):
        if isinstance(arg, dict):
            for key, value in arg.items():
                if isinstance(value, str) and value in c:
                    arg[key] = c[value]
        return arg

    return json.loads(string, object_hook=parser)


def get_config_keys(
        object_type: str
) -> list:
    """ Get a list of keys for a config describing an object of the specified type.
    """
    with open(SAVE_DIR_STRUCTURE_PATH) as f:
        save_dir_structure = json.loads(f.read())[object_type]

    return [k for k, v in save_dir_structure.items() if v["add_key_name_flag"] is not None]


def send_epoch_results(
        epochs=None,
        metrics_values=None,
        stats_data=None,
        weights=None,
        loss=None,
        obligate=False,
        socket=None
):
    """
    Send updates to the frontend after a training epoch: epoch, metrics, logits, loss.

    :param weights:
    :param metrics_values: quality metrics (accuracy, F1)
    :param stats_data: model statistics (logits, predictions)
    :param loss: train loss
    """
    # Metrics values, epoch, loss
    if metrics_values:
        metrics_data = {"epochs": epochs}
        if loss:
            metrics_data["loss"] = loss
        metrics_data["metrics_values"] = metrics_values
        socket.send("mt", {"metrics": metrics_data}, tag='model_metrics')
    if weights:
        socket.send("mt", weights, tag='model_weights', obligate=obligate)
    if stats_data:
        socket.send("mt", stats_data, tag='model_stats', obligate=obligate)


def compute_stats_data(
        gen_dataset,
        model_manager,
        predictions: bool = False,
        logits: bool = False
):
    """
    :param gen_dataset: wrapper over the dataset, stores the dataset
     and all meta-information about the dataset
    :param predictions: boolean flag that indicates the need to enter model predictions
     in the statistics for the front
    :param logits: boolean flag that indicates the need to enter model logits
     in the statistics for the front
    :return: dict with model weights. Also function can add in dict model predictions
     and logits
    """
    stats_data = {}

    # Stats: weights, logits, predictions
    if predictions:  # and hasattr(self.gnn, 'get_predictions'):
        predictions = model_manager.run_model(gen_dataset, mask='all', out='predictions')
        stats_data["predictions"] = predictions.detach().cpu().tolist()
    if logits:  # and hasattr(gnn, 'forward'):
        logits = model_manager.run_model(gen_dataset, mask='all', out='logits')
        stats_data["logits"] = logits.detach().cpu().tolist()

    if gen_dataset.dataset_var_config.task.is_edge_level():
        # Convert list ot dict
        edge_label_index = gen_dataset.edge_label_index
        keys = list(zip(*edge_label_index.tolist()))
        stats_data = {key: dict(zip(keys, value))
                      for key, value in stats_data.items()}

    if model_manager.stats_data is None:
        model_manager.stats_data = {}
    model_manager.stats_data.update(**stats_data)

    # Note: we update all stats data at once because it can be requested from frontend during
    # the update
    return stats_data


def get_sid_logger(sid: str | None = None) -> logging.LoggerAdapter:
    base_logger = logging.getLogger("gnn_aid.web")
    return logging.LoggerAdapter(base_logger, {"sid": sid or "-"})
