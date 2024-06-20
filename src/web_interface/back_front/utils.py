import json
from collections import deque
from threading import Thread
from time import sleep

import numpy as np


class WebInterfaceError(Exception):
    def __init__(self, *args):
        self.message = args[0] if args else None

    def __str__(self):
        if self.message:
            return f"WebInterfaceError: {self.message}"
        else:
            return "WebInterfaceError has been raised!"


class Queue(deque):
    def __init__(self, *args, **kwargs):
        super(Queue, self).__init__(*args, **kwargs)
        self.last_obl = True

    def push(self, obj, id, obligate):
        # If last is not obligate - replace it
        if len(self) > 0 and self.last_obl is False:
            self.pop()
        super(Queue, self).append((obj, id))
        self.last_obl = obligate

    def get_first_id(self):
        if len(self) > 0:
            obj, id = self.popleft()
            self.appendleft((obj, id))
            return id
        else:
            return np.inf


class SocketConnect:
    """ Sends messages to JS socket from python process
    """

    # max_packet_size = 1024**2  # 1MB limit by default

    def __init__(self, socket=None):
        if socket is None:
            from flask_socketio import SocketIO
            self.socket = SocketIO(message_queue='redis://')
        else:
            self.socket = socket
        self.queue = deque()  # general queue
        self.tag_queue = {}  # {tag -> Queue}
        self.obj_id = 0  # Messages ids counter
        self.sleep_time = 0.5
        self.active = False  # True when sending cycle is running

    def send(self, block, msg, func=None, tag='all', obligate=True):
        """ Send info message to frontend.
        :param dst: destination, e.g. "" (to console), "model", "explainer"
        :param msg: dict
        :param tag: keep messages in a separate queue with this tag, all but last unobligate
         messages will be squashed
        :param obligate: if not obligate, this message would be replaced by a new one if the queue
         is not empty
        """
        data = {"block": block or "", "msg": json_dumps(msg)}
        if func is not None:
            data["func"] = func

        self.queue.append((data, tag, self.obj_id))
        if tag not in self.tag_queue:
            self.tag_queue[tag] = Queue()
        print('push', tag, self.obj_id, obligate)
        self.tag_queue[tag].push(data, self.obj_id, obligate)  # FIXME tmp
        self.obj_id += 1

        if not self.active:
            Thread(target=self._cycle, args=()).start()

    def _send(self):
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

        self.socket.send(data)
        size = len(json_dumps(data))
        if size > 25e6:
            raise RuntimeError(f"Too big package size: {size} bytes")
        self.sleep_time = 0.5 * size / 25e6 * 10
        print('sent data', id, tag, 'of len=', size, 'sleep', self.sleep_time)

    def _cycle(self):
        """ Send messages from the queue until it is empty. """
        self.active = True
        while True:
            if len(self.queue) == 0:
                self.active = False
                break
            self._send()
            sleep(self.sleep_time)


def json_dumps(object):
    """ Dump an object to JSON properly handling values "-Infinity", "Infinity", and "NaN"
    """
    string = json.dumps(object, ensure_ascii=False)
    return string \
        .replace('NaN', '"NaN"') \
        .replace('-Infinity', '"-Infinity"') \
        .replace('Infinity', '"Infinity"')


def json_loads(string):
    """ Parse JSON string properly handling values "-Infinity", "Infinity", and "NaN"
    """
    c = {"-Infinity": -np.inf, "Infinity": np.inf, "NaN": np.nan}

    def parser(arg):
        if isinstance(arg, dict):
            for key, value in arg.items():
                if isinstance(value, str) and value in c:
                    arg[key] = c[value]
        return arg

    return json.loads(string, object_hook=parser)