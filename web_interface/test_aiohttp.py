import asyncio
from multiprocessing import Queue, Process
from random import random
from time import sleep
from typing import Dict
from aiohttp import web
import socketio
import jinja2
import aiohttp_jinja2

# Socket.IO server (ASGI not used here)
from torch.nn import Linear
from torch_geometric.nn import GCNConv

sio = socketio.AsyncServer(async_mode="aiohttp")
app = web.Application()
sio.attach(app)

aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader("templates"))

clients = {}  # client_id (sid) -> asyncio.Task
process_refs: Dict[str, Process] = {}
queues: Dict[str, Queue] = {}


# Route: Home page
@aiohttp_jinja2.template("test.html")
async def handle_home(request):
    return {}


# Route: Static file serving
app.router.add_static('/static/', path='static', name='static')
app.router.add_get('/', handle_home)


# Route: POST /block
async def handle_block(request):
    data = await request.post()
    sid = data.get("sid")

    if sid not in clients:
        return web.Response(status=404, text="Unknown SID")

    response_queue, _, request_queue = queues[sid]

    print('handler: http request from', sid, 'args', dict(data))
    request_queue.put({"type": 'url', "args": dict(data)})

    # Wait for response from worker
    result = response_queue.get()
    print('handler: got response')
    return web.Response(text=str(result))

app.router.add_post('/block', handle_block)


# Socket.IO connection
@sio.event
async def connect(sid, environ):
    print(f"[connect] {sid}")
    task = asyncio.create_task(client_wrapper(sid))
    clients[sid] = task


async def client_wrapper(sid: str):
    response_queue = Queue()
    msg_queue = Queue()
    request_queue = Queue()
    queues[sid] = response_queue, msg_queue, request_queue

    print("creating process")
    proc = Process(
        target=worker_process,
        args=(sid, response_queue, msg_queue, request_queue, 'mode'))
    proc.start()
    process_refs[sid] = proc
    print("Process created")

    loop = asyncio.get_event_loop()

    # def start_queue_watcher(loop, msg_queue, sid):
    #     def watcher():
    #         while True:
    #             try:
    #                 msg = msg_queue.get()  # блокирующий вызов
    #                 if msg is None:
    #                     print("Watcher received stop signal")
    #                     break
    #
    #                 print(f"[watcher] got msg from queue [{len(msg)}] {str(msg)[:80]}")
    #                 async def emit_message(msg):
    #                     print(f"EMIT START: {msg['msg'][:30]}")
    #                     await sio.emit("message", msg, to=sid)
    #                     print("EMIT DONE")
    #
    #                 asyncio.run_coroutine_threadsafe(emit_message(msg), loop)
    #                 # loop.call_soon_threadsafe(asyncio.create_task, emit_message(msg))
    #
    #             except Exception as e:
    #                 print("[watcher] exception:", e)
    #                 break
    #
    #     thread = Thread(target=watcher, daemon=True)
    #     thread.start()
    #     return thread
    #
    # start_queue_watcher(loop, msg_queue, sid)

    try:
        while True:

            if not proc.is_alive() and msg_queue.empty():
                print('proc is not alive anymore and queue is empty')
                break

            msg = await loop.run_in_executor(None, msg_queue.get)
            print("got msg from queue", str(msg)[:80])
            await sio.emit("message", msg, to=sid)

        print("end while")

    # except asyncio.CancelledError:
    #     print('asyncio.CancelledError')
    #     stop_event.set()
    #     proc.join(timeout=1)
    #     if proc.is_alive():
    #         proc.terminate()
    #         proc.join(timeout=1)
    except Exception as e:
        print('exception', e)


# Worker process logic
def worker_process(
        process_id: str,
        response_queue: Queue,
        msg_queue: Queue,
        request_queue: Queue,
        mode: str
) -> None:
    print(f"Process {process_id} started")

    sid = process_id
    client = FrontendClient(sid, msg_queue)
    print(f"Created FrontendClient with sid {sid}")
    ctr = 0

    while True:
        print('Worker is waiting for command...')
        command = request_queue.get()
        type = command.get('type')
        args = command.get('args')
        print(f"Worker: received command: {type} with args: {args}")

        if args.get('do', None) == 'train':
            epochs = int(3 + 100*random())
            proc = Process(
                target=client.start,
                args=(epochs,))
            proc.start()
            result = 'client process started'
        else:
            result = {'data': f"some http result #{ctr}"}
            ctr += 1

        response_queue.put(result)
        print('Worker: put response')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class FrontendClient:
    def __init__(self, sid, msg_queue):
        self.sid = sid
        self.mpqueue = msg_queue

    def start(self, epochs):
        print("Client starts training")
        self.ctr = 0

        # Параметры
        num_nodes = 1000
        num_node_features = 1600
        num_classes = 5
        num_edges = 50000
        # epochs = 500

        # Генерируем случайные признаки узлов
        x = torch.randn((num_nodes, num_node_features))

        # Генерируем случайные ребра (двунаправленные)
        edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

        # Генерируем случайные метки классов для узлов
        y = torch.randint(0, num_classes, (num_nodes,))

        # Создаем графовый объект
        graph_data = Data(x=x, edge_index=edge_index, y=y)

        # Простая GCN модель
        class SimpleGCN(torch.nn.Module):
            def __init__(self):
                super(SimpleGCN, self).__init__()
                self.conv1 = GCNConv(num_node_features, 64)
                self.conv2 = GCNConv(64, num_classes)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = self.conv2(x, edge_index)
                return x

        model = SimpleGCN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        epochs = 100
        for _ in range(epochs):
            # sleep(random())

            optimizer.zero_grad()
            out = model(graph_data)
            loss = criterion(out, graph_data.y)
            loss.backward()
            optimizer.step()

            data = {
                'msg': f"socket message from client {self.sid} #{self.ctr}",
                'data': 5 * 1024 * [1]
            }
            self.mpqueue.put(data)
            print(f'Client sid={self.sid} puts msg #{self.ctr} to queue')  # print выводится
            self.ctr += 1

    # def _send(
    #         self
    # ) -> None:
    #     """ Send leftmost actual data element from the queue. """
    #     print('_send()')
    #     data = None
    #     # Find actual data elem
    #     print('len(self.queue)', len(self.queue))
    #     while len(self.queue) > 0:
    #         data, tag, id = self.queue.popleft()
    #         # Check if actual
    #         if self.tag_queue[tag].get_first_id() <= id:
    #             break
    #
    #     if data is None:
    #         return
    #
    #     print('self.socket.send()')
    #     self.socket.send(data, to=self.sid)
    #     # await self.sio.emit(data, to=self.sid)
    #     print('self.socket.send() done')
    #
    # def _cycle(
    #         self
    # ) -> None:
    #     """ Send messages from the queue until it is empty. """
    #     self.active = True
    #     while True:
    #         print('len(self.queue) =', len(self.queue))
    #         if len(self.queue) == 0:
    #             self.active = False
    #             break
    #         self._send()
    #         print('self.sleep_time =', self.sleep_time)
    #         sleep(self.sleep_time)
    #         print('sleep done')


if __name__ == '__main__':
    web.run_app(app, port=5678)
