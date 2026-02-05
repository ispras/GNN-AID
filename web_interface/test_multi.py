import asyncio
from multiprocessing import Queue, SimpleQueue, Event, Process, Manager
from queue import Empty
from random import random
from time import sleep
from typing import Dict

import socketio
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates

sio = socketio.AsyncServer(async_mode="asgi")
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
sio_app = socketio.ASGIApp(sio, app)

clients = {}  # client_id (sid) -> asyncio.Task
process_refs: Dict[str, Process] = {}
stop_flags: Dict[str, Event] = {}
queues: Dict[str, Queue] = {}

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("test.html", {"request": request})


@app.api_route("/block", methods=["GET", "POST"])
async def handle_url(request: Request):
    if request.method == "POST":
        form = await request.form()
        sid = form.get("sid")

        if sid not in clients:
            raise HTTPException(status_code=404, detail="Unknown SID")

        response_queue, _, request_queue = queues[sid]

        print('handler: http request from', sid, 'args', dict(form))
        request_queue.put({"type": 'url', "args": dict(form)})

        # Ждём ответ из очереди
        # result = response_queue.get()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, response_queue.get)

        print('handler: got response')
        return PlainTextResponse(str(result))


async def handle_url_(sid, form):
        if sid not in clients:
            raise RuntimeError(f"Unknown SID {sid}")

        response_queue, _, request_queue = queues[sid]

        print('handler: http request from', sid, 'args', dict(form))
        request_queue.put({"type": 'url', "args": dict(form)})

        # Ждём ответ из очереди
        result = response_queue.get()
        # loop = asyncio.get_running_loop()
        # result = await loop.run_in_executor(None, response_queue.get)

        print('handler: got response')
        return PlainTextResponse(str(result))


@sio.event
async def connect(sid, environ):
    print(f"[connect] {sid}")
    task = asyncio.create_task(client_wrapper(sid, sio))
    clients[sid] = task


@sio.event
async def disconnect(sid):
    print(f"[disconnect] {sid}")
    clients.pop(sid, None)


async def client_wrapper(sid: str, sio):
    # stop_event = Event()
    response_queue = Queue()
    msg_queue = Queue()
    request_queue = Queue()
    queues[sid] = response_queue, msg_queue, request_queue
    print("queues at wrapper", queues)
    # stop_flags[sid] = stop_event

    print("creating process")
    proc = Process(
        target=worker_process,
        args=(sid, response_queue, msg_queue, request_queue, 'mode'))
    proc.start()
    process_refs[sid] = proc
    print("Process created")


    # loop = asyncio.get_event_loop()
    #
    # try:
    #     while True:
    #
    #         if not proc.is_alive() and msg_queue.empty():
    #             print('proc is not alive anymore and queue is empty')
    #             break
    #
    #         msg = await loop.run_in_executor(None, msg_queue.get)
    #         print("got msg from queue", str(msg)[:80])
    #         await sio.emit("message", msg, to=sid)
    #
    #     print("end while")
    #
    # # except asyncio.CancelledError:
    # #     print('asyncio.CancelledError')
    # #     stop_event.set()
    # #     proc.join(timeout=1)
    # #     if proc.is_alive():
    # #         proc.terminate()
    # #         proc.join(timeout=1)
    # except Exception as e:
    #     print('exception', e)


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
        command = request_queue.get()  # This blocks until a command is received
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


class FrontendClient:
    def __init__(self, sid, msg_queue):
        self.sid = sid
        self.mpqueue = msg_queue

    def start(self, epochs):
        print("Client starts training")
        self.ctr = 0

        for _ in range(epochs):
            sleep(random())

            data = {
                'msg': f"socket message from client {self.sid} #{self.ctr}",
                'data': 5000 * 1024 * [1]
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


def run_async_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(my_async_server())
    finally:
        loop.close()


async def my_async_server():
    sid = "123"
    await connect(sid, 0)
    # await connect("234", 0)

    # await asyncio.sleep(1)
    # print("queues at server", queues)

    for i in range(10):
        await asyncio.sleep(10*random())
        form = {}
        print(f"Server: Request {i}")
        res = await handle_url_(sid, form)
        print(f"Server: Got result {i}")


if __name__ == '__main__':
    port = 5678
    uvicorn.run("test_multi:sio_app", host="0.0.0.0", port=port, reload=False)

    # run_async_server()


    # from multiprocessing import Process, Queue
    # import time
    #
    #
    # def worker(msg_queue, request_queue):
    #     while True:
    #         cmd = request_queue.get()
    #         print("Worker got:", cmd)
    #         msg_queue.put(f"Processed {cmd}")
    #
    #
    # msg_queue = Queue()
    # request_queue = Queue()
    #
    # p = Process(target=worker, args=(msg_queue, request_queue))
    # p.start()
    #
    # request_queue.put("train")
    # request_queue.put("train")
    # request_queue.put("train")
    # request_queue.put("train")
    # request_queue.put("train")
    #
    # # main читает независимо
    # while True:
    #     if not msg_queue.empty():
    #         print("Main received:", msg_queue.get())
    #     time.sleep(0.5)
    #
