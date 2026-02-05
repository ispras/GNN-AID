import asyncio
import json
import logging
import traceback
from multiprocessing import Queue, SimpleQueue, Event, Process, Manager
from queue import Empty
from random import random
from time import sleep
from typing import Dict

import socketio
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from aux.data_info import DataInfo
from web_interface.back_front.frontend_client import ClientMode, FrontendClient
from web_interface.back_front.utils import WebInterfaceError, json_dumps, json_loads

sio = socketio.AsyncServer(async_mode="asgi")
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
sio_app = socketio.ASGIApp(sio, app)

clients = {}  # client_id (sid) -> asyncio.Task
process_refs: Dict[str, Process] = {}
stop_flags: Dict[str, Event] = {}
queues: Dict[str, Queue] = {}


# Папка с шаблонами (например, "templates")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # DataInfo.refresh_all_data_info()

    return templates.TemplateResponse("analysis.html", {
        "request": request, "mode": ClientMode.analysis.value})


@app.api_route("/{url}", methods=["GET", "POST"])
async def handle_url(url: str, request: Request):
    print('url', url)
    if url not in ['dataset', 'model', 'explainer', 'block']:
        raise HTTPException(status_code=404, detail=f"Invalid URL, {url}")

    if request.method == "POST":
        form = await request.form()
        sid = form.get("sid")

        if sid not in clients:
            print(f"Error: sid {sid} not in clients: {list(clients.keys())}")
            raise HTTPException(status_code=404, detail=f"Unknown SID {sid}")

        response_queue, _, request_queue = queues[sid]

        print('http request from', sid, url, 'args', dict(form))
        request_queue.put({"type": url, "args": dict(form)})

        # Ждём ответ из очереди
        result = response_queue.get()

        return PlainTextResponse(str(result))

    # # Обработка GET если нужно
    # return PlainTextResponse(f"GET request to /{url}")


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
    # stop_flags[sid] = stop_event

    print("creating process")
    proc = Process(
        target=worker_process,
        args=(sid, response_queue, msg_queue, request_queue, 'mode'))
    proc.start()
    process_refs[sid] = proc
    print("Process created")

    loop = asyncio.get_event_loop()

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


def worker_process(
        process_id: str,
        response_queue: Queue,
        msg_queue: Queue,
        request_queue: Queue,
        mode: str
) -> None:
    print(f"Process {process_id} started")

    sid = process_id
    client = FrontendClient(sid, mode, msg_queue)
    print(f"Created FrontendClient with sid {sid}")

    try:
        while True:
            print('Worker is waiting for command...')
            # command = await async_queue_get(request_queue)
            command = request_queue.get()  # This blocks until a command is received
            type = command.get('type')
            args = command.get('args')
            print(f"Worker: received command: {type} with args: {args}")

            if type == "dataset":
                get = args.get('get')
                set = args.get('set')
                part = args.get('part')
                if part:
                    part = json_loads(part)

                if set == "visible_part":
                    result = client.dcBlock.set_visible_part(part=part)

                elif get == "data":
                    dataset_data = client.dcBlock.get_dataset_data(part=part)
                    data = dataset_data.to_json()
                    logging.info(f"Length of dataset_data: {len(data)}")
                    result = data

                elif get == "var_data":
                    if not client.dvcBlock.is_set():
                        result = ''
                    else:
                        dataset_var_data = client.dvcBlock.get_dataset_var_data(part=part)
                        data = dataset_var_data.to_json()
                        logging.info(f"Length of dataset_var_data: {len(data)}")
                        result = data

                elif get == "stat":
                    stat = args.get('stat')
                    result = json_dumps(client.dcBlock.get_stat(stat))

                elif get == "index":
                    result = client.dcBlock.get_index()

                else:
                    raise WebInterfaceError(f"Unknown 'part' command {get} for dataset")

                response_queue.put(result)
                # return

            elif type == "block":
                block = args.get('block')
                func = args.get('func')
                params = args.get('params')
                if params:
                    params = json_loads(params)
                print(f"request_block: block={block}, func={func}, params={params}")
                # TODO what if raise exception? process will stop
                client.request_block(block, func, params)
                response_queue.put('')

            else:
                raise WebInterfaceError()

            # response_queue.put(result)
            print('Worker: put response')
    except Exception as e:
        print("Worker crashed:", e)
        traceback.print_exc()

# class FrontendClient:
#     def __init__(self, sid, msg_queue):
#         self.sid = sid
#         self.mpqueue = msg_queue
#
#     def start(self, epochs):
#         print("Client starts training")
#         self.ctr = 0
#
#         for _ in range(epochs):
#             sleep(random())
#
#             data = {
#                 'msg': f"socket message from client {self.sid} #{self.ctr}",
#                 'data': 5000 * 1024 * [1]
#             }
#             self.mpqueue.put(data)
#             print(f'Client sid={self.sid} puts msg #{self.ctr} to queue')  # print выводится
#             self.ctr += 1

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
    port = 5678
    uvicorn.run("test_fastapi:sio_app", host="0.0.0.0", port=port, reload=False, workers=1)

