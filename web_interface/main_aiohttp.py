import asyncio
import json
import logging
# import multiprocessing as mp
# mp.set_start_method("spawn", force=True)

from multiprocessing import Process, Queue
from typing import Dict
from aiohttp import web
import socketio
import jinja2
import aiohttp_jinja2

from gnn_aid.aux.data_info import DataInfo
from web_interface.back_front.frontend_client import ClientMode, FrontendClient
from web_interface.back_front.utils import WebInterfaceError, json_loads, json_dumps, SocketConnect

# Socket.IO server (ASGI not used here)
sio = socketio.AsyncServer(
    async_mode="aiohttp",
    ping_timeout=600,  # wait pong from client
    ping_interval=25,
    cors_allowed_origins='*'
)
app = web.Application()
sio.attach(app)

aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader("templates"))

clients = {}  # client_id (sid) -> asyncio.Task
queues: Dict[str, Queue] = {}


# Route for interpretation
@aiohttp_jinja2.template("interpretation.html")
async def handle_interpretation(request):
    # print("[http] /interpretation")
    DataInfo.refresh_all_data_info()
    return {
        "request": request,
        "mode": ClientMode.interpretation.value
    }


# Route for home and analysis
@aiohttp_jinja2.template("analysis.html")
async def handle_analysis(request):
    # print("[http] /analysis")
    DataInfo.refresh_all_data_info()
    return {
        "request": request,
        "mode": ClientMode.analysis.value
    }


# Route for defense
@aiohttp_jinja2.template("defense.html")
async def handle_defense(request):
    # print("[http] /defense")
    DataInfo.refresh_all_data_info()
    return {
        "request": request,
        "mode": ClientMode.defense.value
    }


# Route: POST /block
async def handle_ask(request):
    data = await request.post()
    sid = data.get('sid')
    if sid not in clients:
        return web.Response(status=400, text="Unknown SID")

    print('ask request from', sid)
    ask_cmd = data.get('ask')

    if ask_cmd == "parameters":
        type_ = data.get('type')
        params = FrontendClient.get_parameters(type_)
        params = json_dumps(params)
        return web.Response(text=params)
    else:
        return web.Response(status=400, text=f"Unknown 'ask' command {ask_cmd}")


# dynamic /{url} endpoint
async def handle_url(request):
    url = request.match_info.get("url")
    print('url', url)
    if url not in ['dataset', 'model', 'explainer', 'block']:
        return web.Response(status=404, text="Invalid URL")

    if request.method == "POST":
        data = await request.post()
        sid = data.get("sid")

        if sid not in clients:
            return web.Response(status=404, text="Unknown SID")

        response_queue, _, request_queue = queues[sid]

        print(url, 'http request from', sid, 'args', dict(data))
        request_queue.put({"type": url, "args": dict(data)})

        # Wait for response from worker
        result = response_queue.get()
        return web.Response(text=json.dumps(result), content_type='application/json')

    return web.Response(status=405, text="Method Not Allowed")


# Register routes
app.router.add_get("/", handle_analysis)
app.router.add_get("/analysis", handle_analysis)
app.router.add_get("/defense", handle_defense)
app.router.add_get("/interpretation", handle_interpretation)
app.router.add_post("/ask", handle_ask)
app.router.add_post("/{url}", handle_url)

app.router.add_static('/static/', path='static', name='static')


# Socket.IO connection
@sio.event
async def connect(sid, environ):
    # Parse mode from query
    query_string = environ.get('QUERY_STRING', '')
    query = dict(qc.split('=') for qc in query_string.split('&') if '=' in qc)
    mode = query.get('mode', None)
    mode = ClientMode(mode)

    print(f"[connect] {sid}, mode={mode}")
    task = asyncio.create_task(client_wrapper(sid, mode))
    clients[sid] = task


@sio.event
async def disconnect(sid):
    print(f"[disconnect] {sid}")
    clients.pop(sid, None)


async def client_wrapper(sid: str, mode: ClientMode):
    response_queue = Queue()
    msg_queue = Queue()
    request_queue = Queue()
    queues[sid] = response_queue, msg_queue, request_queue

    print("creating process")
    proc = Process(
        target=worker_process,
        args=(sid, response_queue, msg_queue, request_queue, mode))
    proc.start()
    print("Process created")

    loop = asyncio.get_event_loop()
    try:
        while True:
            if not proc.is_alive() and msg_queue.empty():
                print('proc is not alive anymore and queue is empty')
                break

            msg = await loop.run_in_executor(None, msg_queue.get)
            print(f"got msg from queue [{len(msg)}] {str(msg)[:80]}")
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
        sid: str,
        response_queue: Queue,
        msg_queue: Queue,
        request_queue: Queue,
        mode: ClientMode
) -> None:
    print(f"Process {sid} started")

    sid = sid
    socket_connect = AiohttpSocketConnect(msg_queue)
    client = FrontendClient(socket_connect, mode)
    print(f"Created FrontendClient with sid {sid}")

    client.run_loop(response_queue, msg_queue, request_queue)


class AiohttpSocketConnect(SocketConnect):
    """
    Implementation for aiohttp - puts messages to multiprocessing.Queue
    """
    def __init__(self, queue: Queue):
        super().__init__()
        self.mp_queue = queue

    def _send_data(self, data):
        self.mp_queue.put_nowait(data)
        print(f"put msg to mpqueue [{len(str(data))}] {str(data)[:40]}")


def run_aiohttp_server(port=5000):
    web.run_app(app, port=port)


if __name__ == '__main__':
    run_aiohttp_server()
