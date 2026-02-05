import asyncio
import json
import logging
import multiprocessing
from multiprocessing import Queue, SimpleQueue, Event, Process
from queue import Empty
from typing import Dict

import socketio
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import PlainTextResponse
from starlette.responses import HTMLResponse, JSONResponse
from starlette.templating import Jinja2Templates

from gnn_aid.aux.data_info import DataInfo
from web_interface.back_front.frontend_client import FrontendClient, ClientMode
from web_interface.back_front.utils import WebInterfaceError, json_loads, json_dumps

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
    DataInfo.refresh_all_data_info()

    return templates.TemplateResponse("analysis.html", {
        "request": request, "mode": ClientMode.analysis.value})


@app.get("/analysis", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("analysis.html", {
        "request": request, "mode": ClientMode.analysis.value})


@app.get("/defense", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("defense.html", {
        "request": request, "mode": ClientMode.defense.value})


@app.get("/interpretation", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("interpretation.html", {
        "request": request, "mode": ClientMode.interpretation.value})


# @app.get("/datasets")
# async def list_datasets():
#     return {"datasets": ["mnist", "cifar10", "custom.csv"]}

@app.api_route("/ask", methods=["GET", "POST"])
async def ask(request: Request):
    form = await request.form()
    sid = form.get('sid')
    if sid not in clients:
        raise WebInterfaceError("Unknown SID")

    print('ask request from', sid)
    ask_cmd = form.get('ask')

    if ask_cmd == "parameters":
        type_ = form.get('type')
        return JSONResponse(content=FrontendClient.get_parameters(type_))
    else:
        raise WebInterfaceError(f"Unknown 'ask' command {ask_cmd}")


@app.api_route("/block", methods=["GET", "POST"])
async def block(
        request: Request
) -> str:
    form = await request.form()
    sid = form.get('sid')
    if sid not in clients:
        raise WebInterfaceError("Unknown SID")

    print('block request from', sid)
    request_queue, _, response_queue = queues[sid]
    request_queue.put({'type': 'block', 'args': form})

    result = response_queue.get()
    return result


@app.api_route("/{url}", methods=["GET", "POST"])
async def handle_url(url: str, request: Request):
    print('url', url)
    if url not in ['dataset', 'model', 'explainer']:
        raise HTTPException(status_code=404, detail="Invalid URL")

    if request.method == "POST":
        form = await request.form()
        sid = form.get("sid")

        if sid not in clients:
            raise HTTPException(status_code=404, detail="Unknown SID")

        request_queue, _, response_queue = queues[sid]

        print(url, 'request from', sid)
        request_queue.put({"type": url, "args": dict(form)})

        # Ждём ответ из очереди
        result = response_queue.get()
        return PlainTextResponse(str(result))

    # Обработка GET если нужно
    return PlainTextResponse(f"GET request to /{url}")


@sio.event
async def connect(sid, environ):
    print(f"[connect] {sid}")

    config = {}
    task = asyncio.create_task(client_wrapper(sid, sio, config))
    clients[sid] = task


@sio.event
async def disconnect(sid):
    print(f"[disconnect] {sid}")
    # await stop_training_for_client(sid)
    clients.pop(sid, None)


async def client_wrapper(sid: str, sio, config: dict):
    request_queue = Queue()
    msg_queue = SimpleQueue()
    response_queue = Queue()
    stop_event = Event()
    queues[sid] = request_queue, msg_queue, response_queue
    stop_flags[sid] = stop_event

    print("creating process")
    proc = Process(
        target=worker_process,
        args=(sid, request_queue, msg_queue, response_queue, ClientMode.analysis))
    proc.start()
    process_refs[sid] = proc
    print("Process created")

    loop = asyncio.get_event_loop()

    try:
        while True:
            if not proc.is_alive() and msg_queue.empty():
                print('proc is not alive anymore and queue is empty')
                break

            print('waiting msg from queue')
            msg = await loop.run_in_executor(None, msg_queue.get)
            print("got msg from queue", str(msg)[:40])
            await sio.emit("message", msg, to=sid)

    except asyncio.CancelledError:
        print('asyncio.CancelledError')
        stop_event.set()
        proc.join(timeout=1)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)


def worker_process(
        process_id: str,
        request_queue: Queue,
        msg_queue: Queue,
        response_queue: Queue,
        mode: ClientMode
) -> None:
    print(f"Process {process_id} started")
    # TODO problem is each process sends data to main process then to frontend.
    #  Easier to send it directly to url

    sid = process_id
    client = FrontendClient(socketio, mode, msg_queue)
    print(f"Created FrontendClient with sid {sid}")

    client.run_loop(request_queue, msg_queue, response_queue)


def run_fastapi_server(port=5000):
    uvicorn.run("main_fastapi:sio_app", host="0.0.0.0", port=port, reload=False)


if __name__ == '__main__':
    run_fastapi_server()
