import asyncio
import contextlib
import json
import logging
import queue
import signal
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from multiprocessing import Process, Queue
from datetime import datetime
from typing import Dict, Any

from aiohttp import web
import socketio
import jinja2
import aiohttp_jinja2

from gnn_aid.aux.data_info import DataInfo
from web_interface.back_front import json_dumps
from web_interface.back_front.frontend_client import ClientMode, FrontendClient
from web_interface.back_front.utils import SocketConnect, STATIC_DIR, TEMPLATES_DIR, LOG_DIR

LOG_FILE = LOG_DIR / f"server_{datetime.now()}.log"
CLIENT_RESTART_DELAY_SEC = 30


def setup_logging(force: bool = False) -> logging.Logger:
    logger = logging.getLogger("gnn_aid.web")

    if logger.handlers and not force:
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [pid=%(process)d] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            with contextlib.suppress(Exception):
                handler.close()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()


sio = socketio.AsyncServer(
    async_mode="aiohttp",
    ping_timeout=600,
    ping_interval=25,
    cors_allowed_origins='*'
)
app = web.Application()
sio.attach(app)

aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(TEMPLATES_DIR))

clients: Dict[str, asyncio.Task] = {}
workers: Dict[str, Dict[str, Any]] = {}


def queue_get_with_timeout(q, timeout=0.5):
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


def make_worker_state(mode: ClientMode) -> Dict[str, Any]:
    return {
        "mode": mode,
        "response_queue": Queue(),
        "msg_queue": Queue(),
        "request_queue": Queue(),
        "proc": None,
        "restart_lock": asyncio.Lock(),
        "is_restarting": False,
    }


def close_worker_state(state: Dict[str, Any]) -> None:
    response_queue = state["response_queue"]
    msg_queue = state["msg_queue"]
    request_queue = state["request_queue"]
    proc = state["proc"]

    with contextlib.suppress(Exception):
        request_queue.put({"type": "STOP", "args": {}})

    if proc is not None:
        if proc.is_alive():
            proc.join(timeout=1)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)

    for q in (response_queue, msg_queue, request_queue):
        with contextlib.suppress(Exception):
            q.close()
        with contextlib.suppress(Exception):
            q.join_thread()


def start_worker_for_sid(sid: str) -> Dict[str, Any]:
    state = workers[sid]

    proc = Process(
        target=worker_process,
        args=(
            sid,
            state["response_queue"],
            state["msg_queue"],
            state["request_queue"],
            state["mode"],
        ),
        daemon=True,
    )
    proc.start()
    state["proc"] = proc

    logger.info(
        "Started worker for sid=%s pid=%s mode=%s",
        sid, proc.pid, state["mode"].value
    )
    return state


def replace_worker_for_sid(sid: str) -> Dict[str, Any]:
    old_state = workers[sid]
    mode = old_state["mode"]

    logger.info("Replacing worker state for sid=%s", sid)
    close_worker_state(old_state)

    new_state = make_worker_state(mode)
    workers[sid] = new_state
    start_worker_for_sid(sid)
    return new_state


async def restart_only_this_client(sid: str, error_text: str, tb: str = "") -> None:
    state = workers.get(sid)
    if state is None:
        return

    async with state["restart_lock"]:
        if state["is_restarting"]:
            return

        state["is_restarting"] = True

        logger.exception("Restarting only sid=%s because of backend error: %s", sid, error_text)

        with contextlib.suppress(Exception):
            await sio.emit(
                "message",
                {
                    "type": "server_error",
                    "title": "Backend error",
                    "text": f'{error_text}\n\nSee the logs in {LOG_FILE.resolve()}',
                    "traceback": tb,
                    "restart_in_sec": CLIENT_RESTART_DELAY_SEC,
                },
                to=sid,
            )

        await asyncio.sleep(CLIENT_RESTART_DELAY_SEC)

        if sid not in clients or sid not in workers:
            return

        replace_worker_for_sid(sid)

        workers[sid]["is_restarting"] = False

        with contextlib.suppress(Exception):
            await sio.emit(
                "message",
                {
                    "type": "server_info",
                    "text": "Backend restarted successfully",
                },
                to=sid,
            )


@aiohttp_jinja2.template("interpretation.html")
async def handle_interpretation(request):
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

    logger.info("ask request from sid=%s", sid)
    ask_cmd = data.get('ask')

    if ask_cmd == "parameters":
        type_ = data.get('type')
        params = FrontendClient.get_parameters(type_)
        params = json_dumps(params)
        return web.Response(text=params)

    return web.Response(status=400, text=f"Unknown 'ask' command {ask_cmd}")


async def handle_url(request):
    url = request.match_info.get("url")
    logger.info("url=%s", url)

    if url not in ['dataset', 'model', 'explainer', 'block']:
        return web.Response(status=404, text="Invalid URL")

    if request.method != "POST":
        return web.Response(status=405, text="Method Not Allowed")

    data = await request.post()
    sid = data.get("sid")

    if sid not in clients:
        return web.Response(status=404, text="Unknown SID")

    state = workers.get(sid)
    if state is None:
        return web.Response(status=503, text="Worker state not found")

    if state["is_restarting"]:
        return web.Response(status=503, text="Backend is restarting for this client")

    response_queue = state["response_queue"]
    request_queue = state["request_queue"]
    proc = state["proc"]

    if proc is None or not proc.is_alive():
        return web.Response(status=503, text="Worker is not alive")

    logger.info("%s http request from sid=%s args=%s", url, sid, dict(data))
    request_queue.put({"type": url, "args": dict(data)})

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, queue_get_with_timeout, response_queue, 10.0)

    if result is None:
        return web.Response(status=504, text="Timeout waiting for worker response")

    if isinstance(result, dict) and result.get("__meta__") == "worker_error_response":
        return web.Response(
            status=500,
            text=json.dumps(result),
            content_type='application/json'
        )

    return web.Response(text=json.dumps(result), content_type='application/json')



# Register routes
app.router.add_get("/", handle_analysis)
app.router.add_get("/analysis", handle_analysis)
app.router.add_get("/defense", handle_defense)
app.router.add_get("/interpretation", handle_interpretation)
app.router.add_post("/ask", handle_ask)
app.router.add_post("/{url}", handle_url)

app.router.add_static('/static/', path=str(STATIC_DIR), name='static')


@sio.event
async def connect(sid, environ):
    query_string = environ.get('QUERY_STRING', '')
    query = dict(qc.split('=') for qc in query_string.split('&') if '=' in qc)
    mode = ClientMode(query.get('mode', None))

    logger.info("[connect] sid=%s mode=%s", sid, mode.value)

    workers[sid] = make_worker_state(mode)
    task = asyncio.create_task(client_wrapper(sid))
    clients[sid] = task


@sio.event
async def disconnect(sid):
    logger.info("[disconnect] sid=%s", sid)

    task = clients.get(sid)
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def client_wrapper(sid: str):
    state = workers.get(sid)
    if state is None:
        logger.warning("client_wrapper started without worker state sid=%s", sid)
        return

    start_worker_for_sid(sid)
    loop = asyncio.get_running_loop()

    try:
        while True:
            state = workers.get(sid)
            if state is None:
                break

            if state["is_restarting"]:
                await asyncio.sleep(0.5)
                continue

            proc = state["proc"]
            msg_queue = state["msg_queue"]

            if proc is None:
                await asyncio.sleep(0.2)
                continue

            if not proc.is_alive() and msg_queue.empty():
                exitcode = proc.exitcode
                await restart_only_this_client(
                    sid, f"Worker process crashed unexpectedly. exitcode={exitcode}")
                continue

            msg = await loop.run_in_executor(None, queue_get_with_timeout, msg_queue, 0.5)
            if msg is None:
                continue

            if isinstance(msg, dict) and msg.get("__meta__") == "worker_crash":
                tb = msg.get("traceback", "")
                err = msg.get("error_text", "Unknown backend error")
                logger.error("Worker crash for sid=%s: %s\n%s", sid, err, tb)
                await restart_only_this_client(sid, err, tb)
                continue

            logger.info("got msg from queue sid=%s [%s] %s", sid, len(str(msg)), str(msg)[:120])
            await sio.emit("message", msg, to=sid)

    except asyncio.CancelledError:
        logger.info("client_wrapper cancelled sid=%s", sid)
        raise

    except Exception:
        logger.exception("Unhandled exception in client_wrapper sid=%s", sid)
        raise

    finally:
        logger.info("cleanup for sid=%s", sid)

        state = workers.pop(sid, None)
        if state is not None:
            await async_close_worker_state(state)

        clients.pop(sid, None)


def worker_process(
    sid: str,
    response_queue: Queue,
    msg_queue: Queue,
    request_queue: Queue,
    mode: ClientMode
) -> None:
    # We ignore SIGINT in worker - it is for server
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    setup_logging(force=True)
    logger = logging.getLogger("gnn_aid.web")

    logger.info("Process started sid=%s", sid)

    try:
        socket_connect = AiohttpSocketConnect(msg_queue)
        client = FrontendClient(socket_connect, mode)
        logger.info("Created FrontendClient sid=%s", sid)

        client.run_loop(response_queue, msg_queue, request_queue)

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Unhandled exception in worker_process sid=%s", sid)

        with contextlib.suppress(Exception):
            msg_queue.put_nowait({
                "__meta__": "worker_crash",
                "sid": sid,
                "error_text": f"{type(e).__name__}: {e}",
                "traceback": tb,
            })

        with contextlib.suppress(Exception):
            response_queue.put_nowait({
                "__meta__": "worker_error_response",
                "error_text": f"{type(e).__name__}: {e}",
            })

        raise


class AiohttpSocketConnect(SocketConnect):
    def __init__(self, queue: Queue):
        super().__init__()
        self.mp_queue = queue

    def _send_data(self, data):
        self.mp_queue.put_nowait(data)
        logger.info("put msg to mpqueue [%s] %s", len(str(data)), str(data)[:100])


async def async_close_worker_state(state):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, close_worker_state, state)


async def emit_fatal_stop_to_all(reason: str, traceback_text: str = ""):
    payload = {
        "type": "fatal_stop",
        "title": "Server shutdown",
        "text": reason,
        "traceback": traceback_text,
    }

    for sid in list(clients.keys()):
        with contextlib.suppress(Exception):
            await sio.emit("message", payload, to=sid)


async def on_shutdown(app):
    logger.info("Server shutdown started")

    reason = "Server is shutting down (Ctrl+C or server stop)"
    await emit_fatal_stop_to_all(reason)

    # Даем фронту шанс получить сообщение
    await asyncio.sleep(0.5)

    # Закрываем socket-соединения
    for sid in list(clients.keys()):
        with contextlib.suppress(Exception):
            await sio.disconnect(sid)

    tasks = []

    for sid, task in list(clients.items()):
        if task is not None and not task.done():
            task.cancel()
            tasks.append(task)

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("Server shutdown finished")


app.on_shutdown.append(on_shutdown)


def run_aiohttp_server(port=5000):
    logger.info("Starting aiohttp server on port %s", port)
    web.run_app(app, port=port)


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    run_aiohttp_server()