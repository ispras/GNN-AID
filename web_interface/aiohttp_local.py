import asyncio
import contextlib
import ctypes
import json
import logging
import os
import queue
import signal
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from multiprocessing import Process, Queue
from typing import Any, Dict

import aiohttp_jinja2
import jinja2
import socketio
from aiohttp import web

from gnn_aid.auxil.data_info import DataInfo
from web_interface.back_front import json_dumps
from web_interface.back_front.frontend_client import ClientMode, FrontendClient
from web_interface.back_front.utils import (
    LOG_DIR,
    STATIC_DIR,
    TEMPLATES_DIR,
    SocketConnect,
    get_sid_logger,
)

LOG_FILE = LOG_DIR / datetime.now().strftime("server_local_%Y-%m-%d_%H-%M-%S.log")
CLIENT_RESTART_DELAY_SEC = 5
RESPONSE_TIMEOUT_SEC = 30.0
GRACEFUL_WORKER_STOP_SEC = 2.0
FORCEFUL_WORKER_KILL_SEC = 1.0

shutdown_lock = asyncio.Lock()
shutdown_started = False


class SidFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "sid"):
            record.sid = "-"
        return True


def setup_logging(force: bool = False) -> logging.Logger:
    logger = logging.getLogger("gnn_aid.web.local")

    if logger.handlers and not force:
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [pid=%(process)d] [sid=%(sid)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.setLevel(logging.INFO)

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            with contextlib.suppress(Exception):
                handler.close()

    sid_filter = SidFilter()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(sid_filter)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(sid_filter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


setup_logging()
server_logger = get_sid_logger()


sio = socketio.AsyncServer(
    async_mode="aiohttp",
    ping_timeout=600,
    ping_interval=25,
    cors_allowed_origins="*",
    serializer="msgpack",
)
app = web.Application()
sio.attach(app)

aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(TEMPLATES_DIR))

clients: Dict[str, asyncio.Task] = {}
workers: Dict[str, Dict[str, Any]] = {}


def queue_get_with_timeout(q: Queue, timeout: float = 0.5):
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
            proc.join(timeout=GRACEFUL_WORKER_STOP_SEC)

        if proc.is_alive():
            server_logger.warning("Worker pid=%s did not stop gracefully, terminating", proc.pid)
            with contextlib.suppress(Exception):
                proc.terminate()
            proc.join(timeout=FORCEFUL_WORKER_KILL_SEC)

        if proc.is_alive():
            server_logger.error("Worker pid=%s survived terminate(), killing", proc.pid)
            with contextlib.suppress(Exception):
                os.kill(proc.pid, signal.SIGKILL)
            proc.join(timeout=FORCEFUL_WORKER_KILL_SEC)

        if proc.is_alive():
            server_logger.critical("Worker pid=%s is still alive after SIGKILL", proc.pid)

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

    server_logger.info(
        "Started worker for sid=%s pid=%s mode=%s",
        sid,
        proc.pid,
        state["mode"].value,
    )
    return state



def replace_worker_for_sid(sid: str) -> Dict[str, Any]:
    old_state = workers[sid]
    mode = old_state["mode"]

    server_logger.info("Replacing worker state for sid=%s", sid)
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
        server_logger.exception("Restarting sid=%s because of backend error: %s", sid, error_text)

        with contextlib.suppress(Exception):
            await sio.emit(
                "message",
                {
                    "type": "server_error",
                    "title": "Backend error",
                    "text": f"{error_text}\n\nSee logs in {LOG_FILE.resolve()}",
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
                {"type": "server_info", "text": "Backend restarted successfully"},
                to=sid,
            )


@aiohttp_jinja2.template("interpretation.html")
async def handle_interpretation(request: web.Request):
    DataInfo.refresh_all_data_info()
    return {"request": request, "mode": ClientMode.interpretation.value}


@aiohttp_jinja2.template("analysis.html")
async def handle_analysis(request: web.Request):
    DataInfo.refresh_all_data_info()
    return {"request": request, "mode": ClientMode.analysis.value}


@aiohttp_jinja2.template("defense.html")
async def handle_defense(request: web.Request):
    DataInfo.refresh_all_data_info()
    return {"request": request, "mode": ClientMode.defense.value}


async def handle_ask(request: web.Request):
    data = await request.post()
    sid = data.get("sid")
    if sid not in clients:
        return web.Response(status=400, text="Unknown SID")

    server_logger.info("ask request from sid=%s", sid)
    ask_cmd = data.get("ask")

    if ask_cmd == "parameters":
        type_ = data.get("type")
        params = FrontendClient.get_parameters(type_)
        return web.Response(text=json_dumps(params))

    return web.Response(status=400, text=f"Unknown 'ask' command {ask_cmd}")


async def handle_url(request: web.Request):
    url = request.match_info.get("url")
    server_logger.info("url=%s", url)

    if url not in ["dataset", "model", "explainer", "block"]:
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
        return web.Response(status=503, text="Backend is restarting for this tab")

    response_queue = state["response_queue"]
    request_queue = state["request_queue"]
    proc = state["proc"]

    if proc is None or not proc.is_alive():
        return web.Response(status=503, text="Worker is not alive")

    payload = dict(data)
    server_logger.info("%s http request from sid=%s args=%s", url, sid, payload)
    request_queue.put({"type": url, "args": payload})

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, queue_get_with_timeout, response_queue, RESPONSE_TIMEOUT_SEC)

    if result is None:
        return web.Response(status=504, text="Timeout waiting for worker response")

    if isinstance(result, dict) and result.get("__meta__") == "worker_error_response":
        return web.Response(
            status=500,
            text=json.dumps(result),
            content_type="application/json",
        )

    return web.Response(text=json.dumps(result), content_type="application/json")


app.router.add_get("/", handle_analysis)
app.router.add_get("/analysis", handle_analysis)
app.router.add_get("/defense", handle_defense)
app.router.add_get("/interpretation", handle_interpretation)
app.router.add_post("/ask", handle_ask)
app.router.add_post("/{url}", handle_url)
app.router.add_static("/static/", path=str(STATIC_DIR), name="static")


@sio.event
async def connect(sid, environ, auth=None):
    del auth

    query_string = environ.get("QUERY_STRING", "")
    query = dict(qc.split("=", 1) for qc in query_string.split("&") if "=" in qc)
    mode = ClientMode(query.get("mode", None))

    sid_logger = get_sid_logger(sid)
    sid_logger.info("Client connected, mode=%s", mode.value)

    workers[sid] = make_worker_state(mode)
    task = asyncio.create_task(client_wrapper(sid))
    clients[sid] = task


@sio.event
async def disconnect(sid):
    get_sid_logger(sid).info("Client disconnected")

    task = clients.get(sid)
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def client_wrapper(sid: str):
    sid_logger = get_sid_logger(sid)

    state = workers.get(sid)
    if state is None:
        sid_logger.info("client_wrapper started without state")
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
                    sid,
                    f"Worker process crashed unexpectedly. exitcode={exitcode}",
                )
                continue

            msg = await loop.run_in_executor(None, queue_get_with_timeout, msg_queue, 0.5)
            if msg is None:
                continue

            if isinstance(msg, dict) and msg.get("__meta__") == "worker_crash":
                tb = msg.get("traceback", "")
                err = msg.get("error_text", "Unknown backend error")
                sid_logger.error("Worker crash %s\n%s", err, tb)
                await restart_only_this_client(sid, err, tb)
                continue

            sid_logger.info("got msg from queue [%s] %s", len(str(msg)), str(msg)[:120])
            await sio.emit("message", msg, to=sid)

    except asyncio.CancelledError:
        sid_logger.info("client_wrapper cancelled")
        raise

    except Exception:
        sid_logger.exception("Unhandled exception in client_wrapper")
        raise

    finally:
        sid_logger.info("cleanup")

        state = workers.pop(sid, None)
        if state is not None:
            await async_close_worker_state(state)

        clients.pop(sid, None)



def set_pdeathsig(sig=signal.SIGKILL):
    libc = ctypes.CDLL("libc.so.6")
    PR_SET_PDEATHSIG = 1
    return libc.prctl(PR_SET_PDEATHSIG, sig)



def worker_process(
    sid: str,
    response_queue: Queue,
    msg_queue: Queue,
    request_queue: Queue,
    mode: ClientMode,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    with contextlib.suppress(Exception):
        set_pdeathsig(signal.SIGKILL)

    setup_logging(force=True)
    logger = get_sid_logger(sid)
    logger.info("Process started")

    try:
        socket_connect = AiohttpSocketConnect(msg_queue, sid)
        client = FrontendClient(socket_connect, mode, sid)
        logger.info("Created FrontendClient")
        client.run_loop(response_queue, msg_queue, request_queue)

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Unhandled exception in worker_process")

        with contextlib.suppress(Exception):
            msg_queue.put_nowait(
                {
                    "__meta__": "worker_crash",
                    "sid": sid,
                    "error_text": f"{type(e).__name__}: {e}",
                    "traceback": tb,
                }
            )

        with contextlib.suppress(Exception):
            response_queue.put_nowait(
                {
                    "__meta__": "worker_error_response",
                    "error_text": f"{type(e).__name__}: {e}",
                }
            )

        raise


class AiohttpSocketConnect(SocketConnect):
    def __init__(self, queue_obj: Queue, sid: str):
        super().__init__()
        self.mp_queue = queue_obj
        self.logger = get_sid_logger(sid)

    def _send_data(self, data):
        self.mp_queue.put_nowait(data)
        self.logger.debug("put msg to mpqueue [len=%s] '%s'", len(str(data)), str(data)[:100])


async def async_close_worker_state(state: Dict[str, Any]):
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


async def graceful_shutdown(reason: str = "Server is shutting down"):
    global shutdown_started

    async with shutdown_lock:
        if shutdown_started:
            return
        shutdown_started = True

        server_logger.info("Graceful shutdown started: %s", reason)

        await emit_fatal_stop_to_all(reason)
        await asyncio.sleep(0.5)

        for sid in list(clients.keys()):
            with contextlib.suppress(Exception):
                await sio.disconnect(sid)

        loop = asyncio.get_running_loop()
        states = list(workers.values())
        for state in states:
            await loop.run_in_executor(None, close_worker_state, state)

        tasks = []
        for sid, task in list(clients.items()):
            if task is not None and not task.done():
                task.cancel()
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        workers.clear()
        clients.clear()
        server_logger.info("Graceful shutdown finished")


async def on_shutdown(app_: web.Application):
    del app_
    server_logger.info("aiohttp on_shutdown called")


app.on_shutdown.append(on_shutdown)


def run_aiohttp_server(port: int = 5000):
    server_logger.info("Starting local aiohttp server on port %s", port)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    runner = web.AppRunner(app)

    async def start():
        await runner.setup()
        site = web.TCPSite(runner, host="0.0.0.0", port=port)
        await site.start()
        server_logger.info(f"Server started http://127.0.0.1:{port}")

    stop_event = asyncio.Event()

    async def shutdown_and_stop():
        try:
            await graceful_shutdown("Server is shutting down (Ctrl+C)")
        finally:
            with contextlib.suppress(Exception):
                await runner.cleanup()
            stop_event.set()

    def handle_signal():
        server_logger.info("Signal received, scheduling graceful shutdown")
        loop.create_task(shutdown_and_stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, handle_signal)

    try:
        loop.run_until_complete(start())
        loop.run_until_complete(stop_event.wait())
    finally:
        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in pending:
            task.cancel()
        with contextlib.suppress(Exception):
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    run_aiohttp_server()
