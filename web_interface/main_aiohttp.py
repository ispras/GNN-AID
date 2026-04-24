import asyncio
import contextlib
import ctypes
import json
import logging
import os
import queue
import shutil
import signal
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any
from typing import Dict
from unittest import mock

import aiohttp_jinja2
import jinja2
import socketio
from aiohttp import web
from aiohttp.web_request import FileField

from web_interface.back_front import json_dumps
from web_interface.back_front.frontend_client import ClientMode, FrontendClient
from web_interface.back_front.utils import SocketConnect, STATIC_DIR, TEMPLATES_DIR, LOG_DIR, \
    get_sid_logger, CLIENTS_STORAGE_ROOT, CLIENT_META_FILENAME, \
    CLIENT_STORAGE_TTL, CLEANUP_INTERVAL_SEC

LOG_FILE = LOG_DIR / datetime.now().strftime("server_%Y-%m-%d_%H-%M-%S.log")
CLIENT_RESTART_DELAY_SEC = 30
GRACEFUL_WORKER_STOP_SEC = 2.0
FORCEFUL_WORKER_KILL_SEC = 1.0
DIR_PATCH_MODULES = [
    'gnn_aid.aux.utils',
    'gnn_aid.aux.data_info',
    'gnn_aid.aux.declaration',
    'web_interface.back_front.model_blocks',
    'web_interface.back_front.explainer_blocks',
]
MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200 MB

shutdown_lock = asyncio.Lock()
shutdown_started = False


class SidFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "sid"):
            record.sid = "-"
        return True


def setup_logging(force: bool = False) -> logging.Logger:
    logger = logging.getLogger("gnn_aid.web")

    if logger.handlers and not force:
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [pid=%(process)d] [sid=%(sid)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.setLevel(logging.INFO)
    # logger.propagate = False

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
    ping_timeout=600,  # wait pong from client
    ping_interval=25,
    cors_allowed_origins='*',
    serializer='msgpack'
)
app = web.Application(client_max_size=MAX_UPLOAD_SIZE)
sio.attach(app)

aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(TEMPLATES_DIR))

clients: Dict[str, asyncio.Task] = {}
workers: Dict[str, Dict[str, Any]] = {}
cleanup_task: asyncio.Task | None = None


def queue_get_with_timeout(q, timeout=0.5):
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


def make_worker_state(mode: ClientMode, client_id: str) -> Dict[str, Any]:
    client_dirs = ensure_client_dirs(client_id)

    return {
        "mode": mode,
        "client_id": client_id,
        "client_dirs": client_dirs,
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

    # 1) Пытаемся завершить штатно
    with contextlib.suppress(Exception):
        request_queue.put({"type": "STOP", "args": {}})

    if proc is not None:
        # 2) Немного ждем мягкого выхода
        if proc.is_alive():
            proc.join(timeout=GRACEFUL_WORKER_STOP_SEC)

        # 3) Если не вышел - terminate()
        if proc.is_alive():
            server_logger.warning("Worker pid=%s did not stop gracefully, terminating", proc.pid)
            with contextlib.suppress(Exception):
                proc.terminate()
            proc.join(timeout=FORCEFUL_WORKER_KILL_SEC)

        # 4) Если все еще жив - kill
        if proc.is_alive():
            server_logger.error("Worker pid=%s survived terminate(), killing", proc.pid)
            with contextlib.suppress(Exception):
                os.kill(proc.pid, signal.SIGKILL)
            proc.join(timeout=FORCEFUL_WORKER_KILL_SEC)

        # 5) Последняя проверка
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
            state["client_id"],
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
        "Started worker for sid=%s client_id=%s pid=%s mode=%s",
        sid, state["client_id"], proc.pid, state["mode"].value
    )
    return state


def replace_worker_for_sid(sid: str) -> Dict[str, Any]:
    old_state = workers[sid]
    mode = old_state["mode"]
    client_id = old_state["client_id"]

    server_logger.info("Replacing worker state for sid=%s client_id=%s", sid, client_id)
    close_worker_state(old_state)

    new_state = make_worker_state(mode, client_id)
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

        server_logger.exception("Restarting only sid=%s because of backend error: %s", sid, error_text)

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
    return {
        "request": request,
        "mode": ClientMode.interpretation.value
    }


# Route for home and analysis
@aiohttp_jinja2.template("analysis.html")
async def handle_analysis(request):
    return {
        "request": request,
        "mode": ClientMode.analysis.value
    }


# Route for defense
@aiohttp_jinja2.template("defense.html")
async def handle_defense(request):
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

    server_logger.info("ask request from sid=%s", sid)
    ask_cmd = data.get('ask')

    if ask_cmd == "parameters":
        type_ = data.get('type')
        params = FrontendClient.get_parameters(type_)
        params = json_dumps(params)
        return web.Response(text=params)

    return web.Response(status=400, text=f"Unknown 'ask' command {ask_cmd}")


async def handle_url(request):
    url = request.match_info.get("url")
    server_logger.info("url=%s", url)

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

    client_id = state["client_id"]
    touch_client_access(client_id)

    response_queue = state["response_queue"]
    request_queue = state["request_queue"]
    proc = state["proc"]

    if proc is None or not proc.is_alive():
        return web.Response(status=503, text="Worker is not alive")

    args = {key: value for key, value in data.items() if not isinstance(value, FileField)}

    # Специальный случай: загрузка файлов датасета.
    # В очередь worker-а нельзя класть FileField, поэтому сохраняем файлы на диск,
    # а worker-у передаем только обычный dict со строками.
    if url == "dataset" and "uploadId" in args:
        upload_id = args.get("uploadId")
        upload_dir = make_upload_dir(client_id, upload_id)

        # todo clear_before=False when we can process many files
        files = save_uploaded_files(data, upload_dir, clear_before=True)

        args["upload_dir"] = str(upload_dir)
        args["files"] = files

        server_logger.info(
            "%s upload request from sid=%s upload_dir=%s files=%s",
            url,
            sid,
            upload_dir,
            [f["relative_path"] for f in files],
        )
    else:
        server_logger.info("%s http request from sid=%s args=%s", url, sid, args)

    request_queue.put({
        "type": url,
        "args": args,
    })

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


def safe_relative_upload_path(filename: str) -> Path:
    """Безопасно превращает имя файла из multipart в относительный путь.

    Нужен, чтобы сохранить структуру папки из webkitRelativePath,
    но не дать имени файла выйти за пределы upload_dir через ../.
    """
    filename = str(filename or "unnamed")
    filename = filename.replace("\\", "/")

    parts = []

    for part in filename.split("/"):
        part = part.strip()

        if not part:
            continue

        if part in (".", ".."):
            continue

        # На всякий случай убираем совсем странные символы из имени сегмента пути.
        safe_part = "".join(
            ch for ch in part
            if ch not in '<>:"|?*\0'
        )

        if safe_part:
            parts.append(safe_part)

    if not parts:
        return Path("unnamed")

    return Path(*parts)


def make_upload_dir(client_id: str, upload_id: str | None = None) -> Path:
    if not upload_id:
        upload_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    upload_dir = get_client_root(client_id) / "uploads" / f"Upload_{upload_id}"
    upload_dir.mkdir(parents=True, exist_ok=True)

    return upload_dir


def save_uploaded_files(post, upload_dir: Path, clear_before: bool = False) -> list[dict[str, str | None]]:
    files = []

    for field in post.getall("files", []):
        if not isinstance(field, FileField):
            continue

        if clear_before:
            shutil.rmtree(upload_dir)
            clear_before = False

        rel_path = safe_relative_upload_path(field.filename)
        dst = upload_dir / rel_path

        dst.parent.mkdir(parents=True, exist_ok=True)

        field.file.seek(0)

        with dst.open("wb") as f:
            shutil.copyfileobj(field.file, f)

        files.append({
            "name": Path(rel_path).name,
            "relative_path": str(rel_path),
            "path": str(dst),
            "content_type": field.content_type,
        })

    return files


app.router.add_get("/", handle_analysis)
app.router.add_get("/analysis", handle_analysis)
app.router.add_get("/defense", handle_defense)
app.router.add_get("/interpretation", handle_interpretation)
app.router.add_post("/ask", handle_ask)
app.router.add_post("/{url}", handle_url)

app.router.add_static('/static/', path=str(STATIC_DIR), name='static')


@sio.event
async def connect(sid, environ, auth=None):
    query_string = environ.get('QUERY_STRING', '')
    query = dict(qc.split('=', 1) for qc in query_string.split('&') if '=' in qc)
    mode = ClientMode(query.get('mode', None))

    client_id = get_client_id_from_environ(sid, environ, auth)
    touch_client_access(client_id)

    sid_logger = get_sid_logger(sid)
    sid_logger.info("Client connected, mode=%s client_id=%s", mode.value, client_id)

    workers[sid] = make_worker_state(mode, client_id)
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
        sid_logger.info("client_wrapper started")
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
    client_id: str,
    response_queue: Queue,
    msg_queue: Queue,
    request_queue: Queue,
    mode: ClientMode
) -> None:
    # We ignore SIGINT in worker - it is for server
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Additional safety: Чтобы worker не становился сиротой, если главный процесс умер внезапно
    with contextlib.suppress(Exception):
        set_pdeathsig(signal.SIGKILL)

    setup_logging(force=True)
    logger = get_sid_logger(sid)

    logger.info("Process started for client_id=%s", client_id)

    try:
        with patched_client_dirs(client_id) as client_dirs:
            logger.info(
                "Patched dirs for client_id=%s data=%s models=%s explanations=%s",
                client_id,
                client_dirs["DATASETS_DIR"],
                client_dirs["MODELS_DIR"],
                client_dirs["EXPLANATIONS_DIR"],
            )

            socket_connect = AiohttpSocketConnect(msg_queue, sid)
            client = FrontendClient(socket_connect, mode, sid)
            logger.info("Created FrontendClient")

            client.run_loop(response_queue, msg_queue, request_queue)

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Unhandled exception in worker_process")

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
    def __init__(self, queue: Queue, sid: str):
        super().__init__()
        self.mp_queue = queue
        self.logger = get_sid_logger(sid)

    def _send_data(self, data):
        self.mp_queue.put_nowait(data)
        self.logger.debug("put msg to mpqueue [len=%s] '%s'", len(str(data)), str(data)[:100])


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


async def graceful_shutdown(reason: str = "Server is shutting down"):
    global shutdown_started, cleanup_task

    async with shutdown_lock:
        if shutdown_started:
            return
        shutdown_started = True

        server_logger.info("Graceful shutdown started: %s", reason)

        # 1. Сообщаем фронту, пока transport еще жив
        await emit_fatal_stop_to_all(reason)

        # Даем шанс сообщению дойти
        await asyncio.sleep(0.7)

        # 2. Закрываем socket-соединения
        for sid in list(clients.keys()):
            with contextlib.suppress(Exception):
                await sio.disconnect(sid)

        # 3. Останавливаем worker-процессы
        loop = asyncio.get_running_loop()
        states = list(workers.values())
        for state in states:
            await loop.run_in_executor(None, close_worker_state, state)

        # 4. Отменяем client_wrapper tasks
        tasks = []
        for sid, task in list(clients.items()):
            if task is not None and not task.done():
                task.cancel()
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        if cleanup_task is not None:
            cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup_task
            cleanup_task = None

        workers.clear()
        clients.clear()

        server_logger.info("Graceful shutdown finished")


async def on_shutdown(app):
    server_logger.info("aiohttp on_shutdown called")


app.on_shutdown.append(on_shutdown)


def run_aiohttp_server(port=5000):
    server_logger.info("Starting aiohttp server on port %s", port)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    runner = web.AppRunner(app)

    async def start():
        await runner.setup()
        site = web.TCPSite(runner, host="0.0.0.0", port=port)
        await site.start()
        global cleanup_task
        cleanup_task = asyncio.create_task(cleanup_stale_client_storage_loop())
        server_logger.info("Server started on port %s", port)

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
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for task in pending:
            task.cancel()
        with contextlib.suppress(Exception):
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

# ========== Personal storage for each client

def sanitize_client_id(client_id: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    sanitized = "".join(ch for ch in str(client_id) if ch in allowed)
    if not sanitized:
        raise ValueError("Empty client_id after sanitizing")
    return sanitized


def get_client_id_from_environ(sid: str, environ: dict, auth=None) -> str:
    server_logger.info("connect auth type for sid=%s: %s", sid, type(auth).__name__)

    if isinstance(auth, dict):
        client_id = auth.get("client_id")
        if client_id:
            return sanitize_client_id(client_id)

    query_string = environ.get("QUERY_STRING", "")
    query = dict(
        qc.split("=", 1)
        for qc in query_string.split("&")
        if "=" in qc
    )

    client_id = query.get("client_id")
    if client_id:
        return sanitize_client_id(client_id)

    return sanitize_client_id(f"sid_{sid}")


def ensure_client_dirs(client_id: str) -> dict[str, Path]:
    from gnn_aid.aux.utils import root_dir, GRAPHS_DIR, DATASETS_DIR, MODELS_DIR, EXPLANATIONS_DIR, DATA_INFO_DIR

    CLIENTS_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    client_root = CLIENTS_STORAGE_ROOT / client_id
    client_root.mkdir(parents=True, exist_ok=True)

    mapping = {
        "root_dir": client_root,
        "GRAPHS_DIR": client_root / GRAPHS_DIR.name,
        "DATASETS_DIR": client_root / DATASETS_DIR.name,
        "MODELS_DIR": client_root / MODELS_DIR.name,
        "EXPLANATIONS_DIR": client_root / EXPLANATIONS_DIR.name,
        "DATA_INFO_DIR": client_root / DATA_INFO_DIR.name,
    }

    source_mapping = {
        "GRAPHS_DIR": (GRAPHS_DIR, [
            Path('example') / 'example',
            Path('example') / 'example3',
            Path('example') / 'example3_gml',
        ])
    }

    for key, dst in mapping.items():
        if dst.exists():
            continue
        if key not in source_mapping:
            continue

        src, subpaths = source_mapping[key]

        if src and src.exists():
            print(subpaths)
            for sub in subpaths:
                dst_sub = dst / sub
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src / sub, dst_sub)
        else:
            dst.mkdir(parents=True, exist_ok=True)

    return {
        "client_root": client_root,
        **mapping,
    }


def build_dir_patchers(client_dirs: dict[str, Path]) -> list[mock._patch]:
    patchers = []

    for var_name in ("root_dir", "GRAPHS_DIR", "DATASETS_DIR", "MODELS_DIR", "EXPLANATIONS_DIR", "DATA_INFO_DIR"):
        patched_path = client_dirs[var_name]
        for module in DIR_PATCH_MODULES:
            patchers.append(mock.patch(f"{module}.{var_name}", patched_path))

    return patchers


@contextlib.contextmanager
def patched_client_dirs(client_id: str):
    client_dirs = ensure_client_dirs(client_id)
    patchers = build_dir_patchers(client_dirs)

    try:
        for patcher in patchers:
            try:
                patcher.start()
            except AttributeError: pass
        yield client_dirs
    finally:
        for patcher in reversed(patchers):
            with contextlib.suppress(Exception):
                patcher.stop()

# ========== Automatic cleanup

def utc_now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def get_client_root(client_id: str) -> Path:
    return CLIENTS_STORAGE_ROOT / client_id


def get_client_meta_path(client_id: str) -> Path:
    return get_client_root(client_id) / CLIENT_META_FILENAME


def read_client_meta(client_id: str) -> dict:
    meta_path = get_client_meta_path(client_id)
    if not meta_path.exists():
        return {}

    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        server_logger.exception("Failed to read client meta for client_id=%s", client_id)
        return {}


def write_client_meta(client_id: str, meta: dict) -> None:
    client_root = get_client_root(client_id)
    client_root.mkdir(parents=True, exist_ok=True)

    meta_path = get_client_meta_path(client_id)
    tmp_path = meta_path.with_suffix(meta_path.suffix + ".tmp")

    tmp_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(meta_path)


def touch_client_access(client_id: str) -> None:
    meta = read_client_meta(client_id)
    meta["client_id"] = client_id
    meta["last_access_ts"] = utc_now_ts()
    write_client_meta(client_id, meta)


def is_client_id_active(client_id: str) -> bool:
    for state in workers.values():
        if state.get("client_id") == client_id:
            return True
    return False


def remove_stale_client_storage_once() -> None:
    """ Удаление просроченных хранилищ
    """
    server_logger.info("Checking for outdated client storages...")

    CLIENTS_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    now_ts = utc_now_ts()

    for path in CLIENTS_STORAGE_ROOT.iterdir():
        if not path.is_dir():
            continue

        client_id = path.name

        if is_client_id_active(client_id):
            continue

        meta = read_client_meta(client_id)
        last_access_ts = meta.get("last_access_ts")

        # если меты нет, можно использовать mtime каталога как fallback
        if last_access_ts is None:
            try:
                last_access_ts = path.stat().st_mtime
            except Exception:
                server_logger.exception("Failed to stat client dir %s", path)
                continue

        age_sec = now_ts - float(last_access_ts)
        if age_sec < CLIENT_STORAGE_TTL.total_seconds():
            continue

        try:
            shutil.rmtree(path)
            server_logger.info(
                "Deleted stale client storage: client_id=%s path=%s age_days=%.1f",
                client_id, path, age_sec / 86400,
            )
        except Exception:
            server_logger.exception(
                "Failed to delete stale client storage for client_id=%s path=%s",
                client_id, path,
            )


async def cleanup_stale_client_storage_loop():
    """ Фоновая задача очистки
    """
    loop = asyncio.get_running_loop()

    while not shutdown_started:
        try:
            await loop.run_in_executor(None, remove_stale_client_storage_once)
        except asyncio.CancelledError:
            raise
        except Exception:
            server_logger.exception("Unhandled exception in cleanup_stale_client_storage_loop")

        try:
            await asyncio.sleep(CLEANUP_INTERVAL_SEC)
        except asyncio.CancelledError:
            raise

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    run_aiohttp_server()
