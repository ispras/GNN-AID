from gevent import monkey; monkey.patch_all()  # Must go before other imports

import logging
import gevent.queue
from gevent import Greenlet
from multiprocessing import Queue

from flask import Flask, render_template, request, session
from flask_socketio import SocketIO
import uuid

from gnn_aid.aux.data_info import DataInfo
from web_interface.back_front.frontend_client import FrontendClient, ClientMode
from web_interface.back_front.utils import WebInterfaceError, json_dumps, json_loads, SocketConnect

app = Flask(__name__)
app.config['SECRET_KEY'] = '57916211bb0b13ce0c676dfde280ba245'
# Need to run redis server: sudo apt install redis-server
socketio = SocketIO(app, async_mode='gevent', message_queue='redis://', cors_allowed_origins="*")

# Store active sessions
active_sessions = {}  # {session Id -> greenlet (process), child_queue, parent_queue}


def worker_process(
        process_id: str,
        # conn: Connection,
        child_queue: Queue,
        parent_queue: Queue,
        mode: ClientMode
) -> None:
    print(f"Process {process_id} started")
    # TODO problem is each process sends data to main process then to frontend.
    #  Easier to send it directly to url

    sid = parent_queue.get()  # Wait until sid is received
    socket_connect = FlaskSocketConnect(sid, socketio)
    client = FrontendClient(socket_connect, mode)
    print(f"Created FrontendClient with sid {sid}")

    client.run_loop(child_queue, None, parent_queue)


@socketio.on('connect')
def handle_connect(
) -> None:
    session_id = session.get('session_id')
    print('handle_connect, session_id=', session_id, 'sid=', request.sid)

    _, child_queue, parent_queue = active_sessions[session_id]
    parent_queue.put(request.sid)


@socketio.on('disconnect')
def handle_disconnect(*args, **kwargs
) -> None:
    session_id = session.get('session_id')
    print('handle_disconnect, session_id=', session_id, 'sid=', request.sid)
    stop_session(session_id)


@app.route('/')
def home(
) -> str:
    DataInfo.refresh_all_data_info()

    mode = ClientMode.analysis
    session_id = start_client(mode)
    return render_template('analysis.html', sessionId=session_id, mode=mode.value)


@app.route('/analysis')
def analysis(
) -> str:
    mode = ClientMode.analysis
    session_id = start_client(mode)
    return render_template('analysis.html', sessionId=session_id, mode=mode.value)


@app.route('/interpretation')
def interpretation(
) -> str:
    mode = ClientMode.interpretation
    session_id = start_client(mode)
    return render_template('interpretation.html', sessionId=session_id, mode='lalala')


@app.route('/defense')
def defense(
) -> str:
    mode = ClientMode.defense
    session_id = start_client(mode)
    return render_template('defense.html', sessionId=session_id, mode=mode.value)


def start_client(
        mode: ClientMode
) -> str:
    # Generate unique session id and save it in flask.session
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    print('start_client', mode, 'session_id', session_id)

    # Create 2 gevent queues for communication
    parent_queue = gevent.queue.Queue()  # messages from main to worker
    child_queue = gevent.queue.Queue()  # messages from worker to main

    # Create and start greenlet (alternative to Process)
    greenlet = Greenlet.spawn(
        worker_process,
        session_id, child_queue, parent_queue, mode)

    active_sessions[session_id] = greenlet, child_queue, parent_queue

    return session_id


def stop_session(
        session_id: str
):
    print('stop_session', session_id)
    process, child_queue, parent_queue = active_sessions[session_id]

    # Stop corresponding process
    try:
        # Send stop command
        parent_queue.put({'type': "STOP"})
    except Exception as e:
        print('exception:', e)

    # Wait for the process to terminate
    process.join(timeout=1)

    # If the process is still alive, terminate it
    if not process.dead:
        print(f"Forcefully terminating process {session_id}")
        process.kill()
        process.join(timeout=1)

    del active_sessions[session_id]


@app.route("/ask", methods=['GET', 'POST'])
def storage(
) -> str:
    if request.method == 'POST':
        session_id = request.form.get('sessionId')
        assert session_id in active_sessions
        print('ask request from', session_id)
        ask = request.form.get('ask')

        if ask == "parameters":
            type = request.form.get('type')
            return json_dumps(FrontendClient.get_parameters(type))

        else:
            raise WebInterfaceError(f"Unknown 'ask' command {ask}")


@app.route("/block", methods=['GET', 'POST'])
def block(
) -> str:
    if request.method == 'POST':
        session_id = request.form.get('sessionId')
        _, _, parent_queue = active_sessions[session_id]
        print('block request from', session_id)
        parent_queue.put({'type': 'block', 'args': request.form})
        return '{}'


@app.route("/<url>", methods=['GET', 'POST'])
def url(
        url: str
) -> str:
    assert url in ['dataset', 'model', 'explainer']
    if request.method == 'POST':
        session_id = request.form.get('sessionId')
        _, child_queue, parent_queue = active_sessions[session_id]
        print(url, 'request from', session_id)

        parent_queue.put({'type': url, 'args': request.form})
        return child_queue.get()


class FlaskSocketConnect(SocketConnect):
    def __init__(self, sid: str, socket: SocketIO = None):
        super().__init__()
        if socket is None:
            self.socket = SocketIO(message_queue='redis://')
        else:
            self.socket = socket
        self.sid = sid

    def _send_data(self, data):
        self.socket.send(data, to=self.sid)


def run_flask_server(port=5000, debug=False):
    # print(f"Async mode is: {socketio.async_mode}")
    app.debug = debug

    # For development
    if app.debug:
        socketio.run(app, host='0.0.0.0', port=port, debug=True)
        print('Flask DEBUG')
        print(f'Go to http://127.0.0.1:{port}/')
    # For production
    else:
        from gevent import pywsgi
        from geventwebsocket.handler import WebSocketHandler
        server = pywsgi.WSGIServer(('0.0.0.0', port), app, handler_class=WebSocketHandler)
        server.serve_forever()


if __name__ == '__main__':
    run_flask_server()
