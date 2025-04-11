from time import sleep

from gevent import monkey
monkey.patch_all(subprocess=True, os=False, select=True)

import json
import logging
import multiprocessing
from multiprocessing import Pipe, Queue
from multiprocessing.connection import Connection

from flask import Flask, render_template, request, session
from flask_socketio import SocketIO, emit
import uuid

from aux.data_info import DataInfo
from web_interface.back_front.frontend_client import FrontendClient, ClientMode
from web_interface.back_front.utils import WebInterfaceError, json_dumps, json_loads

app = Flask(__name__)
app.config['SECRET_KEY'] = '57916211bb0b13ce0c676dfde280ba245'
## Need to run redis server: sudo apt install redis-server
# socketio = SocketIO(app, cors_allowed_origins="*")
socketio = SocketIO(app, async_mode='gevent', message_queue='redis://', cors_allowed_origins="*")

# Store active sessions
active_sessions = {}  # {session Id -> sid, process, conn}


def worker_process(
        process_id: str,
        # conn: Connection,
        child_queue: Queue,
        parent_queue: Queue,
        sid: str,
        mode: ClientMode
) -> None:
    print(f"Process {process_id} started")
    # TODO problem is each process sends data to main process then to frontend.
    #  Easier to send it directly to url

    print("wait sid from parent_queue")
    sid = parent_queue.get()  # Wait until sid is received
    print("got sid from parent_queue")
    # sid = conn.recv()  # Wait until sid is received
    client = FrontendClient(sid, mode)
    print(f"Created FrontendClient with sid {sid}")

    # client.socket.socket.send('hello from subprocess')

    # import time
    # from threading import Thread
    # 
    # def report(process_id):
    #     while True:
    #         print(f"Process {process_id} is working...")
    #         time.sleep(1)
    #
    # Thread(target=report, args=(process_id,)).start()

    while True:
        print("wait command from parent_queue", parent_queue)
        # command = conn.recv()  # This blocks until a command is received
        command = parent_queue.get()  # This blocks until a command is received
        print("got command from parent_queue", command)
        type = command.get('type')
        args = command.get('args')
        print(f"Received command: {type} with args: {args}")

        if type == "dataset":
            get = args.get('get')
            set = args.get('set')
            part = args.get('part')
            if part:
                part = json_loads(part)

            # # FIXME tmp
            #
            # from web_interface.back_front.communication import SocketConnect, WebInterfaceError
            # socket = SocketConnect(socket=socketio)
            # for i in range(300):
            #     print('sending', i, 'big')
            #     socket.send(i, 1000000 * "x")
            #     # print('sending', i, 'small')
            #     # socket.send(i, "small")
            #     sleep(0.5/25)

            if set == "visible_part":
                result = client.dcBlock.set_visible_part(part=part)

            elif get == "data":
                dataset_data = client.dcBlock.get_dataset_data(part=part)
                data = json.dumps(dataset_data)
                logging.info(f"Length of dataset_data: {len(data)}")
                result = data

            elif get == "var_data":
                if not client.dvcBlock.is_set():
                    result = ''
                else:
                    dataset_var_data = client.dvcBlock.get_dataset_var_data(part=part)
                    data = json.dumps(dataset_var_data)
                    logging.info(f"Length of dataset_var_data: {len(data)}")
                    result = data

            elif get == "stat":
                stat = args.get('stat')
                result = json_dumps(client.dcBlock.get_stat(stat))

            elif get == "index":
                result = client.dcBlock.get_index()

            else:
                raise WebInterfaceError(f"Unknown 'part' command {get} for dataset")

            child_queue.put(result)
            # conn.send(result)

        elif type == "block":
            block = args.get('block')
            func = args.get('func')
            params = args.get('params')
            if params:
                params = json_loads(params)
            print(f"request_block: block={block}, func={func}, params={params}")
            # TODO what if raise exception? process will stop
            client.request_block(block, func, params)
            # conn.send('{}')

        elif type == "model":
            do = args.get('do')
            get = args.get('get')

            if do:
                print(f"model.do: do={do}, params={args}")
                if do == 'index':
                    type = args.get('type')
                    if type == "saved":
                        result = client.mloadBlock.get_index()
                    elif type == "custom":
                        result = client.mcustomBlock.get_index()
                elif do in ['train', 'reset', 'run', 'save']:
                    result = client.mtBlock.do(do, args)
                elif do in ['run with attacks']:
                    result = client.atBlock.do(do, args)
                else:
                    raise WebInterfaceError(f"Unknown do command: '{do}'")

            if get:
                if get == "satellites":
                    if client.mmcBlock.is_set():
                        part = args.get('part')
                        if part:
                            part = json_loads(part)
                        result = client.mmcBlock.get_satellites(part=part)
                    else:
                        result = ''

            assert result is not None
            child_queue.put(result)
            # conn.send(result)

        elif type == "explainer":
            do = args.get('do')

            print(f"explainer.do: do={do}, params={args}")

            if do in ["run", "stop"]:
                result = client.erBlock.do(do, args)

            elif do == 'index':
                result = client.elBlock.get_index()

            # elif do == "save":
            #     return client.save_explanation()

            else:
                raise WebInterfaceError(f"Unknown 'do' command {do} for explainer")

            child_queue.put(result)
            # conn.send(result)

        elif type == "EXIT":
            break

    print(f"Process {process_id} received STOP command")
    # client.drop()


@socketio.on('connect')
def handle_connect(
) -> None:
    # sid = request.args.get('sid')
    session_id = session.get('session_id')
    print('handle_connect, session_id=', session_id, 'sid=', request.sid)

    _, _, child_queue, parent_queue = active_sessions[session_id]
    parent_queue.put(request.sid)
    print('put sid in parent queue')
    # _, _, parent_conn = active_sessions[session_id]
    # parent_conn.send(request.sid)


# @socketio.on('disconnect')
# def handle_disconnect(
# ) -> None:
#     session_id = session.get('session_id')
#     print('handle_disconnect, session_id=', session_id, 'sid=', request.sid)
#     stop_session(session_id)
#
#     # for session_id, (sid, process, parent_conn) in active_sessions.items():
#     #     if sid == request.sid:
#     #         # print(f"Disconnected: {session_id}")
#     #         stop_session(session_id)
#     #         break
#

@app.route('/')
def home(
) -> str:
    # FIXME ?
    DataInfo.refresh_all_data_info()

    mode = ClientMode.analysis
    session_id = start_client(mode)
    return render_template('analysis.html', session_id=session_id, mode=mode.value)


# @app.route('/analysis')
# def analysis(
# ) -> str:
#     mode = ClientMode.analysis
#     session_id = start_client(mode)
#     return render_template('analysis.html', session_id=session_id, mode=mode.value)
#

# @app.route('/interpretation')
# def interpretation(
# ) -> str:
#     mode = ClientMode.interpretation
#     session_id = start_client(mode)
#     return render_template('interpretation.html', session_id=session_id, mode='lalala')
#
#
# @app.route('/defense')
# def defense(
# ) -> str:
#     mode = ClientMode.defense
#     session_id = start_client(mode)
#     return render_template('defense.html', session_id=session_id, mode=mode.value)


def start_client(
        mode: ClientMode
) -> str:
    # Generate unique session id and save it in flask.session
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    print('start_client', mode, 'session_id', session_id)

    # Create a couple of connections
    # parent_conn, child_conn = Pipe()
    child_queue = Queue()
    parent_queue = Queue()

    # Create the worker process
    process = multiprocessing.Process(
        target=worker_process,
        args=(session_id, child_queue, parent_queue, session_id, mode))
        # args=(session_id, child_conn, session_id, mode))
    active_sessions[session_id] = session_id, process, child_queue, parent_queue
    # active_sessions[session_id] = session_id, process, parent_conn

    process.start()

    return session_id


# @app.route("/drop", methods=['GET', 'POST'])
# def drop(
# ) -> str:
#     # FIXME we don't need it, since process is stopped in handle_disconnect
#     if request.method == 'POST':
#         session_id = json.loads(request.data)['sessionId']
#         if session_id in active_sessions:
#             # raise WebInterfaceError(f"Session {session_id} is not active")
#             stop_session(session_id)
#         return ''

#
# def stop_session(
#         session_id: str
# ):
#     _, process, child_queue, parent_queue = active_sessions[session_id]
#
#     # Stop corresponding process
#     try:
#         # Send stop command
#         # conn.send({'type': "STOP"})
#         parent_queue.put({'type': "STOP"})
#     except Exception as e:
#         print('exception:', e)
#
#     # Wait for the process to terminate
#     process.join(timeout=1)
#
#     # If the process is still alive, terminate it
#     if process.is_alive():
#         print(f"Forcefully terminating process {session_id}")
#         process.terminate()
#         process.join(timeout=1)
#
#     del active_sessions[session_id]


# @app.route("/ask", methods=['GET', 'POST'])
# def storage(
# ) -> str:
#     if request.method == 'POST':
#         session_id = request.form.get('sessionId')
#         assert session_id in active_sessions
#         print('ask request from', session_id)
#         ask = request.form.get('ask')
#
#         if ask == "parameters":
#             type = request.form.get('type')
#             return json_dumps(FrontendClient.get_parameters(type))
#
#         else:
#             raise WebInterfaceError(f"Unknown 'ask' command {ask}")


# @app.route("/block", methods=['GET', 'POST'])
# def block(
# ) -> str:
#     if request.method == 'POST':
#         session_id = request.form.get('sessionId')
#         assert session_id in active_sessions
#         print('block request from', session_id)
#         _, process, child_queue, parent_queue = active_sessions[session_id]
#
#         # conn.send({'type': 'block', 'args': request.form})
#         parent_queue.put({'type': 'block', 'args': request.form})
#         return '{}'


@app.route("/<url>", methods=['GET', 'POST'])
def url(
        url: str
) -> str:
    assert url in ['dataset', 'model', 'explainer']
    if request.method == 'POST':
        session_id = request.form.get('sessionId')
        _, process, child_queue, parent_queue = active_sessions[session_id]
        print(url, 'request from', session_id)

        # conn.send({'type': url, 'args': request.form})
        parent_queue.put({'type': url, 'args': request.form})

        sleep(3)
        parent_queue.put('just some msg')

        print('put request in parent queue', parent_queue)
        res = child_queue.get()
        print('got res from child queue')

        return res
        # return conn.recv()


if __name__ == '__main__':
    # print(f"Async mode is: {socketio.async_mode}")
    # socketio.run(app, debug=True, allow_unsafe_werkzeug=True)

    socketio.run(app, host='0.0.0.0')

    # TODO switch to 'run' in production
    #  In production mode the eventlet web server is used if available,
    #  else the gevent web server is used. If eventlet and gevent are not installed,
    #  the Werkzeug development web server is used.
    # app.run(debug=True, port=4568)

    # TODO Flask development web server is used, use eventlet or gevent,
    #  see https://flask-socketio.readthedocs.io/en/latest/deployment.html
    # socketio.run(app, host='0.0.0.0', debug=True, port=4567,
    #              allow_unsafe_werkzeug=True)
