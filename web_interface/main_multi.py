import json
import logging
import multiprocessing
import time
from multiprocessing import Pipe
from threading import Thread

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import uuid

from aux.data_info import DataInfo
from web_interface.back_front.frontend_client import FrontendClient
from web_interface.back_front.utils import WebInterfaceError, json_dumps, json_loads

app = Flask(__name__)
app.config['SECRET_KEY'] = '57916211bb0b13ce0c676dfde280ba245'
## Need to run redis server: sudo apt install redis-server
socketio = SocketIO(app, async_mode='threading', message_queue='redis://', cors_allowed_origins="*")

# Store active sessions
active_sessions = {}  # {session Id -> process, conn}


def worker_process(process_id, conn, sid):
    print(f"Process {process_id} started")
    # TODO problem is each process sends data to main process then to frontend.
    #  Easier to send it directly to url

    client = FrontendClient(sid)
    # client.socket.socket.send('hello from subprocess')

    def report(process_id):
        while True:
            print(f"Process {process_id} is working...")
            time.sleep(1)

    Thread(target=report, args=(process_id,)).start()

    while True:
        command = conn.recv()  # This blocks until a command is received
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

            conn.send(result)

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
                else:
                    result = client.mtBlock.do(do, args)

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
            conn.send(result)

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

            conn.send(result)

        elif type == "EXIT":
            break

    print(f"Process {process_id} received STOP command")
    # client.drop()


@socketio.on('connect')
def handle_connect():
    # FIXME create process not when socket connects but when a new tab is open
    session_id = str(uuid.uuid4())
    print('handle_connect', session_id, request.sid)
    emit('session_id', {'session_id': session_id})

    # Create a couple of connections
    parent_conn, child_conn = Pipe()

    # Start the worker process
    process = multiprocessing.Process(target=worker_process,
                                      args=(session_id, child_conn, request.sid))
    active_sessions[session_id] = process, parent_conn
    process.start()


@app.route('/')
def home():
    # FIXME ?
    DataInfo.refresh_all_data_info()
    return render_template('analysis.html')


@app.route('/analysis')
def analysis():
    return render_template('analysis.html')


@app.route('/interpretation')
def interpretation():
    return render_template('interpretation.html')


@app.route('/defense')
def defense():
    return render_template('defense.html')


@app.route("/drop", methods=['GET', 'POST'])
def drop():
    if request.method == 'POST':
        session_id = json.loads(request.data)['sessionId']
        if session_id not in active_sessions:
            raise WebInterfaceError(f"Session {session_id} is not active")
        process, conn = active_sessions[session_id]
        print("drop", session_id)

        # Stop corresponding process
        try:
            # Send stop command
            conn.send({'type': "STOP"})
        except Exception as e:
            print('exception:', e)

        # Wait for the process to terminate
        process.join(timeout=1)

        # If the process is still alive, terminate it
        if process.is_alive():
            print(f"Forcefully terminating process {session_id}")
            process.terminate()
            process.join(timeout=1)

        del active_sessions[session_id]
        return ''


@app.route("/ask", methods=['GET', 'POST'])
def storage():
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
def block():
    if request.method == 'POST':
        session_id = request.form.get('sessionId')
        assert session_id in active_sessions
        print('block request from', session_id)
        process, conn = active_sessions[session_id]

        conn.send({'type': 'block', 'args': request.form})
        return '{}'


@app.route("/<url>", methods=['GET', 'POST'])
def url(url):
    assert url in ['dataset', 'model', 'explainer']
    if request.method == 'POST':
        sid = request.form.get('sessionId')
        process, conn = active_sessions[sid]
        print(url, 'request from', sid)

        conn.send({'type': url, 'args': request.form})
        return conn.recv()


if __name__ == '__main__':
    # print(f"Async mode is: {socketio.async_mode}")
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)

    # TODO switch to 'run' in production
    #  In production mode the eventlet web server is used if available,
    #  else the gevent web server is used. If eventlet and gevent are not installed,
    #  the Werkzeug development web server is used.
    # app.run(debug=True, port=4568)

    # TODO Flask development web server is used, use eventlet or gevent,
    #  see https://flask-socketio.readthedocs.io/en/latest/deployment.html
    # socketio.run(app, host='0.0.0.0', debug=True, port=4567,
    #              allow_unsafe_werkzeug=True)
