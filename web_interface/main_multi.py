from gevent import monkey; monkey.patch_all()  # Must go before other imports

import json
import logging
import gevent.queue
from gevent import Greenlet
from multiprocessing import Pipe, Queue

from flask import Flask, render_template, request, session
from flask_socketio import SocketIO, emit
import uuid

from aux.data_info import DataInfo
from web_interface.back_front.frontend_client import FrontendClient, ClientMode
from web_interface.back_front.utils import WebInterfaceError, json_dumps, json_loads

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
    client = FrontendClient(sid, mode)
    print(f"Created FrontendClient with sid {sid}")

    while True:
        command = parent_queue.get()  # This blocks until a command is received
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

        elif type == "block":
            block = args.get('block')
            func = args.get('func')
            params = args.get('params')
            if params:
                params = json_loads(params)
            print(f"request_block: block={block}, func={func}, params={params}")
            # TODO what if raise exception? process will stop
            client.request_block(block, func, params)

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

        elif type == "STOP":
            print(f"Process {process_id} received STOP command; it will finish")
            break


@socketio.on('connect')
def handle_connect(
) -> None:
    session_id = session.get('session_id')
    print('handle_connect, session_id=', session_id, 'sid=', request.sid)

    _, child_queue, parent_queue = active_sessions[session_id]
    parent_queue.put(request.sid)


@socketio.on('disconnect')
def handle_disconnect(
) -> None:
    session_id = session.get('session_id')
    print('handle_disconnect, session_id=', session_id, 'sid=', request.sid)
    stop_session(session_id)


@app.route('/')
def home(
) -> str:
    # FIXME ?
    DataInfo.refresh_all_data_info()

    mode = ClientMode.analysis
    session_id = start_client(mode)
    return render_template('analysis.html', session_id=session_id, mode=mode.value)


@app.route('/analysis')
def analysis(
) -> str:
    mode = ClientMode.analysis
    session_id = start_client(mode)
    return render_template('analysis.html', session_id=session_id, mode=mode.value)


@app.route('/interpretation')
def interpretation(
) -> str:
    mode = ClientMode.interpretation
    session_id = start_client(mode)
    return render_template('interpretation.html', session_id=session_id, mode='lalala')


@app.route('/defense')
def defense(
) -> str:
    mode = ClientMode.defense
    session_id = start_client(mode)
    return render_template('defense.html', session_id=session_id, mode=mode.value)


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


if __name__ == '__main__':
    # print(f"Async mode is: {socketio.async_mode}")
    app.debug = True

    # For development
    if app.debug:
        print('Flask DEBUG')
        socketio.run(app, host='0.0.0.0', debug=False)
    # For production
    else:
        from gevent import pywsgi
        from geventwebsocket.handler import WebSocketHandler
        server = pywsgi.WSGIServer(('0.0.0.0', 5000), app, handler_class=WebSocketHandler)
        server.serve_forever()