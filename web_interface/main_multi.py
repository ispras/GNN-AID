import json
import logging
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import uuid

from aux.data_info import DataInfo
from web_interface.back_front.frontend_client import FrontendClient
from web_interface.back_front.utils import WebInterfaceError, json_dumps, json_loads

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Store active sessions
active_sessions = {}


@app.route('/')
def home():
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


@socketio.on('connect')
def handle_connect():
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = FrontendClient(socketio, request.sid)
    print('handle_connect', session_id, request.sid)
    emit('session_id', {'session_id': session_id})


@socketio.on('disconnect')
def handle_disconnect():
    for session_id, client in active_sessions.items():
        if client.sid == request.sid:
            client.drop()
            del active_sessions[session_id]
            print(f"Disconnected: {session_id}")
            break


# @socketio.on('heartbeat')
# def handle_heartbeat(data):
#     print('handle_heartbeat', data)
#     emit('heartbeat_response', {'session_id': data['session_id'], 'server_time': 123})
#     # emit('heartbeat_response', {'session_id': data['session_id'], 'server_time': socketio.time()})


@app.route("/ask", methods=['GET', 'POST'])
def storage():
    if request.method == 'POST':
        sid = request.form.get('sessionId')
        client = active_sessions[sid]
        print('request from', sid)
        ask = request.form.get('ask')

        if ask == "parameters":
            type = request.form.get('type')
            return json_dumps(client.get_parameters(type))

        else:
            raise WebInterfaceError(f"Unknown 'ask' command {ask}")


@app.route("/block", methods=['GET', 'POST'])
def block():
    if request.method == 'POST':
        sid = request.form.get('sessionId')
        client = active_sessions[sid]
        print('request from', sid)
        block = request.form.get('block')
        func = request.form.get('func')
        params = request.form.get('params')
        if params:
            params = json_loads(params)
        print(f"request_block: block={block}, func={func}, params={params}")
        client.request_block(block, func, params)
        return '{}'


@app.route("/dataset", methods=['GET', 'POST'])
def dataset():
    if request.method == 'POST':
        sid = request.form.get('sessionId')
        client = active_sessions[sid]
        print('request from', sid)
        get = request.form.get('get')
        set = request.form.get('set')
        part = request.form.get('part')
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
            return client.dcBlock.set_visible_part(part=part)

        if get == "data":
            dataset_data = client.dcBlock.get_dataset_data(part=part)
            data = json.dumps(dataset_data)
            logging.info(f"Length of dataset_data: {len(data)}")
            return data

        elif get == "var_data":
            if not client.dvcBlock.is_set():
                return ''
            dataset_var_data = client.dvcBlock.get_dataset_var_data(part=part)
            data = json.dumps(dataset_var_data)
            logging.info(f"Length of dataset_var_data: {len(data)}")
            return data

        elif get == "stat":
            stat = request.form.get('stat')
            return json_dumps(client.dcBlock.get_stat(stat))

        elif get == "index":
            return client.dcBlock.get_index()

        else:
            raise WebInterfaceError(f"Unknown 'part' command {get} for dataset")


@app.route("/model", methods=['GET', 'POST'])
def model():
    if request.method == 'POST':
        sid = request.form.get('sessionId')
        client = active_sessions[sid]
        print('request from', sid)
        do = request.form.get('do')
        get = request.form.get('get')

        if do:
            print(f"model.do: do={do}, params={request.form}")
            if do == 'index':
                type = request.form.get('type')
                if type == "saved":
                    return client.mloadBlock.get_index()
                if type == "custom":
                    return client.mcustomBlock.get_index()
            else:
                return client.mtBlock.do(do, request.form)

        if get:
            if get == "satellites":
                if client.mmcBlock.is_set():
                    part = request.form.get('part')
                    if part:
                        part = json_loads(part)
                    return client.mmcBlock.get_satellites(part=part)
                else:
                    return ''


@app.route("/explainer", methods=['GET', 'POST'])
def explainer():
    sid = request.form.get('sessionId')
    client = active_sessions[sid]
    print('request from', sid)
    # session.clear()
    if request.method == 'POST':
        do = request.form.get('do')

        print(f"explainer.do: do={do}, params={request.form}")

        if do in ["run", "stop"]:
            return client.erBlock.do(do, request.form)

        elif do == 'index':
            return client.elBlock.get_index()

        # elif do == "save":
        #     return client.save_explanation()

        else:
            raise WebInterfaceError(f"Unknown 'do' command {do} for explainer")


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)