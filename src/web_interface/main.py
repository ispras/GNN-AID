import json
import logging
from time import sleep

from flask import Flask, render_template, request
from flask_socketio import SocketIO

from aux.data_info import DataInfo
from web_interface.back_front.frontend_client import FrontendClient
from web_interface.back_front.utils import WebInterfaceError, json_dumps, json_loads

app = Flask(__name__)
app.config['SECRET_KEY'] = '57916211bb0b13ce0c676dfde280ba245'
socketio = SocketIO(app, async_mode='threading', message_queue='redis://')
FRONTEND_CLIENT = FrontendClient(socketio)


@app.route("/")
@app.route("/home", methods=['GET', 'POST'])
def home():
    # Drop all data at page reload
    FRONTEND_CLIENT.drop()
    DataInfo.refresh_all_data_info()
    return render_template('home.html')


@app.route("/ask", methods=['GET', 'POST'])
def storage():
    if request.method == 'POST':
        ask = request.form.get('ask')

        if ask == "parameters":
            type = request.form.get('type')
            return json_dumps(FRONTEND_CLIENT.get_parameters(type))

        else:
            raise WebInterfaceError(f"Unknown 'ask' command {ask}")


@app.route("/block", methods=['GET', 'POST'])
def block():
    if request.method == 'POST':
        block = request.form.get('block')
        func = request.form.get('func')
        params = request.form.get('params')
        if params:
            params = json_loads(params)
        print(f"request_block: block={block}, func={func}, params={params}")
        FRONTEND_CLIENT.request_block(block, func, params)
        return '{}'


@app.route("/dataset", methods=['GET', 'POST'])
def dataset():
    if request.method == 'POST':
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
            return FRONTEND_CLIENT.dcBlock.set_visible_part(part=part)

        if get == "data":
            dataset_data = FRONTEND_CLIENT.dcBlock.get_dataset_data(part=part)
            data = json.dumps(dataset_data)
            logging.info(f"Length of dataset_data: {len(data)}")
            return data

        elif get == "var_data":
            if not FRONTEND_CLIENT.dvcBlock.is_set():
                return ''
            dataset_var_data = FRONTEND_CLIENT.dvcBlock.get_dataset_var_data(part=part)
            data = json.dumps(dataset_var_data)
            logging.info(f"Length of dataset_var_data: {len(data)}")
            return data

        elif get == "stat":
            stat = request.form.get('stat')
            return json_dumps(FRONTEND_CLIENT.dcBlock.get_stat(stat))

        elif get == "index":
            return FRONTEND_CLIENT.dcBlock.get_index()

        else:
            raise WebInterfaceError(f"Unknown 'part' command {get} for dataset")


@app.route("/model", methods=['GET', 'POST'])
def model():
    if request.method == 'POST':
        do = request.form.get('do')
        get = request.form.get('get')

        if do:
            print(f"model.do: do={do}, params={request.form}")
            if do == 'index':
                type = request.form.get('type')
                if type == "saved":
                    return FRONTEND_CLIENT.mloadBlock.get_index()
                if type == "custom":
                    return FRONTEND_CLIENT.mcustomBlock.get_index()
            else:
                return FRONTEND_CLIENT.mtBlock.do(do, request.form)

        if get:
            if get == "satellites":
                if FRONTEND_CLIENT.mmcBlock.is_set():
                    part = request.form.get('part')
                    if part:
                        part = json_loads(part)
                    return FRONTEND_CLIENT.mmcBlock.get_satellites(part=part)
                else:
                    return ''


@app.route("/explainer", methods=['GET', 'POST'])
def explainer():
    # session.clear()
    if request.method == 'POST':
        do = request.form.get('do')

        print(f"explainer.do: do={do}, params={request.form}")

        if do in ["run", "stop"]:
            return FRONTEND_CLIENT.erBlock.do(do, request.form)

        elif do == 'index':
            return FRONTEND_CLIENT.elBlock.get_index()

        # elif do == "save":
        #     return FRONTEND_CLIENT.save_explanation()

        else:
            raise WebInterfaceError(f"Unknown 'do' command {do} for explainer")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # TODO think about multiple instances of a client

    # TODO switch to 'run' in production
    #  In production mode the eventlet web server is used if available,
    #  else the gevent web server is used. If eventlet and gevent are not installed,
    #  the Werkzeug development web server is used.
    # app.run(debug=True, port=4568)

    # TODO Flask development web server is used, use eventlet or gevent,
    #  see https://flask-socketio.readthedocs.io/en/latest/deployment.html
    socketio.run(app, host='0.0.0.0', debug=True, port=4567,
                 allow_unsafe_werkzeug=True)
