class Controller {
    static sessionId // ID of session
    static isActive = false // this controller was started

    constructor() {
        this.presenter = new Presenter()

        // Setup socket connection
        this.socket = io()
        this.socket.on('connect', () => {
            console.log('socket connected')
            if (Controller.isActive) {
                // Means re-connection to server. Need to reload the page
                alert("This session is outdated. Press OK to reload the page.")
                Controller.isActive = false
                window.location.reload(true)
            }
        })
        this.socket.on('session_id', (data) => {
            Controller.sessionId = data["session_id"]
            Controller.isActive = true
            console.log('session_id', Controller.sessionId)
            this.run()
        })
        this.socket.on('disconnect', () => {
            // Controller.isActive = false
            console.log('Disconnected from server');
        });

        this.socket.on('message', async (data) => {
            // Message to block listeners
            let msg = JSON_parse(data["msg"])
            let block = data["block"]
            let func = data["func"]
            // if (msg)
            //     console.log('received msg from', block, 'of len =', data["msg"].length)
            if (block in this.presenter.blockListeners) {
                for (const listener of this.presenter.blockListeners[block]) {
                    switch (func) {
                        case "onInit":
                            await listener.onInit(block, msg)
                            break
                        case "onModify":
                            listener.onModify(block, msg)
                            break
                        case "onUnlock":
                            listener.onUnlock(block, msg)
                            break
                        case "onBreak":
                            listener.onBreak(block, msg)
                            break
                        case "onSubmit":
                            await listener.onSubmit(block, msg)
                            break
                        default:
                            await listener.onReceive(block, msg)
                    }
                }
            }
            else {
                console.log('received', data)
            }
        })
    }

    async run() {
        this.presenter.createViews()

        // Start with dataset config
        this.presenter.menuDatasetView.init()
    }

    static async blockRequest(blockName, funcName, requestParams) {
        let data = {
            block: blockName,
            func: funcName,
            params: JSON_stringify(requestParams),
        }
        return await Controller.ajaxRequest('/block', data)
    }

    // Setup storage contents
    static async getStorageContents(type) {
        let url = '/ask'
        let data = {
            ask: "storage",
            type: type,
        }
        let [ps, info] = await Controller.ajaxRequest(url, data)
        ps = PrefixStorage.fromJSON(ps)
        info = JSON_parse(info)
        return [ps, info]
    }

    static async ajaxRequest(url, data) {
        let result = null
        console.assert(!('sessionId' in data))
        data['sessionId'] = Controller.sessionId
        // console.log('ajaxRequest', data)
        await $.ajax({
            type: 'POST',
            url: url,
            data: data,
            success: (res, status, jqXHR) => {
                result = res
            }
        })
        if (result && '{['.includes(result[0])) {
            return JSON_parse(result)
        }
        return result
    }
}

