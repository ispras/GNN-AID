class Controller {
    constructor() {
        this.presenter = new Presenter()

        // Setup socket connection
        this.socket = io()
        this.socket.on('connect', () => console.log('socket connected'))
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
        await $.ajax({
            type: 'POST',
            url: url,
            data: data,
            success: (res, status, jqXHR) => {
                // console.log('Result: ' + res)
                // console.log('status: ' + status)
                // console.log('jqXHR: ' + jqXHR)
                result = res
            }
        })
        if (result && '{['.includes(result[0])) {
            return JSON_parse(result)
        }
        return result
    }
}

