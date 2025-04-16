class Controller {
    constructor(sessionId, mode) {
        this.sessionId = sessionId // ID of session
        this.isActive = false // this controller was started
        this.mode = mode
        this.presenter = new Presenter()

        // Setup socket connection
        this.socket = io({
            reconnection: true,
            reconnectionAttempts: Infinity,
            reconnectionDelay: 1000,
        })
        this.socket.on('connect', () => {
            console.log('socket connected')
            if (this.isActive) {
                // FIXME seems sometimes it happens when backend is busy - we don't need to reload?
                // Means re-connection to server. Need to reload the page
                alert("This session is outdated. Press OK to reload the page.")
                this.isActive = false
                window.location.reload(true)
            }
            else {
                this.isActive = true
                console.log('session_id', this.sessionId)
                this.run()
            }
        })
        this.socket.on('reconnect', () => {
            console.log('socket re-connected')
            console.log('this.isActive=', this.isActive)
        })
        this.socket.on('reconnect_attempt', (attemptNumber) => {
            console.log('socket reconnect_attempt', attemptNumber)
        })
        this.socket.on('connect_error', (err) => {
            console.log('socket connect_error', err.message, 'sid=', this.socket.id)
        })
        this.socket.on('reconnect_error', (err) => {
            console.log('socket reconnect_error', err)
        })
        this.socket.on('reconnect_failed', () => {
            console.log('socket reconnect_failed')
        })
        this.socket.on('error', (err) => {
            console.log('socket error', err)
        })

        this.socket.on('disconnect', () => {
            // this.isActive = false
            console.log('Disconnected from server, sid=', this.socket.id);
            if (this.isActive) {
                // Show a notification to the user
                alert("This session is outdated. Press OK to reload the page.")
                this.isActive = false
                window.location.reload(true)
            }
        })

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

    async blockRequest(blockName, funcName, requestParams) {
        let data = {
            block: blockName,
            func: funcName,
            params: JSON_stringify(requestParams),
        }
        return await this.ajaxRequest('/block', data)
    }

    // // Setup storage contents
    // async getStorageContents(type) {
    //     let url = '/ask'
    //     let data = {
    //         ask: "storage",
    //         type: type,
    //     }
    //     let [ps, info] = await this.ajaxRequest(url, data)
    //     ps = PrefixStorage.fromJSON(ps)
    //     info = JSON_parse(info)
    //     return [ps, info]
    // }

    async ajaxRequest(url, data) {
        let result = null
        console.assert(!('sessionId' in data))
        data['sessionId'] = this.sessionId
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

