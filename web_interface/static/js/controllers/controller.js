class Controller {
    constructor(sessionId, mode) {
        this.sessionId = sessionId // ID of session
        this.sid = null
        this.mode = mode
        this.isActive = false // this controller was started
        this.presenter = new Presenter()
        this.backendRestartTimer = null
        this.backendRestartInterval = null
        this.backendRestartOverlay = null
        this.isBackendRestarting = false

        // Setup socket connection
        this.socket = io({
            reconnection: true,
            reconnectionAttempts: Infinity,
            reconnectionDelay: 1000,
            query: {mode: mode},

            // Longer timeout for backend debug - 10 mins
            timeout: 600*1000,
            pingTimeout: 600*1000,
            pingInterval: 600*1000,
        })
        this.socket.on('connect', () => {
            console.log('socket connected, sid=', this.socket.id)
            this.sid = this.socket.id
            if (this.isActive) {
                console.log('This is a reconnection')
                // FIXME seems sometimes it happens when backend is busy - we don't need to reload?
                // // Means re-connection to server. Need to reload the page
                alert("Web-socket connection is lost. Press OK to reload the page.")
                this.isActive = false
                window.location.reload(true)
            }
            else {
                this.isActive = true
                console.log('sid', this.sid, 'sessionId', this.sessionId)
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

        this.socket.on('disconnect', (reason) => {
            // this.isActive = false
            console.log('Disconnected from server, reason=', reason, ', sid=', this.socket.id);
            if (this.isActive) {
                const softDisconnectReasons = ['ping timeout', 'transport error'];
                if (softDisconnectReasons.includes(reason)) {
                    console.log('Soft disconnect, will try to reconnect...');
                    // this.reconnectStartTime = Date.now()
                } else {
                    // Show a notification to the user
                    alert("Web-socket connection is lost. Press OK to reload the page.")
                    this.isActive = false
                    window.location.reload(true)
                }
            }
        })

        this.socket.on('message', async (data) => {
            // Service/backend lifecycle messages
            if (await this.handleServiceMessage(data)) {
                return
            }

            // During backend restart ignore ordinary messages just in case
            if (this.isBackendRestarting) {
                console.log('Ignoring normal message while backend is restarting', data)
                return
            }

            // Standard block message
            if (!data || typeof data !== 'object') {
                console.log('received unexpected socket payload', data)
                return
            }

            // if (!('msg' in data) || !('block' in data) || !('func' in data)) {
            //     console.log('received non-block socket payload', data)
            //     return
            // }

            let msg = JSON_parse(data["msg"])
            let block = data["block"]
            let func = data["func"]

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

    showBackendOverlay() {
        let overlay = document.getElementById('backend-service-overlay')

        if (!overlay) {
            overlay = document.createElement('div')
            overlay.id = 'backend-service-overlay'
            overlay.style.position = 'fixed'
            overlay.style.top = '20px'
            overlay.style.left = '100px'
            overlay.style.zIndex = '99999'
            overlay.style.maxWidth = '560px'
            overlay.style.padding = '16px 20px'
            overlay.style.background = '#fff4e5'
            overlay.style.border = '1px solid #ffb74d'
            overlay.style.borderRadius = '8px'
            overlay.style.boxShadow = '0 4px 16px rgba(0,0,0,0.15)'
            overlay.style.color = '#333'
            overlay.style.fontFamily = 'sans-serif'
            overlay.style.fontSize = '14px'
            overlay.style.lineHeight = '1.45'
            document.body.appendChild(overlay)
        }

        overlay.style.display = 'block'
        this.backendRestartOverlay = overlay
        return overlay
    }

    stopBackendRestartCountdown() {
        if (this.backendRestartTimer) {
            clearTimeout(this.backendRestartTimer)
            this.backendRestartTimer = null
        }
        if (this.backendRestartInterval) {
            clearInterval(this.backendRestartInterval)
            this.backendRestartInterval = null
        }
    }

    startBackendRestartCountdown(title, text, traceback, restartInSec) {
        this.stopBackendRestartCountdown()

        const overlay = this.showBackendOverlay()
        let remaining = Number.isFinite(restartInSec) ? restartInSec : 30

        // Создаем DOM только один раз
        overlay.innerHTML = ''

        const titleEl = document.createElement('div')
        titleEl.style.fontWeight = 'bold'
        titleEl.style.marginBottom = '8px'
        titleEl.textContent = title

        const textEl = document.createElement('div')
        textEl.style.marginBottom = '10px'
        textEl.style.whiteSpace = 'pre-wrap'
        textEl.textContent = text

        const timerEl = document.createElement('div')
        timerEl.innerHTML = `Restarting in <b>${remaining}</b>s.`

        overlay.appendChild(titleEl)
        overlay.appendChild(textEl)
        overlay.appendChild(timerEl)

        if (traceback) {
            const detailsEl = document.createElement('details')
            detailsEl.style.marginTop = '10px'

            const summaryEl = document.createElement('summary')
            summaryEl.style.cursor = 'pointer'
            summaryEl.textContent = 'Stack trace'

            const preEl = document.createElement('pre')
            preEl.style.marginTop = '8px'
            preEl.style.maxHeight = '300px'
            preEl.style.overflow = 'auto'
            preEl.style.background = '#f7f7f7'
            preEl.style.padding = '10px'
            preEl.style.border = '1px solid #ddd'
            preEl.style.whiteSpace = 'pre-wrap'
            preEl.textContent = traceback

            detailsEl.appendChild(summaryEl)
            detailsEl.appendChild(preEl)
            overlay.appendChild(detailsEl)
        }

        const renderTimer = () => {
            timerEl.innerHTML = `Restarting in <b>${remaining}</b>s.`
        }

        renderTimer()

        this.backendRestartInterval = setInterval(() => {
            remaining = Math.max(0, remaining - 1)
            renderTimer()

            if (remaining <= 0) {
                this.stopBackendRestartCountdown()
            }
        }, 1000)
    }

    async handleServiceMessage(data) {
        if (!data || typeof data !== 'object' || !data.type) {
            return false
        }

        if (data.type === 'server_error') {
            console.error('Backend service error:', data)
            if (data.traceback) {
                console.error(data.traceback)
            }

            this.isBackendRestarting = true

            const restartInSec = data.restart_in_sec ?? 30
            const title = data.title || 'Backend error'
            const text = data.text || 'Unknown backend error'
            const traceback = data.traceback || ''

            this.startBackendRestartCountdown(title, text, traceback, restartInSec)
            return true
        }

        if (data.type === 'server_info') {
            console.log('Backend service info:', data)

            this.stopBackendRestartCountdown()

            const overlay = this.showBackendOverlay()
            overlay.innerHTML = `
                <div style="font-weight: bold; margin-bottom: 8px;">Backend info</div>
                <div>${data.text || 'Backend restarted successfully'}</div>
                <div style="margin-top: 8px;">Reload page...</div>
            `

            window.location.reload()
            return true
        }

        // if (data.type === 'fatal_stop') {
        //     console.error('Fatal stop:', data)
        //     if (data.traceback) {
        //         console.error(data.traceback)
        //     }
        //
        //     const overlay = this.showBackendOverlay()
        //     overlay.innerHTML = `
        //         <div style="font-weight: bold; margin-bottom: 8px;">${data.title || 'Server shutdown'}</div>
        //         <div style="white-space: pre-wrap; margin-bottom: 10px;">${data.text || 'Server stopped'}</div>
        //         ${data.traceback ? `
        //             <details style="margin-top: 10px;">
        //                 <summary style="cursor: pointer;">Stack trace</summary>
        //                 <pre style="
        //                     margin-top: 8px;
        //                     max-height: 300px;
        //                     overflow: auto;
        //                     background: #f7f7f7;
        //                     padding: 10px;
        //                     border: 1px solid #ddd;
        //                     white-space: pre-wrap;
        //                 ">${escapeHtml(data.traceback)}</pre>
        //             </details>
        //         ` : ''}
        //         <div style="margin-top: 10px;">Server is down. Please reload later.</div>
        //     `
        //
        //     this.isBackendRestarting = true
        //     return true
        // }

        return false
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
        console.assert(!('sid' in data))
        data['sid'] = this.sid
        data['sessionId'] = this.sessionId
        // console.log('ajax request', data)
        await $.ajax({
            type: 'POST',
            url: url,
            data: data,
            // contentType: 'application/json',
            // contentType: 'application/x-www-form-urlencoded; charset=UTF-8',
            success: (res, status, jqXHR) => {
                result = res
                // console.log('got ajax result', result)
            },
            error: (xhr, status, error) => {
                if (xhr.status === 503) {
                    console.warn('Backend temporarily unavailable:', xhr.responseText)
                    return
                }
                console.error('AJAX error:', status, error)
                console.error('Response:', xhr.responseText)
            }
        })
        if (result && '{['.includes(result[0])) {
            return JSON_parse(result)
        }
        return result
    }
}

