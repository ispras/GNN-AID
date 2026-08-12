class Layout {
    constructor() {
        this.visibleGraph = null
        this.directed = null

        this.dt = 0.1
        this.moving = false
        this.iteration = 0
        this.pos = {} // node -> Vec
        this.lockedNode = null
        this.freeze = false

        // worker state
        this.worker = null
        this.workerReady = false
        this.useWorker = true
        this.workerUrl = '/static/js/presentation/dataset/layout/layout.worker.js'
        // this.workerUrl = '/web_interface/static/js/presentation/dataset/layout/layout.worker.js'

        // latest worker snapshot
        this.pendingPositions = null      // Float32Array or null
        this.hasPendingPositions = false
        this.lastWorkerIteration = 0
        this.lastWorkerStepTime = 0

        // cached graph data
        this.nodes = null
        this.nodeToIndex = null
        this.indexToNode = null
    }

    setVisibleGraph(visibleGraph) {
        this.visibleGraph = visibleGraph
        this.directed = visibleGraph.datasetInfo.directed

        this.nodes = this.visibleGraph.getNodes()
        this._buildNodeIndex()

        this.respawn()
        this.startMoving()
    }

    _buildNodeIndex() {
        this.nodeToIndex = new Map()
        this.indexToNode = new Array(this.nodes.length)

        for (let k = 0; k < this.nodes.length; k++) {
            const node = this.nodes[k]
            this.nodeToIndex.set(node, k)
            this.indexToNode[k] = node
        }
    }

    respawn() {
        this.iteration = 0
        this.pos = {}

        let r = this.nodes.length ** 0.5
        for (const n of this.nodes) {
            this.pos[n] = new Vec(r * Math.random(), r * Math.random())
        }

        if (this.useWorker) {
            this._ensureWorker()
            if (this.workerReady) {
                this._sendGraphToWorker()
            }
        }
    }

    startMoving() {
        if (this.freeze) return
        this.iteration = 0

        if (!this.moving) {
            this.moving = true
            this.run()
        }
    }

    stopMoving() {
        this.moving = false
        if (this.worker) {
            this.worker.postMessage({ type: 'stop' })
        }
    }

    setFreeze(freeze) {
        this.freeze = freeze
        if (this.freeze) {
            this.stopMoving()
        } else {
            this.startMoving()
        }
    }

    async run() {
        if (this.useWorker) {
            this._ensureWorker()
            if (this.workerReady) {
                this.worker.postMessage({ type: 'start' })
            }
            return
        }

        while ((this.moving || (this.lockedNode != null)) && !this.freeze) {
            let t = performance.now()
            this.step()
            Debug.LAYOUT_STEP_TIME = performance.now() - t
            await sleep(Math.max(0, 50 - Debug.LAYOUT_STEP_TIME))
        }
    }

    step() {
        this.moving = false
    }

    lock(node, pos) {
        this.lockedNode = node
        this.pos[this.lockedNode].set(pos)

        if (this.worker && this.workerReady) {
            const idx = this.nodeToIndex.get(node)
            this.worker.postMessage({
                type: 'lock',
                nodeIndex: idx,
                x: pos.x,
                y: pos.y
            })
        }
    }

    release() {
        this.lockedNode = null
        if (this.worker && this.workerReady) {
            this.worker.postMessage({ type: 'release' })
        }
    }

    destroy() {
        if (this.worker) {
            this.worker.terminate()
            this.worker = null
            this.workerReady = false
        }
    }

    _ensureWorker() {
        if (this.worker || !this.useWorker) return

        this.worker = new Worker(this.workerUrl)

        this.worker.onmessage = (e) => {
            const msg = e.data

            if (msg.type === 'ready') {
                this.workerReady = true

                if (this.visibleGraph) {
                    this._sendGraphToWorker()
                    if (this.moving && !this.freeze) {
                        this.worker.postMessage({ type: 'start' })
                    }
                }
                return
            }

            if (msg.type === 'tick') {
                this.pendingPositions = new Float32Array(msg.positions)
                this.hasPendingPositions = true
                this.lastWorkerIteration = msg.iteration
                this.lastWorkerStepTime = msg.stepTime
                this.moving = true
                return
            }

            if (msg.type === 'idle') {
                this.moving = false
                return
            }
        }

        this.worker.onerror = (err) => {
            console.error('Layout worker error:', err)
        }

        this.worker.onmessageerror = (err) => {
            console.error('Layout worker message error:', err)
        }
    }
    _sendGraphToWorker() {
        if (!this.worker || !this.workerReady || !this.visibleGraph) return

        const nodeCount = this.nodes.length
        const edges = this.visibleGraph.getEdges()

        const positions = new Float32Array(nodeCount * 2)
        for (let k = 0; k < nodeCount; k++) {
            const node = this.indexToNode[k]
            positions[2 * k] = this.pos[node].x
            positions[2 * k + 1] = this.pos[node].y
        }

        const edgeIndex = new Uint32Array(edges.length * 2)
        for (let e = 0; e < edges.length; e++) {
            const srcNode = edges[e][0]
            const dstNode = edges[e][1]
            edgeIndex[2 * e] = this.nodeToIndex.get(srcNode)
            edgeIndex[2 * e + 1] = this.nodeToIndex.get(dstNode)
        }

        const extra = this.serializeGraphData()

        this.worker.postMessage({
            type: 'init',
            layoutType: this.getLayoutType(),
            graph: {
                nodeCount,
                edgeIndex,
                positions,
                extra
            },
            params: this.serializeParams()
        })
    }

    // Вызывается рендерером / visibleGraph раз в 100мс
    // Возвращает true если координаты были обновлены
    consumePositionUpdates() {
        if (!this.hasPendingPositions || !this.pendingPositions) {
            return false
        }

        const arr = this.pendingPositions
        for (let k = 0; k < this.indexToNode.length; k++) {
            const node = this.indexToNode[k]
            this.pos[node].x = arr[2 * k]
            this.pos[node].y = arr[2 * k + 1]
        }

        this.iteration = this.lastWorkerIteration
        Debug.LAYOUT_STEP_TIME = this.lastWorkerStepTime

        this.pendingPositions = null
        this.hasPendingPositions = false
        return true
    }

    getLayoutType() {
        return 'base'
    }

    serializeParams() {
        return {}
    }

    serializeGraphData() {
        return {}
    }
}