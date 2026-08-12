importScripts(
    '/static/js/presentation/dataset/layout/worker/worker_layout.js',
    '/static/js/presentation/dataset/layout/worker/base_layout.js',
    '/static/js/presentation/dataset/layout/worker/force_layout.js',
    '/static/js/presentation/dataset/layout/worker/force_neighborhood_layout.js',
    '/static/js/presentation/dataset/layout/worker/create_layout.js'
)

let running = false
let layout = null
let graph = null
let iteration = 0

self.postMessage({ type: 'ready' })

self.onmessage = (e) => {
    const msg = e.data

    if (msg.type === 'init') {
        graph = {
            nodeCount: msg.graph.nodeCount,
            edgeIndex: msg.graph.edgeIndex ? new Uint32Array(msg.graph.edgeIndex) : null,
            positions: new Float32Array(msg.graph.positions),
            extra: msg.graph.extra || {}
        }

        layout = createLayout(msg.layoutType, graph, msg.params)
        iteration = 0
        return
    }

    if (msg.type === 'start') {
        if (!running) {
            running = true
            loop()
        }
        return
    }

    if (msg.type === 'stop') {
        running = false
        return
    }

    if (msg.type === 'lock') {
        layout.lockedNode = msg.nodeIndex
        graph.positions[2 * msg.nodeIndex] = msg.x
        graph.positions[2 * msg.nodeIndex + 1] = msg.y
        return
    }

    if (msg.type === 'release') {
        layout.lockedNode = null
        return
    }
}

function loop() {
    if (!running || !layout || !graph) return

    const t0 = performance.now()

    const stepsPerTick = 2
    let changed = false

    for (let i = 0; i < stepsPerTick; i++) {
        const moved = layout.step()
        iteration += 1
        changed = changed || moved
    }

    const dt = performance.now() - t0

    if (changed) {
        const out = new Float32Array(graph.positions)
        self.postMessage({
            type: 'tick',
            positions: out,
            iteration,
            stepTime: dt
        }, [out.buffer])
    } else {
        self.postMessage({
            type: 'idle',
            iteration,
            stepTime: dt
        })
    }

    setTimeout(loop, 16)
}