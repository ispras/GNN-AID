class WorkerForceNeighborhoodLayout extends WorkerLayout {
    constructor(graph, params) {
        super(graph, params)

        this.alpha = params.alpha
        this.beta = params.beta
        this.visc = params.visc
        this.minV = params.minV
        this.temp = params.temp
        this.rad = params.rad
        this.stateFlag = params.stateFlag
        this.constT = params.constT
        this.thrA = params.thrA
        this.decayFactor = params.decayFactor

        this.depth = this.extra.depth
        this.n0 = this.extra.n0
        this.layers = this.extra.layers || {}
        this.edgesByDepth = this.extra.edgesByDepth || {}

        this.vx = new Float32Array(this.nodeCount)
        this.vy = new Float32Array(this.nodeCount)
        this.m = new Float32Array(this.nodeCount)

        for (let i = 0; i < this.nodeCount; i++) {
            this.m[i] = 0.1
        }

        this.m[this.n0] = 3 * this.nodeCount

        if (this.depth >= 1 && this.layers[1]) {
            for (const n of this.layers[1]) {
                this.m[n] = this._degreeFromEdgesByDepth(n)
            }
        }

        if (this.depth >= 2) {
            for (let d = 2; d <= this.depth; d++) {
                const layer = this.layers[d]
                if (!layer) continue
                for (const n of layer) {
                    this.m[n] = 1
                }
            }
        }

        this.grid = new Map()
        this.cellSize = 1
    }

    _degreeFromEdgesByDepth(node) {
        let deg = 0
        for (const arr of Object.values(this.edgesByDepth)) {
            for (let i = 0; i < arr.length; i += 2) {
                if (arr[i] === node || arr[i + 1] === node) {
                    deg++
                }
            }
        }
        return deg
    }

    cool() {
        this.temp *= this.decayFactor
    }

    force_r(x) {
        return this.rad ** 2 / x
    }

    force_a(x) {
        return x ** 3 / this.rad
    }

    _cellKey(cx, cy) {
        return `${cx},${cy}`
    }

    _buildGrid(interactionRadius) {
        this.grid.clear()
        this.cellSize = interactionRadius

        for (let i = 0; i < this.nodeCount; i++) {
            const x = this.pos[2 * i]
            const y = this.pos[2 * i + 1]
            const cx = Math.floor(x / interactionRadius)
            const cy = Math.floor(y / interactionRadius)
            const key = this._cellKey(cx, cy)

            let bucket = this.grid.get(key)
            if (bucket === undefined) {
                bucket = []
                this.grid.set(key, bucket)
            }
            bucket.push(i)
        }
    }

    step() {
        const n = this.nodeCount

        if (this.lockedNode != null) {
            this.stateFlag = 1
            this.temp = this.constT
        } else if (this.stateFlag === 1) {
            this.temp *= 10 / Math.sqrt(Math.sqrt(n))
            this.stateFlag = 0
        }

        if (this.stateFlag === 2) {
            this.stateFlag = 0
            this.temp *= Math.sqrt(n) / 100
            this.constT = this.temp / (5 * Math.sqrt(n))
            this.rad *= Math.sqrt(Math.sqrt(n))
        }

        const interactionRadius = 2 * this.rad
        const interactionRadiusSq = interactionRadius * interactionRadius
        this._buildGrid(interactionRadius)

        for (let i = 0; i < n; i++) {
            this.vx[i] = 0
            this.vy[i] = 0

            const pix = this.pos[2 * i]
            const piy = this.pos[2 * i + 1]

            const baseCx = Math.floor(pix / this.cellSize)
            const baseCy = Math.floor(piy / this.cellSize)

            for (let oy = -1; oy <= 1; oy++) {
                for (let ox = -1; ox <= 1; ox++) {
                    const bucket = this.grid.get(this._cellKey(baseCx + ox, baseCy + oy))
                    if (!bucket) continue

                    for (let k = 0; k < bucket.length; k++) {
                        const j = bucket[k]
                        if (i === j) continue

                        const dx = pix - this.pos[2 * j]
                        const dy = piy - this.pos[2 * j + 1]
                        const distSq = dx * dx + dy * dy

                        if (distSq === 0 || distSq >= interactionRadiusSq) continue

                        const dist = Math.sqrt(distSq)
                        const scale = this.force_r(dist) / dist
                        this.vx[i] += dx * scale
                        this.vy[i] += dy * scale
                    }
                }
            }
        }

        for (const arr of Object.values(this.edgesByDepth)) {
            for (let e = 0; e < arr.length; e += 2) {
                const i = arr[e]
                const j = arr[e + 1]

                const dx = this.pos[2 * i] - this.pos[2 * j]
                const dy = this.pos[2 * i + 1] - this.pos[2 * j + 1]
                const distSq = dx * dx + dy * dy
                if (distSq === 0) continue

                const dist = Math.sqrt(distSq)
                const scale = this.force_a(dist) / dist
                const fx = dx * scale
                const fy = dy * scale

                this.vx[i] -= fx
                this.vy[i] -= fy
                this.vx[j] += fx
                this.vy[j] += fy
            }
        }

        let moved = false

        for (let i = 0; i < n; i++) {
            if (i === this.lockedNode) continue

            const vx = this.vx[i]
            const vy = this.vy[i]
            const speed = Math.sqrt(vx * vx + vy * vy)
            if (speed === 0) continue

            const stepLen = speed > this.temp ? this.temp : speed
            const inv = 1 / speed
            const dx = vx * inv * stepLen
            const dy = vy * inv * stepLen

            if (dx !== 0 || dy !== 0) moved = true

            this.pos[2 * i] += dx
            this.pos[2 * i + 1] += dy
        }

        if (this.lockedNode == null) {
            this.cool()
        }

        this.iteration += 1
        return moved && this.temp > 0.005
    }
}