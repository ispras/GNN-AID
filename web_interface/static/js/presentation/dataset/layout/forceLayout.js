// class ForceLayout extends Layout {
//     constructor(alpha = 30, beta = 20, minV = 0.01, visc = 0.85, temp = 15, rad = 1.5) {
//         super()
//
//         this.alpha = alpha
//         this.beta = beta
//         this.visc = visc // viscosity
//         this.minV = minV
//         this.temp = temp // temperature used in algorithm
//         this.rad = rad // radius for calculating forces
//         this.stateFlag = 2 // action mode: 0 - free action, 1 - node grabbed, 2 - respawn
//         this.constT = temp / 100 // constant temperature when node is grabbed
//
//         this.numNodes = null // number of nodes for Graph
//         this.edges = null // list of edges
//
//         // typed arrays instead of object-of-Vec
//         this.vx = null
//         this.vy = null
//         this.m = null
//
//         this.thrA = 150 // max total acceleration on a node
//         this.decayFactor = 0.98 // power of forces decay over iterations
//
//         // grid params/cache
//         this.grid = new Map()
//         this.cellSize = 1
//     }
//
//     setVisibleGraph(visibleGraph) {
//         console.assert(visibleGraph instanceof Graph)
//         this.edges = visibleGraph.getEdges()
//         this.numNodes = visibleGraph.getNumNodes()
//         super.setVisibleGraph(visibleGraph)
//     }
//
//     respawn() {
//         super.respawn()
//
//         const n = this.numNodes
//         this.vx = new Float32Array(n)
//         this.vy = new Float32Array(n)
//         this.m = new Float32Array(n)
//
//         for (let i = 0; i < n; i++) {
//             this.m[i] = 0.1
//         }
//
//         for (const [i, j] of this.visibleGraph.getEdges()) {
//             this.m[i] += 1
//             this.m[j] += 1
//         }
//     }
//
//     cool() {
//         this.temp = this.temp * this.decayFactor
//     }
//
//     force_r(x) {
//         return this.rad ** 2 / x
//     }
//
//     force_a(x) {
//         return x ** 3 / this.rad
//     }
//
//     _cellKey(cx, cy) {
//         return `${cx},${cy}`
//     }
//
//     _buildGrid(nodes, interactionRadius) {
//         this.grid.clear()
//         this.cellSize = interactionRadius
//
//         const cellSize = this.cellSize
//
//         for (const i of nodes) {
//             const p = this.pos[i]
//             const cx = Math.floor(p.x / cellSize)
//             const cy = Math.floor(p.y / cellSize)
//             const key = this._cellKey(cx, cy)
//
//             let bucket = this.grid.get(key)
//             if (bucket === undefined) {
//                 bucket = []
//                 this.grid.set(key, bucket)
//             }
//             bucket.push(i)
//         }
//     }
//
//     step() {
//         let add = 0
//         const nodes = this.visibleGraph.getNodes()
//         const edges = this.edges ?? this.visibleGraph.getEdges()
//
//         // action mode check
//         if (this.stateFlag === 2) {
//             this.stateFlag = 0
//             this.temp *= nodes.length / 100
//             this.constT = this.temp / (10 * Math.sqrt(nodes.length))
//             this.rad *= Math.sqrt(Math.sqrt(nodes.length))
//         }
//
//         if (this.lockedNode != null) {
//             this.stateFlag = 1
//             this.temp = this.constT
//         } else {
//             if (this.stateFlag === 1) {
//                 this.stateFlag = 0
//                 this.temp *= 10 / Math.sqrt(Math.sqrt(nodes.length))
//             }
//         }
//
//         const interactionRadius = 2 * this.rad
//         const interactionRadiusSq = interactionRadius * interactionRadius
//
//         // 1) build uniform grid
//         this._buildGrid(nodes, interactionRadius)
//
//         // 2) repulsive forces with grid neighbor search
//         for (const i of nodes) {
//             this.vx[i] = 0
//             this.vy[i] = 0
//
//             const pix = this.pos[i].x
//             const piy = this.pos[i].y
//
//             const baseCx = Math.floor(pix / this.cellSize)
//             const baseCy = Math.floor(piy / this.cellSize)
//
//             for (let oy = -1; oy <= 1; oy++) {
//                 for (let ox = -1; ox <= 1; ox++) {
//                     const key = this._cellKey(baseCx + ox, baseCy + oy)
//                     const bucket = this.grid.get(key)
//                     if (bucket === undefined) {
//                         continue
//                     }
//
//                     for (let k = 0; k < bucket.length; k++) {
//                         const j = bucket[k]
//                         if (i === j) {
//                             continue
//                         }
//
//                         const dx = pix - this.pos[j].x
//                         const dy = piy - this.pos[j].y
//                         const distSq = dx * dx + dy * dy
//
//                         if (distSq >= interactionRadiusSq || distSq === 0) {
//                             continue
//                         }
//
//                         const dist = Math.sqrt(distSq)
//                         const scale = this.force_r(dist) / dist
//
//                         this.vx[i] += dx * scale
//                         this.vy[i] += dy * scale
//                     }
//                 }
//             }
//         }
//
//         // 3) attractive forces over edges
//         for (let e = 0; e < edges.length; e++) {
//             const i = edges[e][0]
//             const j = edges[e][1]
//
//             const dx = this.pos[i].x - this.pos[j].x
//             const dy = this.pos[i].y - this.pos[j].y
//             const distSq = dx * dx + dy * dy
//
//             if (distSq === 0) {
//                 continue
//             }
//
//             const dist = Math.sqrt(distSq)
//             const scale = this.force_a(dist) / dist
//
//             const fx = dx * scale
//             const fy = dy * scale
//
//             this.vx[i] -= fx
//             this.vy[i] += 0 - fy
//
//             this.vx[j] += fx
//             this.vy[j] += fy
//         }
//
//         // recompute positions
//         this.moving = false
//
//         for (const i of nodes) {
//             if (i === this.lockedNode) {
//                 continue
//             }
//
//             const vx = this.vx[i]
//             const vy = this.vy[i]
//             const speed = Math.sqrt(vx * vx + vy * vy)
//
//             let dx = 0
//             let dy = 0
//
//             if (speed !== 0) {
//                 const stepLen = (speed > this.temp) ? this.temp : speed
//                 const inv = 1 / speed
//                 dx = vx * inv * stepLen
//                 dy = vy * inv * stepLen
//             }
//
//             this.pos[i].x += dx
//             this.pos[i].y += dy
//         }
//
//         // check if structure is still moving
//         if (this.temp > 0.005) {
//             this.moving = true
//         }
//
//         if (!this.lockedNode) {
//             this.cool()
//         }
//
//         this.iteration += 1
//     }
// }

class ForceLayout extends Layout {
    constructor(alpha = 30, beta = 20, minV = 0.01, visc = 0.85, temp = 15, rad = 1.5) {
        super()

        this.alpha = alpha
        this.beta = beta
        this.visc = visc
        this.minV = minV
        this.temp = temp
        this.rad = rad
        this.stateFlag = 2
        this.constT = temp / 100

        this.numNodes = null
        this.edges = null

        this.vx = null
        this.vy = null
        this.m = null

        this.thrA = 150
        this.decayFactor = 0.98

        this.grid = new Map()
        this.cellSize = 1
    }

    setVisibleGraph(visibleGraph) {
        console.assert(visibleGraph instanceof Graph)
        this.edges = visibleGraph.getEdges()
        this.numNodes = visibleGraph.getNumNodes()
        super.setVisibleGraph(visibleGraph)
    }

    getLayoutType() {
        return 'force'
    }

    serializeParams() {
        return {
            alpha: this.alpha,
            beta: this.beta,
            visc: this.visc,
            minV: this.minV,
            temp: this.temp,
            rad: this.rad,
            stateFlag: this.stateFlag,
            constT: this.constT,
            thrA: this.thrA,
            decayFactor: this.decayFactor
        }
    }
}