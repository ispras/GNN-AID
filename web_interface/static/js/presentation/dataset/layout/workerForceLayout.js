// class WorkerForceLayout {
//     constructor(graph, params) {
//         this.graph = graph
//         this.numNodes = graph.numNodes
//         this.edgeIndex = graph.edgeIndex
//         this.pos = graph.positions
//
//         this.alpha = params.alpha
//         this.beta = params.beta
//         this.minV = params.minV
//         this.visc = params.visc
//         this.temp = params.temp
//         this.rad = params.rad
//         this.stateFlag = params.stateFlag
//         this.constT = params.constT
//         this.thrA = params.thrA
//         this.decayFactor = params.decayFactor
//
//         this.lockedNode = null
//
//         this.vx = new Float32Array(this.numNodes)
//         this.vy = new Float32Array(this.numNodes)
//         this.mass = new Float32Array(this.numNodes)
//
//         for (let i = 0; i < this.numNodes; i++) {
//             this.mass[i] = 0.1
//         }
//         for (let e = 0; e < this.edgeIndex.length; e += 2) {
//             const a = this.edgeIndex[e]
//             const b = this.edgeIndex[e + 1]
//             this.mass[a] += 1
//             this.mass[b] += 1
//         }
//
//         this.grid = new Map()
//         this.cellSize = 1
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
//     cool() {
//         this.temp = this.temp * this.decayFactor
//     }
//
//     _cellKey(cx, cy) {
//         return `${cx},${cy}`
//     }
//
//     _buildGrid(interactionRadius) {
//         this.grid.clear()
//         this.cellSize = interactionRadius
//
//         for (let i = 0; i < this.numNodes; i++) {
//             const x = this.pos[2 * i]
//             const y = this.pos[2 * i + 1]
//             const cx = Math.floor(x / interactionRadius)
//             const cy = Math.floor(y / interactionRadius)
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
//         const n = this.numNodes
//
//         if (this.stateFlag === 2) {
//             this.stateFlag = 0
//             this.temp *= n / 100
//             this.constT = this.temp / (10 * Math.sqrt(n))
//             this.rad *= Math.sqrt(Math.sqrt(n))
//         }
//
//         if (this.lockedNode != null) {
//             this.stateFlag = 1
//             this.temp = this.constT
//         } else {
//             if (this.stateFlag === 1) {
//                 this.stateFlag = 0
//                 this.temp *= 10 / Math.sqrt(Math.sqrt(n))
//             }
//         }
//
//         const interactionRadius = 2 * this.rad
//         const interactionRadiusSq = interactionRadius * interactionRadius
//
//         this._buildGrid(interactionRadius)
//
//         for (let i = 0; i < n; i++) {
//             this.vx[i] = 0
//             this.vy[i] = 0
//
//             const pix = this.pos[2 * i]
//             const piy = this.pos[2 * i + 1]
//
//             const baseCx = Math.floor(pix / this.cellSize)
//             const baseCy = Math.floor(piy / this.cellSize)
//
//             for (let oy = -1; oy <= 1; oy++) {
//                 for (let ox = -1; ox <= 1; ox++) {
//                     const bucket = this.grid.get(this._cellKey(baseCx + ox, baseCy + oy))
//                     if (!bucket) continue
//
//                     for (let k = 0; k < bucket.length; k++) {
//                         const j = bucket[k]
//                         if (i === j) continue
//
//                         const dx = pix - this.pos[2 * j]
//                         const dy = piy - this.pos[2 * j + 1]
//                         const distSq = dx * dx + dy * dy
//
//                         if (distSq === 0 || distSq >= interactionRadiusSq) continue
//
//                         const dist = Math.sqrt(distSq)
//                         const scale = this.force_r(dist) / dist
//                         this.vx[i] += dx * scale
//                         this.vy[i] += dy * scale
//                     }
//                 }
//             }
//         }
//
//         for (let e = 0; e < this.edgeIndex.length; e += 2) {
//             const i = this.edgeIndex[e]
//             const j = this.edgeIndex[e + 1]
//
//             const dx = this.pos[2 * i] - this.pos[2 * j]
//             const dy = this.pos[2 * i + 1] - this.pos[2 * j + 1]
//             const distSq = dx * dx + dy * dy
//             if (distSq === 0) continue
//
//             const dist = Math.sqrt(distSq)
//             const scale = this.force_a(dist) / dist
//             const fx = dx * scale
//             const fy = dy * scale
//
//             this.vx[i] -= fx
//             this.vy[i] -= fy
//             this.vx[j] += fx
//             this.vy[j] += fy
//         }
//
//         for (let i = 0; i < n; i++) {
//             if (i === this.lockedNode) continue
//
//             const vx = this.vx[i]
//             const vy = this.vy[i]
//             const speed = Math.sqrt(vx * vx + vy * vy)
//             if (speed === 0) continue
//
//             const stepLen = speed > this.temp ? this.temp : speed
//             const inv = 1 / speed
//
//             this.pos[2 * i] += vx * inv * stepLen
//             this.pos[2 * i + 1] += vy * inv * stepLen
//         }
//
//         if (this.lockedNode == null) {
//             this.cool()
//         }
//     }
// }