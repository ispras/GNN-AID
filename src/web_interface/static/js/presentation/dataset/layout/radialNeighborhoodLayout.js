class RadialNeighborhoodLayout extends Layout {
    constructor(alpha = 30, beta = 20,
                minV = 0.01, visc=0.85) {
        super()

        this.alpha = alpha
        this.beta = beta
        this.visc = visc // viscosity
        this.minV = minV

        this.layoutRadius = {1: 1, 2: 2}
        this.r = {} // radial distance
        this.fi = {} // angle, 0 - 2*PI
        this.v = {} // velocity
        this.a = {} // acceleration

        this.thrA = 50 // max total acceleration on a node
        this.iteration = 0 // number of iterations
    }

    respawn() {
        super.respawn()
        // Respawn radial coordinates
        this.r = {}
        this.fi = {}
        this.v = {}
        this.a = {}
        for (const n of this.visibleGraph.n1) {
            this.r[n] = this.layoutRadius[1]
            this.fi[n] = 2*Math.random() * Math.PI
            this.v[n] = 0
            this.a[n] = 0
        }
        for (const n of this.visibleGraph.n2) {
            this.r[n] = this.layoutRadius[2]
            this.fi[n] = 2*Math.random() * Math.PI
            this.v[n] = 0
            this.a[n] = 0
        }
    }

    _dist(i, j) {
        let fi = Math.abs(this.fi[i] - this.fi[j]) % (2*Math.PI)
        let d = Math.min(fi, 2*Math.PI - fi)
        return d
    }
    _repulseNodes(i, j) {
        const d0 = 0.4
        const k = 0.1
        let d = this._dist(i, j)
        let m = 1/(d0 + k * d**2) - 1/(d0 + k * Math.PI**2)
        let sign = (2*Math.PI + this.fi[j] - this.fi[i]) % (2*Math.PI) > Math.PI ? 1 : -1
        let a = this.alpha * m * sign
        return a
    }

    _attractEdge(i, j) {
        const d0 = 0.1
        // const k = 0.1
        let d = this._dist(i, j)
        // let m = 1/(d0 + k * d**2) - 1/(d0 + k * Math.PI**2)
        let m = (d - d0) ** 2
        let sign = (2*Math.PI + this.fi[j] - this.fi[i]) % (2*Math.PI) > Math.PI ? 1 : -1
        let a = -this.beta * m * sign
        return a
    }

    step() {
        // console.log("step Radial")
        let dt = this.dt
        // Update forces
        let nodes = this.visibleGraph.getNodes(true)
        for (const n of nodes)
            this.a[n] = 0

        for (const i of nodes) { // Nodes except 0
            for (const j of nodes) {
                if (i >= j) continue
                let a = this._repulseNodes(i, j)
                this.a[i] += a * dt
                this.a[j] -= a * dt
            }
        }

        for (let [i, js] of Object.entries(this.visibleGraph.e11)) { // Edges 1-1
            i = parseInt(i)
            for (const j of js) {
                let a = this._attractEdge(i, j)
                this.a[i] += a * dt
                this.a[j] -= a * dt
            }
        }
        for (let [i, js] of Object.entries(this.visibleGraph.e12)) { // Edges 1-2
            i = parseInt(i)
            for (const j of js) {
                if (i >= j) continue
                let a = this._attractEdge(i, j)
                this.a[i] += a * dt
                this.a[j] -= a * dt
            }
        }

        // Normalize forces to max
        let maxA = Math.max.apply(null, Object.values(this.a).map((x) => Math.abs(x)))
        if (maxA > this.thrA * (0.9)**this.iteration) {
            for (const n of Object.keys(this.a)) {
                this.a[n] *= this.thrA * (0.9)**this.iteration / maxA
            }
        }

        this.moving = false
        // Update speed and angle
        for (const n of nodes) {
            if (n === this.lockedNode) continue
            this.v[n] += this.a[n] * dt
            this.fi[n] += this.v[n] * dt
            this.v[n] = this.v[n] > 0 ? ((1 + this.v[n]) ** this.visc) - 1 : 1-((1-this.v[n]) ** this.visc)
            // this.v[n] *= this.visc
            if (Math.abs(this.v[n]) > this.minV)
                this.moving = true
        }
        this.iteration += 1

        // TODO this is needed only by request when draw
        this.computePositions()
    }

    // Translate polar coordinates to cartesian
    computePositions() {
        this.pos[this.visibleGraph.n0].x = 0
        this.pos[this.visibleGraph.n0].y = 0
        for (const n of this.visibleGraph.n1) {
            this.pos[n].x = this.layoutRadius[1] * Math.cos(this.fi[n])
            this.pos[n].y = this.layoutRadius[1] * Math.sin(this.fi[n])
        }
        for (const n of this.visibleGraph.n2) {
            this.pos[n].x = this.layoutRadius[2] * Math.cos(this.fi[n])
            this.pos[n].y = this.layoutRadius[2] * Math.sin(this.fi[n])
        }
    }

    // Fix node position
    lock(node, pos) {
        this.lockedNode = node
        let fi = Math.acos(pos.x / pos.abs())
        if (pos.y < 0) fi = 2*Math.PI - fi
        this.fi[this.lockedNode] = fi
    }

}