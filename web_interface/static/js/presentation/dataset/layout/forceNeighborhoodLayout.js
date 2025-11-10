class ForceNeighborhoodLayout extends Layout {
    constructor(alpha=30, beta=20, minV=0.01, visc=0.85, temp=15, rad=1.5) {
        super()

        this.alpha = alpha
        this.beta = beta
        this.visc = visc // viscosity
        this.minV = minV
        this.temp = temp // temperature used in algorithm
        this.rad = rad // radius for calculating forces
        this.stateFlag = 2 // action mode: 0 - free action, 1 - node grabbed, 2 - respawn
        this.constT = temp / 100 // constant temperature when node is grabbed

        this.v = {} // velocity, node -> Vec
        this.a = {} // acceleration, node -> Vec
        this.m = {} // node -> mass

        this.thrA = 150 // max total acceleration on a node
        this.decayFactor = 0.98 // power of forces decay over iterations
    }

    respawn() {
        // Respawn v and a
        super.respawn()
        this.v = {}
        this.a = {}
        this.m = {}
        let nodes = this.visibleGraph.getNodes()
        for (const n of nodes) {
            this.v[n] = new Vec(0, 0)
            this.a[n] = new Vec(0, 0)
            this.m[n] = 0.1
        }
        // FIXME generalize
        let depth = this.visibleGraph.depth
        this.m[this.visibleGraph.n0] = 3*nodes.size
        if (depth >= 1)
            for (const n of this.visibleGraph.nodes[1])
                this.m[n] = this.visibleGraph.getDegree(n)
        if (depth >= 2)
            for (let d=2; d <= depth; d++)
                for (const n of this.visibleGraph.nodes[d])
                    this.m[n] = 1
    }
/*
    _repulseNodes(i, j) {
        const d0 = 0.4
        const k = 0.1
        let d = this.pos[i].dist(this.pos[j])
        let m = 1/(d0 + k * d**2)
        let f = Vec.sub(this.pos[j], this.pos[i])
        f.mul(-this.alpha / d * m)
        f.mul((this.m[i] * this.m[j])**0.5)
        // if (isNaN(f.x))
        //     console.error(f)
        return f
    }

    _attractEdge(i, j, d0=0.1) {
        let d = this.pos[i].dist(this.pos[j])
        let m = (d - d0) ** 2
        let f = Vec.sub(this.pos[j], this.pos[i])
        f.mul(this.beta / d * m)
        f.mul((this.m[i] * this.m[j])**0.5)
        // if (isNaN(f.x))
        //     console.error(f)
        return f
    }

    step() {
        // console.log("step ForceNeighborhoodLayout")
        let dt = this.dt
        let directed = this.visibleGraph.dataset.directed
        // Update forces
        let nodes = this.visibleGraph.getNodes()
        for (const n of nodes) {
            this.a[n].x = 0
            this.a[n].y = 0
        }

        // Nodes
        for (const i of nodes) {
            for (const j of nodes) {
                if (i >= j) continue
                let f = this._repulseNodes(i, j)
                this.a[i].add(Vec.mul(f, dt / this.m[i]))
                this.a[j].sub(Vec.mul(f, dt / this.m[j]))
            }
        }

        // Edges
        let n0 = this.visibleGraph.n0
        for (const n of this.visibleGraph.n1) { // Edges 0-1 and 1-0
            if (n === n0) continue
            let f = this._attractEdge(n0, n, 0.5)
            this.a[n0].add(Vec.mul(f, dt / this.m[n0]))
            this.a[n].sub(Vec.mul(f, dt / this.m[n]))
        }
        for (let [i, js] of Object.entries(this.visibleGraph.e11)) { // Edges 1-1
            i = parseInt(i)
            for (const j of js) {
                if (i === j) continue
                let f = this._attractEdge(i, j, 0.3)
                this.a[i].add(Vec.mul(f, dt / this.m[i]))
                this.a[j].sub(Vec.mul(f, dt / this.m[j]))
            }
        }
        // console.log(this.neighborhood)
        for (let [i, js] of Object.entries(this.visibleGraph.e12)) { // Edges 1-2 (and 2-1)
            i = parseInt(i)
            for (const j of js) {
                if (!this.directed && i >= j) continue
                else if (i === j) continue
                let f = this._attractEdge(i, j, 0.1)
                this.a[i].add(Vec.mul(f, dt / this.m[i]))
                this.a[j].sub(Vec.mul(f, dt / this.m[j]))
            }
        }
        if (directed)
            for (let [i, js] of Object.entries(this.visibleGraph.e21)) { // Edges 2-1
                i = parseInt(i)
                for (const j of js) {
                    if (!this.directed && i >= j) continue
                    else if (i === j) continue
                    let f = this._attractEdge(i, j, 0.1)
                    this.a[i].add(Vec.mul(f, dt / this.m[i]))
                    this.a[j].sub(Vec.mul(f, dt / this.m[j]))
                }
            }

        // Edges 2-2 are ignored

        // Normalize forces to max
        // let thrA = this.thrA * this.decayFactor**this.iterations
        let thrA = this.thrA
        // console.log("It", this.iterations, thrA)
        let maxA = Math.max.apply(null, Object.values(this.a).map((v) => Math.abs(v.x)))
        if (maxA > thrA) {
            for (const n of Object.keys(this.a)) {
                this.a[n].x *= thrA / maxA
            }
            // console.log("Max Ax", maxA)
        }
        maxA = Math.max.apply(null, Object.values(this.a).map((v) => Math.abs(v.y)))
        if (maxA > thrA) {
            for (const n of Object.keys(this.a)) {
                this.a[n].y *= thrA / maxA
            }
            // console.log("Max Ay", maxA)
        }

        this.moving = false
        // Update speed and angle
        for (const n of nodes) {
            if (n === this.lockedNode) continue
            this.pos[n].add(Vec.mul(this.v[n], dt))
            this.v[n].add(Vec.mul(this.a[n], dt))

            // TODO ?
            // this.vx[n] = this.vx[n] > 0 ? ((1 + this.vx[n]) ** this.visc) - 1 : 1-((1-this.vx[n]) ** this.visc)
            // this.vy[n] = this.vy[n] > 0 ? ((1 + this.vy[n]) ** this.visc) - 1 : 1-((1-this.vy[n]) ** this.visc)
            // this.v[n].mul(this.visc)
            this.v[n].mul(this.visc * this.decayFactor**this.iteration)

            if (this.v[n].abs() > this.minV)
                this.moving = true
        }
        // console.log(this.pos)
        this.iteration += 1
    }
*/
    cool() {
        this.temp = this.temp*this.decayFactor
    }

    force_r(x) {
        return this.rad ** 2 / x
    }

    force_a(x) {
        return x ** 3 / this.rad
    }
    step() {

        // initialise variables
        let add = 0
        let delta = new Vec(0, 0)
        let nodes = this.visibleGraph.getNodes()

        //action mode check
        if (this.lockedNode != null) {
            this.stateFlag = 1
            this.temp = this.constT
        } else {
            if (this.stateFlag === 1) {
                this.temp *= 10 / Math.sqrt(Math.sqrt(nodes.size))
                this.stateFlag = 0
            }
        }
        if (this.stateFlag === 2) {
            this.stateFlag = 0
            this.temp *= Math.sqrt(nodes.size) / 100
            this.constT = this.temp /(5 * Math.sqrt(nodes.size))
            this.rad *= Math.sqrt(Math.sqrt(nodes.size))
        }

        // calculating repulsive forces
        for (const i of nodes) {
            this.v[i].x = 0
            this.v[i].y = 0
            for (const j of nodes) {
     	        if (i === j) {
     	            continue
     	        }
                 let a = this.pos[i].dist(this.pos[j])
    	        if (a >= 2*this.rad) {
    	            continue
    	        }
     	        delta = new Vec(this.pos[i].x, this.pos[i].y)
     	        delta.sub(this.pos[j])
     	        add = delta.abs()
      	        delta.mul((add === 0.0) ? 0 : this.force_r(add)/add)
      	        this.v[i].add(delta)
           }
        }

        //calculating attracting forces
        for (const [d, es] of Object.entries(this.visibleGraph.edges)) {
            for (let [i, j] of es) {
                // FIXME check
                // if ((i >= j) && (this.directed === false)) {
                //     continue
                // }
                delta = new Vec(this.pos[i].x, this.pos[i].y)
                delta.sub(this.pos[j])
                add = delta.abs()
                delta.mul((add === 0.0) ? 0 : this.force_a(add)/add)
                this.v[i].sub(delta)
                this.v[j].add(delta)
            }
        }

        //recompute positions
        this.moving = false
        for (const i of nodes) {
            if (i == this.lockedNode) {
                continue
            }
            add = this.v[i].abs()
            delta = new Vec(this.v[i].x, this.v[i].y)
            delta.mul((add === 0.0) ? 1 : 1/add)
            delta.mul((add > this.temp) ? this.temp : add)
            this.pos[i].add(delta)
        }

        //check if structure is still moving
        if (this.temp > 0.005) {
            this.moving = true
        }
        if (this.lockedNode == null) {
         this.cool()
        }
        this.iteration += 1
    }
}