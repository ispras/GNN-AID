/*
function randomPairing(func, numNodes, max=100) {
    if (numNodes <= max)
        for (let i = 0; i < numNodes; i++)
            for (let j = i+1; j < numNodes; j++) {
                func(i, j)
            }
    else {
        for (let i = 0; i < max*max; i++) {
            let x = Math.floor(numNodes * Math.random())
            let y = Math.floor(numNodes * Math.random())
            if (x === y) continue
            x < y ? func(x, y) : func(y, x)
        }
    }
}
*/

class ForceLayout extends Layout {
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

        this.numNodes = null // number of nodes for Graph
        this.edges = null // list of edges
        this.v = {} // velocity, node -> Vec
        this.a = {} // acceleration, node -> Vec
        this.m = {} // node -> mass

        this.thrA = 150 // max total acceleration on a node
        this.decayFactor = 0.98 // power of forces decay over iterations
    }

    setVisibleGraph(visibleGraph) {
        console.assert(visibleGraph instanceof Graph)
        this.edges = visibleGraph.getEdges()
        this.numNodes = visibleGraph.numNodes
        super.setVisibleGraph(visibleGraph)
    }

    respawn() {
        // Respawn v and a
        super.respawn()
        this.v = {}
        this.a = {}
        this.m = {}
        for (let n = 0; n < this.numNodes; n++)
            this.m[n] = 0.1
        for (let n = 0; n < this.numNodes; n++) {
            this.v[n] = new Vec(0, 0)
            this.a[n] = new Vec(0, 0)
            for (const [i, j] of this.visibleGraph.getEdges()) {
                this.m[i]++
                this.m[j]++
            }

            // if (this.directed) {
            //     if (n in this.visibleGraph.adj) {
            //         this.m[n] += this.visibleGraph.adj[n].size
            //         for (const nn of this.visibleGraph.adj[n])
            //             this.m[nn] += 1
            //     }
            // }
            // else
            //     this.m[n] = Math.max(1, this.visibleGraph.getDegree(n))
        }
    }
/*
    _repulseNodes(i, j) {
        const d0 = 0.4
        const dMax = 3
        const k = 0.1
        let d = this.pos[i].dist(this.pos[j])
        let m = 1/(d0 + k * d**2)
        if (m < 1/dMax) m = 0
        let f = Vec.sub(this.pos[j], this.pos[i])
        f.mul(-this.alpha / d * m)
        f.mul((this.m[i] * this.m[j])**0.5)
        return f
    }

    _attractEdge(i, j, d0=0.1) {
        let d = this.pos[i].dist(this.pos[j])
        let m = (d - d0) ** 2
        let f = Vec.sub(this.pos[j], this.pos[i])
        f.mul(this.beta / d * m)
        f.mul((this.m[i] * this.m[j])**0.5)
        return f
    }

    step() {
        // console.log("step ForceLayout")

        let dt = this.dt
        // Update forces
        for (let n = 0; n < this.numNodes; n++) {
            this.a[n].x = 0
            this.a[n].y = 0
        }

        // Nodes
        // let func = (i, j) => {
        //     let f = this._repulseNodes(i, j)
        //     this.a[i].add(Vec.mul(f, dt / this.m[i]))
        //     this.a[j].sub(Vec.mul(f, dt / this.m[j]))
        // }
        // randomPairing(func, nodes.size, 300)
        // TODO this is the most computationally hard part !!!
        for (let i = 0; i < this.numNodes; i++) {
            for (let j = i+1; j < this.numNodes; j++) {
                // if (i >= j) continue
                let f = this._repulseNodes(i, j)
                this.a[i].add(Vec.mul(f, dt / this.m[i]))
                this.a[j].sub(Vec.mul(f, dt / this.m[j]))
            }
        }
        // console.log(`Time of _repulseNodes: ${performance.now() - t}ms`)


        // Edges
        for (const [i, j] of this.edges) {
            if (i === j) continue // omit self-loops
            let f = this._attractEdge(i, j, 0.3)
            this.a[i].add(Vec.mul(f, dt / this.m[i]))
            this.a[j].sub(Vec.mul(f, dt / this.m[j]))
        }

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
        for (let n = 0; n < this.numNodes; n++) {
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
        let edges = this.visibleGraph.getEdges()

        //action mode check
        if (this.stateFlag === 2) {
            this.stateFlag = 0
            this.temp *= nodes.size / 100
            this.constT = this.temp / (10 *Math.sqrt(nodes.size))
            this.rad *= Math.sqrt(Math.sqrt(nodes.size))
        }
        if (this.lockedNode != null) {
            this.stateFlag = 1
            this.temp = this.constT
        } else {
            if (this.stateFlag === 1) {
                this.stateFlag = 0
                this.temp *= 10 / Math.sqrt(Math.sqrt(nodes.size))
            }
        }

        // calculating repulsive forces
        for (const i of nodes) {
            this.v[i].x = 0
            this.v[i].y = 0
            for (const j of nodes) {
     	       if (i === j) {
     	           continue
     	       }
     	       if (this.pos[i].dist(this.pos[j]) >= 2*this.rad) {
     	           continue
     	       }
     	       delta = new Vec(this.pos[i].x, this.pos[i].y)
     	       delta.sub(this.pos[j])
     	       add = delta.abs()
      	      delta.mul((add === 0) ? 0 : this.force_r(add)/add)
      	      this.v[i].add(delta)
           }
       }

        //calculating attracting forces
 	    for (const [i, j] of edges) {
            delta = new Vec(this.pos[i].x, this.pos[i].y)
            delta.sub(this.pos[j])
            add = delta.abs()
            delta.mul((add === 0) ? 0 : this.force_a(add)/add)
            this.v[i].sub(delta)
            this.v[j].add(delta)
        }
         //recompute positions
        this.moving = false
        for (const i of nodes) {
            if (i === this.lockedNode) {
                continue
            }
            add = this.v[i].abs()
            delta = new Vec(this.v[i].x, this.v[i].y)
            delta.mul((add === 0.0) ? 1 : 1/add)
            delta.mul((add > this.temp) ? this.temp : add)
            this.pos[i].add(delta)
       }
        //check if structure is still moving
       if ((this.temp > 0.005)) {
           this.moving = true
       }
    if (!this.lockedNode) {
        this.cool()
    }
    this.iteration += 1
    }

}