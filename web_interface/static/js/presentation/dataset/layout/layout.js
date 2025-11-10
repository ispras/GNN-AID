// Common class managing Neighborhood nodes positions according to some layout.
// Layout model is defined in subclasses.
class Layout {
    constructor() {
        this.visibleGraph = null
        this.directed = null

        this.dt = 0.1
        this.moving = false // whether are still moving
        this.iteration = 0 // number of iterations
        this.pos = {} // node -> position
        this.lockedNode = null // node grabbed by user, its position is fixed
        this.freeze = false // while true do not apply forces
    }

    // Set the VisibleGraph
    setVisibleGraph(visibleGraph) {
        this.visibleGraph = visibleGraph
        this.directed = visibleGraph.datasetInfo.directed
        this.respawn()
        this.startMoving()
        this.run()
    }

    // Respawn positions when a first show
    respawn() {
        this.iteration = 0
        this.pos = {}
        let nodes = this.visibleGraph.getNodes()
        let r = nodes.size ** 0.5
        for (const n of nodes)
            // TODO increase zone according to num of nodes
            this.pos[n] = new Vec(r*Math.random(), r*Math.random())
    }

    // Start recomputing positions - when respawn or manual move
    startMoving() {
        if (this.freeze) return
        this.iteration = 0
        if (!this.moving) {
            this.moving = true
            this.run()
        }
    }

    // Stop running
    stopMoving() {
        this.moving = false
    }

    // Toggle freeze flag
    setFreeze(freeze) {
        this.freeze = freeze
        if (this.freeze)
            this.stopMoving()
        else
            this.startMoving() // TODO this.iteration = 0 leads to high volatility
    }

    // Start a cycle of searching an optimal position
    async run() {
        while ((this.moving || (this.lockedNode != null)) && !this.freeze) {
            //let t = performance.now()
            this.step()
            // console.log(`Time of step(): ${performance.now() - t}ms`)
            await sleep(50)
        }
    }

    // Recompute positions according to the layout model
    step() {
        // Implement in subclass
        this.moving = false
    }

    // Fix node position
    lock(node, pos) {
        this.lockedNode = node
        this.pos[this.lockedNode].set(pos)
    }

    // Unlock node
    release() {
        this.lockedNode = null
    }
}
