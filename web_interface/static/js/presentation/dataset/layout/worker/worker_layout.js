class WorkerLayout {
    constructor(graph, params = {}) {
        this.graph = graph
        this.nodeCount = graph.nodeCount
        this.pos = graph.positions
        this.extra = graph.extra || {}

        this.lockedNode = null
        this.iteration = 0
    }

    step() {
        return false
    }
}