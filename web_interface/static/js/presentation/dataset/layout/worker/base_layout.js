class WorkerRandomLayout extends WorkerLayout {
    constructor(graph, params = {}) {
        super(graph, params)
        this.initialized = false
    }

    step() {
        if (this.initialized) {
            return false
        }
        this.initialized = true
        return true
    }
}