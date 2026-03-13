function createLayout(layoutType, graph, params) {
    switch (layoutType) {
        case 'base':
            return new WorkerRandomLayout(graph, params)
        case 'force':
            return new WorkerForceLayout(graph, params)
        case 'forceNeighborhood':
            return new WorkerForceNeighborhoodLayout(graph, params)
        default:
            throw new Error(`Unknown layout type: ${layoutType}`)
    }
}