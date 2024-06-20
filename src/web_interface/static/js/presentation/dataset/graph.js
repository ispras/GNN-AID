class Graph extends VisibleGraph {
    constructor(datasetInfo, element) {
        super(datasetInfo, element)
        this.layoutFreezeButtonId = this.visView.singleGraphLayoutFreezeId

        // Variables
        this.numNodes = null // Number of nodes
        this.edges = null // List of pairs [i, j]

        // Constants
        this.edgeColor = '#ffffff'
    }

    async _build() {
        // Set graph data from dataset
        this.datasetData = await Controller.ajaxRequest('/dataset', {get: "data"})

        // [this.numNodes, this.adj, this.adjIn] = this.dataset.getGraph()
        this.numNodes = this.datasetData.nodes
        this.edges = this.datasetData.edges[0]

        await super._build()
        $(this.svgElement).css("background-color", "#404040")
    }

    // Variable part of drop - to be overridden
    _drop() {
        super._drop()
        $(this.svgElement).css("background-color", "#e7e7e7")
    }

    createListeners() {
        this.visView.addListener(this.visView.singleGraphLayoutId,
            (_, v) => this.setLayout(v), this._tag)
        this.visView.addListener(this.visView.singleGraphEdgesId,
            (_, v) => this.showEdges(v), this._tag)

        super.createListeners()
    }

    createVarListeners() {
        super.createVarListeners()

        this.visView.addListener(this.visView.singleClassAsColorId,
            (_, v) => this.showClassAsColor(v), this._tagVar)
    }

    // Check parameters to decide whether to turn on a light mode
    checkLightMode() {
        this.nodesVisible = this.scale >= LIGHT_MODE_SCALE_THRESHOLD_SINGLE
        for (const node of Object.values(this.nodePrimitives))
            node.lightMode = !this.nodesVisible
        super.checkLightMode(this.nodesVisible)
    }

    setLayout(layout) {
        if (layout == null)
            layout = this.visView.getValue(this.visView.singleGraphLayoutId)
        switch (layout) {
            case "random":
                this.layout = new Layout()
                break
            case "force":
                this.layout = new ForceLayout()
                break
        }
        super.setLayout()
    }

    // Get degree of a node of a graph
    getDegree(node, graph) {
        let degree = 0
        for (const [i, j] of this.edges) {
            if (i === node || j === node)
                degree++
        }
        return degree
    }

    getNodes() {
        let nodes = new Set() // NOTE copying is not good
        for (let i = 0; i < this.numNodes; i++)
            nodes.add(i)
        return nodes
    }

    getEdges() {
        return this.edges
    }

    // Get a subgraph induced on a given set of nodes
    getInducedSubgraph(nodesSet) {
        let edgeList = []
        for (const [i, j] of this.edges) {
            if (nodesSet.has(i) && nodesSet.has(j))
                edgeList.push([i, j])
        }
        return edgeList

        // let adj = this.adj
        // let adjIn = this.adjIn
        // if (this.isMultipleGraphs()) {
        //     adj = adj[graph]
        //     if (this.directed)
        //         adjIn = adjIn[graph]
        // }
        // let newAdj = {}
        // let newAdjIn = {}
        //
        // for (const n of nodes) {
        //     newAdj[n] = new Set()
        //     for (const n1 of adj[n]) {
        //         if (nodes.has(n1))
        //             newAdj[n].add(n1)
        //     }
        //     if (this.directed) {
        //         newAdjIn[n] = new Set()
        //         for (const n1 of adjIn[n]) {
        //             if (nodes.has(n1))
        //                 newAdjIn[n].add(n1)
        //         }
        //     }
        // }
        // if (asEdgeList) {
        //     // Return result as edge list
        //     let edgeList = []
        //     for (const [n, ns] of Object.entries(newAdj))
        //         for (const n1 of ns)
        //             edgeList.push([n, n1])
        //     if (this.directed)
        //         for (const [n, ns] of Object.entries(newAdjIn))
        //             for (const n1 of ns)
        //                 edgeList.push([n, n1])
        //     return edgeList
        // }
        // else
        //     return [nodes.size, newAdj, newAdjIn]
    }

    // Get information HTML
    getInfo() {
        return '<b>Whole graph</b>'
    }

    // Create HTML for SVG primitives on the given element
    createPrimitives() {
        this.svgElement.innerHTML = ''
        this.nodePrimitives = {}
        let directed = this.datasetInfo.directed

        // Edges
        this.addEdgePrimitivesBatch(
            0, this.getEdges(), this.edgeColor, this.edgeStrokeWidth, directed, true)

        // Nodes
        for (let n=0; n<this.numNodes; n++)
            this.createNodePrimitive(this.svgElement, n,
                this.nodeRadius, this.nodeStrokeWidth, this.nodeColor, true)

        super.createPrimitives()
    }

    // Move all primitives on a specified vector
    move(vec) {
        for(const pos of Object.values(this.layout.pos))
            pos.add(Vec.mul(vec, 1/this.scale))
    }

    showEdges(show) {
        for (const svg of this.edgePrimitivesBatches[0])
            svg.visible(show)
    }

}