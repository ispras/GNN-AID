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
        this.datasetData = await controller.ajaxRequest('/dataset', {get: "data"})

        // [this.numNodes, this.adj, this.adjIn] = this.dataset.getGraph()
        this.numNodes = this.datasetData.nodes
        this.edges = this.datasetData.edges

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

    createVarListeners(tag) {
        super.createVarListeners(tag)

        this.visView.addListener(this.visView.singleClassAsColorId,
            (_, v) => this.showClassAsColor(v), tag)
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
        let nodes = [] // NOTE copying is not good
        for (let i = 0; i < this.numNodes; i++)
            nodes.push(i)
        return nodes
    }

    getNumNodes() {
        return this.numNodes
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
        this.edgePrimitives = {}
        let directed = this.datasetInfo.directed

        // Edges
        // this.addEdgePrimitivesBatch(
        //    0, this.getEdges(), this.edgeColor, this.edgeStrokeWidth, directed, true)

        // Edges - create individual edge primitives
        let edges = this.getEdges()
        for (let edgeIdx = 0; edgeIdx < edges.length; edgeIdx++) {
            let [i, j] = edges[edgeIdx]
            this.createEdgePrimitive(0, `${i},${j}`, i, j, this.edgeRadius, this.edgeColor, this.edgeStrokeWidth, directed, true)
        }

        // Nodes
        for (let n=0; n<this.numNodes; n++)
            this.createNodePrimitive(this.svgElement, n,
                this.nodeRadius, "circle", this.nodeStrokeWidth, this.nodeColor, true)

        super.createPrimitives()
    }

    // Create or change SVG primitives according to dataset diff
    applyDiffToPrimitives() {
        let nodesAdd = Object.keys(this.datasetDiff.node['features'])
        let edgesAdd = this.datasetDiff.edge['add']

        // Edges add
        for (const [i, j] of edgesAdd) {
            this.edges.push([i, j])
            let edgeKey = `${i},${j}`
            let e = this.createEdgePrimitive(0, edgeKey, i, j, this.edgeRadius, this.edgeAddColor,
                this.edgeStrokeWidth, this.datasetInfo.directed, true)
        }

        // Nodes add
        this.numNodes += nodesAdd.length
        for (const id of nodesAdd) {
            let node = this.createNodePrimitive(this.svgElement, id,
                this.nodeRadius, "circle", this.nodeStrokeWidth, this.nodeColor, true)
        }

        super.applyDiffToPrimitives()
    }

    // Move all primitives on a specified vector
    move(vec) {
        for(const pos of Object.values(this.layout.pos))
            pos.add(Vec.mul(vec, 1/this.scale))
    }

    showEdges(show) {
        if (this.edgePrimitives) {
            for (const es of Object.values(this.edgePrimitives))
                for (const edge of Object.values(es))
                    edge.visible(show)
        }
        if (this.edgePrimitivesBatches) {
            for (const batch of Object.values(this.edgePrimitivesBatches))
                for (const svg of batch)
                    svg.visible(show)
        }
    }

}