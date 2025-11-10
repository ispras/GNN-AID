class HeteroGraph extends Graph {
    constructor(datasetInfo, element) {
        super(datasetInfo, element)
        // this.layoutFreezeButtonId = this.visView.singleGraphLayoutFreezeId

        // Variables
        this.nodeTypes = null
        this.edgeTypes = null

        this.numNodes = null // Number of nodes, {node type: count}
        this.edges = null // {edge type: List of pairs [i, j]}
        this.nodeIndexOffsets = null // Offsets for enumerating over all nodes, {node type: index offset}
        this.nodeForms = null // SVG forms for node types, {node type: form}

        // Constants
        this.edgeColor = '#ffffff' // TODO
    }

    async _build() {
        this.numNodes = this.datasetData.nodes
        this.edges = this.datasetData.edges[0]

        this.nodeTypes = Object.keys(this.datasetData.nodes)
        this.edgeTypes = Object.keys(this.datasetData.edges[0])
        this.nodeIndexOffsets = {}
        this.nodeForms = {}
        let offset = 0
        let i = 0
        for (const nt of this.nodeTypes) {
            this.nodeIndexOffsets[nt] = offset
            offset += this.numNodes[nt]
            if (i === 0)
                this.nodeForms[nt] = "circle"
            else
                this.nodeForms[nt] = `poly${i+2}`
            ++i
        }

        await VisibleGraph.prototype._build.call(this)
    }

    // Get degree of a node of a graph
    getDegree(node, graph) {
        // TODO hetero
        let degree = 0
        for (const [i, j] of this.edges) {
            if (i === node || j === node)
                degree++
        }
        return degree
    }

    getNodes() {
        let nodes = new Set() // NOTE copying is not good
        for (const nt of this.nodeTypes)
            for (let i = 0; i < this.numNodes[nt]; i++)
                nodes.add(i + this.nodeIndexOffsets[nt])
        return nodes
    }

    getNumNodes() {
        return Object.values(this.numNodes).sum()
    }

    getEdges(type) {
        const processEdgeType = (edgeType) => {
            let [st, _, dt] = edgeType.split(',')
            st = st.slice(1, -1)
            dt = dt.slice(1, -1)
            return this.edges[edgeType].map(([s, d]) =>
                [s + this.nodeIndexOffsets[st], d + this.nodeIndexOffsets[dt]]
            )
        }

        if (type == null) { // Get all edges
            return this.edgeTypes.flatMap(processEdgeType)
        }
        return processEdgeType(type)
    }

    // Get a subgraph induced on a given set of nodes
    getInducedSubgraph(nodesSet) {
        // TODO hetero
        let edgeList = []
        for (const [i, j] of this.edges) {
            if (nodesSet.has(i) && nodesSet.has(j))
                edgeList.push([i, j])
        }
        return edgeList
    }

    // Get a list of attributes values for a node (of a graph)
    getNodeAttrs(node, graph=0) {
        let nodeAttrs = this.datasetData["node_attributes"]
        if (nodeAttrs == null)
            return null
        let nt = this.nodeTypes[findMaxIndex(Object.values(this.nodeIndexOffsets), node)]
        node -= this.nodeIndexOffsets[nt]
        let res = []
        for (const a of this.datasetInfo["node_attributes"][nt]["names"]) {
            let attrs = nodeAttrs[nt][a][graph]
            res.push(attrs[node])
        }
        return res
    }

    // Create HTML for SVG primitives on the given element
    createPrimitives() {
        this.svgElement.innerHTML = ''
        this.nodePrimitives = {}
        let directed = this.datasetInfo.directed

        // Edges
        for (const et of this.edgeTypes) {
            this.addEdgePrimitivesBatch(
                et, this.getEdges(et), this.edgeColor, this.edgeStrokeWidth, directed, true)
        }

        // Nodes
        for (const nt of this.nodeTypes) {
            for (let n = 0; n < this.numNodes[nt]; n++) {
                // let name = n.toString() + ',' + nt
                let name = n + this.nodeIndexOffsets[nt]
                this.createNodePrimitive(this.svgElement, name,
                    this.nodeRadius, this.nodeForms[nt], this.nodeStrokeWidth, this.nodeColor, true)
            }
        }

        VisibleGraph.prototype.createPrimitives.call(this)
    }
}