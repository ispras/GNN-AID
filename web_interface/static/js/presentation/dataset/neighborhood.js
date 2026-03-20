///
// Node and edges info of the 2nd neighborhood of some node.
// Keeps SVG primitives to draw.
class Neighborhood extends VisibleGraph {
    // static PARTS = ["0-1 edges", "1-1 edges", "2nd nodes", "1-2 edges", "2-2 edges"]
    static MAX_DEPTH = 4
    static PARTS = Array.from({length: Neighborhood.MAX_DEPTH}, (_, k) => k+1);

    constructor(datasetInfo, svgPanel) {
        super(datasetInfo, svgPanel)
        this.layoutFreezeButtonId = this.visView.singleNeighLayoutFreezeId

        // Constants
        this.depthEdgeColors = {
            1: '#ffffff',
            2: '#d0d0d0',
            3: '#888888',
            4: '#555555',
        }
        this.depthNodeRadiuses = {0: 30, 1: 20, 2: 10, 3: 8, 4: 6}
        this.depthNodeStrokeWidthes = {0: 5, 1: 4, 2: 3, 3: 2, 4: 2}
        this.depthEdgeStrokeWidthes = {1: 5.5, 2: 3, 3: 2, 4: 1}

        // Variables
        this.nodes = null // {depth -> List of d-th neighbors nodes}
        this.edges = null // {depth -> List of d-th depth incoming/adjacent edges}
        this.n0 = null // main node
        this.depth = null // neighborhood depth
        this.nodeRadiuses = null // {node -> radius}
        this.nodeStrokeWidthes = null // {node -> StrokeWidth}
        this.edgeColors = null // {edge -> color}
        this.edgeStrokeWidthes = null // {edge -> StrokeWidth}
        this.showDepth = Array(Neighborhood.PARTS+1).fill(true) // whether to show depth
    }

    createListeners() {
        // this.visView.addListener(this.visView.singleClassAsColorId,
        //     (_, v) => this.showClassAsColor(v), this._tag)
        this.visView.addListener(this.visView.singleNeighLayoutId,
            (k, v) => this.setLayout(v), this._tag)
        this.visView.addListener(this.visView.singleNeighNodeId,
            async (k, v) => await this.setNode(parseInt(v)), this._tag)
        this.visView._getById(this.visView.singleNeighNodeId).attr("max", this.datasetInfo.nodes[0]-1)
        this.visView.addListener(this.visView.singleNeighDepthId,
            async (k, v) => await this.setDepth(parseInt(v)), this._tag)

        for (const part of Neighborhood.PARTS)
            this.visView.addListener(this.visView.singleNeighPartsIds[part],
                (_, v) => this.showPart(part, v), this._tag)

        // Order is important
        super.createListeners()
    }

    createVarListeners() {
        super.createVarListeners()

        this.visView.addListener(this.visView.singleClassAsColorId,
            (_, v) => this.showClassAsColor(v), this._tagVar)
    }

    defineVisibleConfig() {
        let node = parseInt(this.visView.getValue(this.visView.singleNeighNodeId))
        let depth = parseInt(this.visView.getValue(this.visView.singleNeighDepthId))
        this.visibleConfig["center"] = node
        this.visibleConfig["depth"] = depth
    }

    // Initialize elements and start layout
    async _build() {
        let node = this.visibleConfig["center"]
        let depth = this.visibleConfig["depth"]
        if (node !== this.n0 || depth !== this.depth) { // Node changed - set graph data from dataset
            this.nodes = this.datasetData.nodes
            this.edges = this.datasetData.edges
            this.n0 = this.nodes[0][0]
            this.depth = this.nodes.length-1
            if (this.depth !== depth) { // Received another depth
                this.visView.setValue(this.visView.singleNeighDepthId, this.depth)
            }

            // Create reverse mappings of nodes and edges for radius, stroke width, color
            this._createMappings()
        }

        await super._build()
        $(this.svgElement).css("background-color", "#404040")
    }

    // // Convert node,edge,graph attributes
    // _convertDatasetData(datasetData) {
    //
    // }

    // Create reverse mappings of nodes and edges to define radius, stroke width, color when drawing
    _createMappings() {
        this.nodesList = []
        this.nodeRadiuses = {}
        this.nodeStrokeWidthes = {}
        for (let d=0; d<this.nodes.length; ++d)
            for (let n of this.nodes[d]) {
                this.nodesList.push(n)
                this.nodeRadiuses[n] = this.depthNodeRadiuses[d]
                this.nodeStrokeWidthes[n] = this.depthNodeStrokeWidthes[d]
            }

        this.edgesList = []
        this.edgeStrokeWidthes = {}
        this.edgeColors = {}
        for (let d=0; d<this.edges.length; ++d)
            for (let [i, j] of this.edges[d]) {
                this.edgesList.push([i, j])
                this.edgeStrokeWidthes[`${i},${j}`] = this.depthEdgeStrokeWidthes[d]
                this.edgeColors[`${i},${j}`] = this.depthEdgeColors[d]
            }
    }

    // Set central node
    async setNode(node) {
        if (node >= this.datasetInfo.nodes[0])
            node = this.datasetInfo.nodes[0]-1
        else if (node < 0)
            node = 0
        if (node === this.n0)
            return

        this.visView.setValue(this.visView.singleNeighNodeId, node, false)
        this.visibleConfig["center"] = node
        await this._reinit(node)
    }

    // Set depth
    async setDepth(depth) {
        if (depth === this.depth)
            return

        this.visView.setValue(this.visView.singleNeighDepthId, depth, false)
        this.visibleConfig["depth"] = depth
        await this._reinit()
    }

    setLayout(layout) {
        if (this.layout) // NOTE: it is needed for some reason :(
            this.layout.stopMoving()
        if (layout == null)
            layout = this.visView.getValue(this.visView.singleNeighLayoutId)
        switch (layout) {
            case "random":
                this.layout = new Layout()
                break
            case "radial":
                this.layout = new RadialNeighborhoodLayout()
                break
            case "force":
                this.layout = new ForceNeighborhoodLayout()
                break
        }
        super.setLayout()
    }

    // Get degree of a node in an induced subgraph (for 2nd neighbors it is less than the actual)
    getDegree(node) {
        let degree = 0
        // TODO O(E) is quite long
        for (const es of this.edges) {
            for (const [i, j] of es) {
                if (i === node || j === node)
                    degree++
            }
        }
        return degree
    }

    getNodes(exceptMain=false) {
        if (exceptMain)
            return this.nodesList.slice(1)
        else
            return this.nodesList
    }

    getEdges() {
        return this.edgesList
    }

    // Get the number of edges depending on show options
    numEdges() {
        return this.edgesList.length
    }

    // Get information HTML
    getInfo() {
        let n = this.n0
        let nodesString = '1'
        for (const [d, ns] of Object.entries(this.nodes))
            if (d > 0)
                nodesString += '+' + ns.length
        let e = this.numEdges()
        return `<b>Neighborhood</b> of '${n}' with ${nodesString} nodes, ${e} edges`
    }

    // Set attributes specific to this node
    setNodeAttributes(node, n) {
        node.r = this.nodeRadiuses[n]
        node.body.setAttribute('stroke-width', this.nodeStrokeWidthes[n])
    }

    // Set attributes specific to this edge
    setEdgeAttributes(edge, i, j) {
        edge.path.setAttribute("stroke-width", this.edgeStrokeWidthes[`${i},${j}`])
        edge.path.setAttribute("stroke-width", this.edgeStrokeWidthes[`${i},${j}`])
        edge.path.setAttribute("stroke", this.edgeColors[`${i},${j}`])
    }

    showPart(part, show) {
        // FIXME
    //     console.assert(Neighborhood.PARTS.includes(part))
    //     if (this.depth < part)
    //         return
    //     this.showDepth[part] = show
    //     if (this.edgePrimitivesBatches)
    //         for (const svg of this.edgePrimitivesBatches[part])
    //             svg.visible(show)
    //     if (this.edgePrimitives) {
    //         for (const es of Object.values(this.edgePrimitives))
    //             for (const edge of Object.values(es))
    //                 edge.visible(show)
    //     }
    //     for (const n of this.nodes[part])
    //         this.nodePrimitives[n].visible(show)
    }
}
