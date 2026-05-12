/// Parse array that is int or a list of ints separated by a comma
function parseIntOrList(nodeString) {
    if (nodeString.includes(','))
        return nodeString.split(',').map(s => parseInt(s.trim()))
    else
        return [parseInt(nodeString)]
}

function sameSet(a, b) {
    const setA = new Set(a)
    const setB = new Set(b)

    if (setA.size !== setB.size) return false

    for (const x of setA) {
        if (!setB.has(x)) return false
    }
    return true
}

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
            0: '#ffffff',
            1: '#ffffff',
            2: 'rgba(176,176,176,0.6)',
            3: 'rgba(136,136,136,0.4)',
            4: 'rgba(55,55,55,0.2)',
        }
        this.depthNodeRadiuses = {0: 30, 1: 20, 2: 10, 3: 8, 4: 6}
        this.depthNodeStrokeWidthes = {0: 5, 1: 4, 2: 3, 3: 2, 4: 2}
        this.depthEdgeStrokeWidthes = {0: 1.5, 1: 1.5, 2: 1, 3: 0.8, 4: 0.5}
        // this.depthEdgeStrokeWidthes = {1: 5.5, 2: 3, 3: 2, 4: 1}

        // Variables
        this.nodes = null // {depth -> List of d-th neighbors nodes}
        this.edges = null // {depth -> List of d-th depth incoming/adjacent edges}
        this.n0 = null // main nodes list
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
            async (k, v) => await this.setNode(v), this._tag)
        this.visView._getById(this.visView.singleNeighNodeId).attr("max", this.datasetInfo.nodes[0]-1)
        this.visView.addListener(this.visView.singleNeighDepthId,
            async (k, v) => await this.setDepth(parseFloat(v)), this._tag)

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
        let node = parseIntOrList(this.visView.getValue(this.visView.singleNeighNodeId))
        let depth = parseFloat(this.visView.getValue(this.visView.singleNeighDepthId))
        this.visibleConfig["center"] = node
        this.visibleConfig["depth"] = depth
    }

    _convertDatasetVar(datasetVar) {
        if (!datasetVar) return null;
        const converted = {};

        const nodeKeys = this.getNodes()
        const edges = this.getEdges()
        const edgeKeys = edges.map(([i, j]) => `${i},${j}`)

        for (const elem of VisibleGraph.ELEMS) {
            if (!datasetVar[elem]) continue;

            converted[elem] = {};
            for (const [field, dataArray] of Object.entries(datasetVar[elem])) {
                if (!dataArray || !Array.isArray(dataArray)) continue;

                if (elem === "graph") {
                    converted[elem][field] = dataArray;
                    continue;
                }

                if (elem === "edge") {
                    const actualArray =
                        dataArray.length === 1 && Array.isArray(dataArray[0]) ? dataArray[0] : dataArray;

                    const varDict = {};
                    edgeKeys.forEach((key, index) => {
                        if (index < actualArray.length && actualArray[index] !== null) {
                            varDict[key] = actualArray[index];
                        }
                    });

                    converted[elem][field] = varDict;
                    continue;
                }

                const varDict = {};
                nodeKeys.forEach((key, index) => {
                    if (index < dataArray.length && dataArray[index] !== null) {
                        varDict[key] = dataArray[index];
                    }
                });

                converted[elem][field] = varDict;
            }

            if (Object.keys(converted[elem]).length === 0)
                delete converted[elem]
        }

        return converted;
    }


    // Initialize elements and start layout
    async _build() {
        let center = this.visibleConfig["center"]
        let depth = this.visibleConfig["depth"]
        if (!sameSet(center, this.n0) || depth !== this.depth) { // Node changed - set graph data from dataset
            this.nodes = this.datasetData.nodes
            this.edges = this.datasetData.edges
            this.n0 = this.nodes[0]
            this.depth = depth
            // this.depth = this.nodes.length-1
            // if (this.depth !== depth) { // Received another depth
            //     this.visView.setValue(this.visView.singleNeighDepthId, this.depth)
            // }

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

    // Set central node(s)
    async setNode(center) {
        center = parseIntOrList(center)
        for (let i=0; i<center.length; ++i) {
            let n = center[i]
            if (n >= this.datasetInfo.nodes[0])
                center[i] = this.datasetInfo.nodes[0]-1
            else if (n < 0)
                center[i] = 0
        }
        if (sameSet(center, this.n0))
            return

        this.visView.setValue(this.visView.singleNeighNodeId, center, false)
        this.visibleConfig["center"] = center
        await this._reinit(center)
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
