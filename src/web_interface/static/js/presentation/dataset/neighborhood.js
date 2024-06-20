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
        this.edgeColor = {
            1: '#000000',
            2: '#242424',
            3: '#838383',
            4: '#d0d0d0',
        }
        this.nodeRadiuses = {0: 30, 1: 20, 2: 10, 3: 8, 4: 6}
        this.nodeStrokeWidthes = {0: 5, 1: 4, 2: 3, 3: 2, 4: 2}
        this.edgeStrokeWidthes = {1: 5.5, 2: 3, 3: 2, 4: 1}
        this.nodeRadius = this.nodeRadiuses[0]

        // Variables
        this.nodes = null // {depth -> Set of d-th neighbors nodes}
        this.edges = null // {depth -> Set of d-th depth incoming/adjacent edges}
        this.n0 = null // main node
        this.depth = null // neighborhood depth
        this.showDepth = Array(Neighborhood.PARTS+1).fill(true) // whether to show depth
    }

    createListeners() {
        this.visView.addListener(this.visView.singleClassAsColorId,
            (_, v) => this.showClassAsColor(v), this._tag)
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
        }

        await super._build()
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

    // Check parameters to decide whether to turn on a light mode
    checkLightMode() {
        this.nodesVisible = this.scale >= LIGHT_MODE_SCALE_THRESHOLD_SINGLE
        for (const node of Object.values(this.nodePrimitives))
            node.lightMode = !this.nodesVisible
        super.checkLightMode(this.nodesVisible)
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
        let set = new Set()
        let d = 0
        for (const ns of this.nodes) {
            if (d === 0 && exceptMain) {
                d++
                continue
            }
            ns.forEach(n => set.add(n))
        }
        return set
    }

    getEdges() {
        console.error('UNDEFINED getEdges() for Neighborhood')
    }

    // Get the number of edges depending on show options
    numEdges() {
        let e = 0
        for (const es of Object.values(this.edges))
            e += es.length
        return e
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

    // Create HTML for SVG primitives on the given element
    createPrimitives() {
        // this.svgElement.innerHTML = ''
        this.svgPanel.$svg.empty()
        this.nodePrimitives = {}
        this.edgePrimitivesBatches = []

        // Add edges
        for (const [d, es] of Object.entries(this.edges))
            this.addEdgePrimitivesBatch(
                d, es, this.edgeColor[d], this.edgeStrokeWidthes[d], this.datasetInfo.directed, this.showDepth[d])

        // Add nodes
        for (const [d, ns] of Object.entries(this.nodes)) {
            for (const n of ns) {
                this.createNodePrimitive(
                    this.svgElement, n, this.nodeRadiuses[d], this.nodeStrokeWidthes[d],
                    this.nodeColor, this.showDepth[d])
            }
        }

        super.createPrimitives()
    }

    showPart(part, show) {
        console.assert(Neighborhood.PARTS.includes(part))
        if (this.depth < part)
            return
        this.showDepth[part] = show
        for (const svg of this.edgePrimitivesBatches[part])
            svg.visible(show)
        for (const n of this.nodes[part])
            this.nodePrimitives[n].visible(show)
    }
}
