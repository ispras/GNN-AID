class MultipleGraphs extends VisibleGraph {
    static MAX_DEPTH = 5

    constructor(datasetInfo, svgPanel) {
        super(datasetInfo, svgPanel)
        this.layoutFreezeButtonId = this.visView.multiLayoutFreezeId

        // Constants
        this.edgeColor = '#000'

        // Variables
        this.graphPrimitives = null // {graph ix -> primitive} on HTML element
        this.graphs = null // {graph ix -> Graph}
        this.layouts = null // {graph ix -> layouts}
        this.graphGrabbed = null // graph index, whose node is currently dragged
        this.arrange = null // arranges graph

        this._count = null // Chosen param value
        this._graph = null // Chosen param value (!= visibleConfig["center"])
        this._arrange = null // Chosen param value
        this._layout = null // Chosen param value
    }

    createListeners() {
        this.visView.addListener(this.visView.multiCountId,
            async (_, v) => await this.setCount(v), this._tag)
        this.visView.addListener(this.visView.multiGraphId,
            async (_, v) => await this.setGraph(parseInt(v)), this._tag)
        this.visView.addListener(this.visView.multiDepthId,
            async (_, v) => await this.setDepth(parseInt(v)), this._tag)
        this.visView.addListener(this.visView.multiLayoutId,
            (_, v) => this.setLayout(v), this._tag)
        this.visView.addListener(this.visView.multiArrangeId,
            (_, v) => this.setArrange(v), this._tag)
        this.visView._getById(this.visView.multiGraphId).attr("max", this.datasetInfo.count-1)

        super.createListeners()
    }

    createVarListeners() {
        super.createVarListeners()

        this.visView.addListener(this.visView.multiNodeTypeAsColorId,
            (_, v) => this.showClassAsColor(v), this._tagVar)
    }

    // Handle nodes and whole SVG drag&drop
    handleDragging() {
        this.svgElement.onmousedown = (e) => {
            if (e.buttons === 1 && e.ctrlKey) {
                this.svgGrabbed = true
                this.svgGrabbedMousePos.x = e.screenX
                this.svgGrabbedMousePos.y = e.screenY
                this.svgGrabbedScreenPos.x = this.screenPos.x
                this.svgGrabbedScreenPos.y = this.screenPos.y
                this.svgElement.style.cursor = 'grabbing'
            }
        }
        $(window).mouseup((e) => {
            this.svgGrabbed = false
            this.nodeGrabbed = null
            this.svgElement.style.cursor = 'default'
            if (this.layouts && this.graphGrabbed != null) {
                this.layouts[this.graphGrabbed].release()
            }
            this.graphGrabbed = null
        })
        this.svgElement.onmousemove = (e) => {
            this.mousePos.x = e.offsetX
            this.mousePos.y = e.offsetY
            if (this.svgGrabbed) {
                this.screenPos.x = this.svgGrabbedScreenPos.x - e.screenX + this.svgGrabbedMousePos.x
                this.screenPos.y = this.svgGrabbedScreenPos.y - e.screenY + this.svgGrabbedMousePos.y
                this.draw()
            }

            else if (this.nodeGrabbed != null) {
                this.layouts[this.graphGrabbed].lock(this.nodeGrabbed, Vec.add(this.mousePos, this.svgPos).mul(1/this.scale))
                this.layouts[this.graphGrabbed].startMoving()
                this.draw()
            }
            this.debugInfo()
        }

        // Handle zoom
        this.svgElement.onwheel = (e) => {
            if (e.ctrlKey) {
                e.preventDefault()
                let z = e.wheelDelta > 0 ? this.zoomFactor : 1/this.zoomFactor
                if (this.scale * z > this.scaleMax || this.scale * z < this.scaleMin)
                    return
                this.scale *= z
                for (const graph of Object.values(this.graphs))
                    graph.scale = this.scale

                // Compute SVG and screen new positions
                this.screenPos.x += (z-1)*(this.mousePos.x + this.svgPos.x)
                this.screenPos.y += (z-1)*(this.mousePos.y + this.svgPos.y)

                this.checkLightMode()
                this.draw()
                // Update mouse pos AFTER draw
                this.mousePos.x = e.layerX
                this.mousePos.y = e.layerY
            }
            this.debugInfo()
        }

        // To avoid computing scroll from screenPos
        this.svgElement.parentElement.onscroll = (e) => {
            this.screenPos.x = this.svgElement.parentElement.scrollLeft + this.svgPos.x
            this.screenPos.y = this.svgElement.parentElement.scrollTop + this.svgPos.y
        }
    }

    async drawCycle() {
        while (true) {
            if (!this.alive) break
            // At least 1 layout is moving
            for (const layout of Object.values(this.layouts))
                if (layout.moving) {
                    this.draw()
                    break
                }
            await sleep(100)
        }
    }

    defineVisibleConfig() {
        this.visibleConfig["center"] = parseInt(this.visView.getValue(this.visView.multiGraphId))
        this.visibleConfig["depth"] = parseInt(this.visView.getValue(this.visView.multiDepthId))
    }

    async _build() {
        // Set graph data from dataset
        this.count = this.datasetInfo.count
        this.graphs = {} // ix -> Graph

        let nodesData = this.datasetData.nodes
        let edgesData = this.datasetData.edges
        let graphsData = this.datasetData.graphs

        let ix = 0
        for (let g of graphsData) {
            // for (let i=0; i<this.count; i++) { // TODO what if not all graphs ?

            let graph = new Graph(this.datasetInfo, this.svgPanel)
            graph.numNodes = nodesData[ix]
            graph.edges = edgesData[ix]
            graph.scale = this.scale
            this.graphs[g] = graph
            graph.onNodeClick = this.onNodeClick
            graph.datasetVar = {}
            ix++
        }
        // To avoid calling _build several times
        if (this.visibleConfig["center"] !== null) {
            this._graph = this.visibleConfig["center"] // fixme what if not
            this._count = "several" // fixme what if not several?
        }
        await super._build()
    }

    _drop() {
        super._drop()
        this.graphs = null
        this.layouts = null
        this.graphGrabbed = null
        this.arrange = null
    }

    _buildVar() {
        for (const [ix, graph] of Object.entries(this.graphs)) {
            graph.datasetVar = {}
            for (const satellite of VisibleGraph.SATELLITES)
                if (satellite in this.datasetVar)
                    graph.datasetVar[satellite] = this.datasetVar[satellite][ix]
        }

        // Check for feature color coding possibility
        if (this.oneHotableFeature) {
            this.coloredNodes = createSetOfColors(
                this.datasetVar['features'][0][0].length, this.svgPanel.$svg)
            this.visView.setEnabled(this.visView.multiNodeTypeAsColorId, true)
        }
        else {
            this.coloredNodes = null
            this.visView.setEnabled(this.visView.multiNodeTypeAsColorId, false)
        }

        super._buildVar()
    }

    // Update according to show mode - all graphs or some
    async setCount(count) {
        if (count == null)
            count = this.visView.getValue(this.visView.multiCountId)
        if (count === this._count)
            return

        console.log("setCount")
        switch (count) {
            case "all":
                this.visibleConfig["center"] = null
                break
            case "several":
                this.visibleConfig["center"] = parseInt(this.visView.getValue(this.visView.multiGraphId))
                break
            default:
                console.error('Not implemented')
        }
        this._count = count
        this._arrange = null // need to be recalculated
        this._layout = null // need to be recalculated
        await this._reinit()
    }

    // Update according to a graph to be shown
    async setGraph(graph) {
        if (graph == null)
            graph = this.visView.getValue(this.visView.multiGraphId)
        if (graph === this._graph)
            return

        console.log("setGraph")
        switch (this.visView.getValue(this.visView.multiCountId)) {
            case "several":
                this.visibleConfig["center"] = graph
                this.visibleConfig["depth"] = parseInt(this.visView.getValue(this.visView.multiDepthId))
                break
            case "all":
                this.visibleConfig["center"] = null
                this.visibleConfig["depth"] = null
                break
            default:
                console.error('Not implemented')
        }
        this._graph = graph
        this._layout = null // need to be recalculated
        this._arrange = null // need to be recalculated
        await this._reinit()
    }

    // Set depth
    async setDepth(depth) {
        if (depth === this.visibleConfig["depth"])
            return

        this.visView.setValue(this.visView.multiDepthId, depth, false)
        this._layout = null // need to be recalculated
        this._arrange = null // need to be recalculated
        this.visibleConfig["depth"] = depth
        await this._reinit()
    }

    // Update according to a chosen arrange method
    setArrange(arrange) {
        if (arrange == null)
            arrange = this.visView.getValue(this.visView.multiArrangeId)
        if (arrange === this._arrange)
            return

        console.log("setArrange")
        switch (arrange) {
            case "vertical":
                this.arrange = new VerticalArrange()
                break
            case "free":
                this.arrange = new Arrange()
                break
            case "grid":
                this.arrange = new GridArrange()
                break
        }
        this._arrange = arrange
        this.arrange.setGraphs(this.graphs)
        this.draw()
    }

    // Update layout for all graphs
    setLayout(layout) {
        if (layout == null)
            layout = this.visView.getValue(this.visView.multiLayoutId)
        if (layout === this._layout)
            return

        console.log("setLayout")
        if (this.layouts)
            for (const layout of Object.values(this.layouts))
                layout.stopMoving()
        this.layouts = {}
        for (const [ix, graph] of Object.entries(this.graphs)) {
            let aLayout = null
            switch (layout) {
                case "random":
                    aLayout = new Layout()
                    break
                case "force":
                    aLayout = new ForceLayout()
                    break
                default:
                    console.error('Unknown layout', layout)
            }
            this.layouts[ix] = aLayout
            graph.layout = aLayout
            aLayout.setVisibleGraph(graph)
        }
        this._layout = layout
        this.setArrange()
        this.draw()
    }

    freezeLayout(freeze) {
        for (let g of Object.values(this.graphs))
            g.layout.setFreeze(freeze)
    }

    // Check parameters to decide whether to turn on a light mode
    checkLightMode() {
        this.nodesVisible = this.scale >= LIGHT_MODE_SCALE_THRESHOLD_MULTI
        for (let aGraph of Object.values(this.graphs))
            for (const node of Object.values(aGraph.nodePrimitives))
                node.lightMode = !this.nodesVisible
        for (const graph of Object.values(this.graphPrimitives))
            graph.lightMode = !this.nodesVisible
        super.checkLightMode(this.nodesVisible)
    }

    // Assign/remove SVG primitives for node satellites: labels, features, predictions, etc
    setSatellite(satellite, on=true) {
        let $g = this.svgPanel.get("graphs-" + satellite)
        if (!on) {
            // Remove SVGs
            if (satellite === 'features') {
                for (const graph of Object.values(this.graphs))
                    graph.setSatellite('features', null)
                    this.showClassAsColor(false)
            }

            else {
                $g.empty()
                for (const graph of Object.values(this.graphPrimitives))
                    graph.satellites[satellite].blocks = null
            }
        }
        else { // Draw or update SVGs
            if (satellite === 'features') { // Draw feature for each node
                for (const [n, graph] of Object.entries(this.graphs))
                    graph.setSatellite('features')

                if (this.coloredNodes)
                    this.showClassAsColor(true)

            }
            else {
                let numClasses = this.datasetInfo["labelings"][this.labeling]
                if (satellite in this.datasetVar)
                    for (const [i, graph] of Object.entries(this.graphPrimitives))
                        if (graph.setSatellite(satellite, this.datasetVar[satellite][i], numClasses))
                            for (const e of graph.satellites[satellite].blocks)
                                $g.append(e)
            }
        }
    }

    // Add (new) explanation
    setExplanation(explanation) {
        if (this.explanation)
            this.dropExplanation()
        this.explanation = explanation

        // Replace explanation elements
        if (this.explanation == null) {
            for (const graph of Object.values(this.graphs))
                graph.dropExplanation()
        }
        else
            for (const [n, graph] of Object.entries(this.graphs))
                graph.setExplanation(this.explanation.reduce(n))
    }

    // getNodes() {
    // }
    //
    // getEdges() {
    // }

    // Get information HTML
    getInfo() {
        return '<b>Multiple graphs</b>'
    }

    // Change SVG elements positions according to layout positions and scale
    draw() {
        if (this.arrange)
            this.arrange.apply(this.graphGrabbed)
        for (const graph of Object.values(this.graphs)) {
            graph.draw(false)
        }
        // Update SVG elements according to layout and scale
        for (const [n, svgGraph] of Object.entries(this.graphPrimitives)) {
            let aGraph = this.graphs[n]
            // FIXME approxBBox() is called twice
            let bbox = aGraph.approxBBox()
            svgGraph.moveTo(bbox.x, bbox.y, bbox.width, bbox.height)
            svgGraph.scale(this.scale)
        }
        this.adjustVisibleArea() // do it once
    }

    createPrimitives() {
        // Clear all
        this.svgElement.innerHTML = ''

        // Graph-level elements
        this.graphPrimitives = {} // {graph ix -> graph}
        for (const [n, graph] of Object.entries(this.graphs))
            this.graphPrimitives[n] = new SvgGraph(0, 0, 1, 1, "#ffffff",
                `Graph ${n}`, true, this.svgPanel.$tip)

        // Edges for all graphs
        for (const graph of Object.values(this.graphs))
            graph.addEdgePrimitivesBatch(
                0, graph.edges, this.edgeColor, this.edgeStrokeWidth, this.datasetInfo.directed, true)

        let _addEvent = (element, node, graph) => {
            element.onmousedown = (e) => {
                this.nodeGrabbed = node
                this.graphGrabbed = graph
            }
            if (this.onNodeClick) {
                element.onclick = (e) => this.onNodeClick("left", node, graph)
                element.oncontextmenu = (e) => this.onNodeClick("right", node, graph)
                element.ondblclick = (e) => this.onNodeClick("double", node, graph)
            }
        }

        // Nodes for all graphs
        this.nodePrimitives = {} // empty
        for (let [graphIx, aGraph] of Object.entries(this.graphs)) {
            aGraph.nodePrimitives = {}
            graphIx = parseInt(graphIx)
            for (let n = 0; n < aGraph.numNodes; n++) {
                aGraph.createNodePrimitive(this.svgElement, n, this.nodeRadius, this.nodeStrokeWidth, this.nodeColor, true)
                _addEvent(aGraph.nodePrimitives[n].circle, n, graphIx)
            }
            // Object.assign(this.nodePrimitives, aGraph.nodePrimitives)
            // this.nodePrimitives.push(...Object.values(aGraph.nodePrimitives))
        }

        // Add <g> for explanation edges
        this.svgPanel.add("explanation-edges")
        // Explanation will be set after the layout

        // Add graph primitives for all graphs
        let g = this.svgPanel.add("graphs")
        for (const svgGraph of Object.values(this.graphPrimitives)) {
            g.appendChild(svgGraph.frame)
            g.appendChild(svgGraph.text)
        }

        // Add <g> for graphs satellites
        // Features for nodes are created within graphs
        for (const satellite of VisibleGraph.SATELLITES.slice(1))
            this.svgPanel.add("graphs-" + satellite)
        this.svgPanel.add("graphs-scores")
        super.createPrimitives()

        // Add nodes of visible graphs
        g = this.svgPanel.get("nodes")
        for (const aGraph of Object.values(this.graphs))
            for (const node of Object.values(aGraph.nodePrimitives)) {
                g.append(node.circle)
                g.append(node.text)
            }
    }

    // Compute approximate bounding box using nodes positions
    approxBBox() {
        let xMin = Infinity
        let yMin = Infinity
        let xMax = -Infinity
        let yMax = -Infinity
        for (const graph of Object.values(this.graphs))
            for (const vec of Object.values(graph.layout.pos)) {
                xMin = Math.min(xMin, vec.x)
                yMin = Math.min(yMin, vec.y)
                xMax = Math.max(xMax, vec.x)
                yMax = Math.max(yMax, vec.y)
            }
        xMin *= this.scale
        yMin *= this.scale
        xMax *= this.scale
        yMax *= this.scale
        let r = Math.max(MIN_NODE_RADIUS, Math.ceil(this.nodeRadius * (this.scale/100) ** 0.5)) + 0*this.pad
        return {
            x: xMin - 2.2 * r,
            y: yMin - 4.5 * r,
            width: xMax-xMin + 6 * r,
            height: yMax-yMin + 10 * r
        }
    }

    showClassAsColor(show) {
        // console.log('showClassAsColor', show)
        if (show) {
            function un1hot(array) {
                for (let i = 0; i < array.length; i++)
                    if (array[i] === 1) return i
                return -1
            }

            if (this.coloredNodes)
                for (const [g, graph] of Object.entries(this.graphs))
                    for (const [n, node] of Object.entries(graph.nodePrimitives))
                        node.setFillColorIdx(un1hot(this.datasetVar['features'][g][n]))
        }
        else {
            for (const graph of Object.values(this.graphs))
                for (const node of Object.values(graph.nodePrimitives))
                    node.dropFillColor()
        }
    }

    // Turn on/off visibility of labels, features, predictions, etc
    showSatellite(satellite, show) {
        console.log('multi.showSatellite', satellite, show)
        if (satellite === 'features') {
            for (const graph of Object.values(this.graphs))
                graph.showSatellite(satellite, show)
        }
        else {
            this.svgPanel.get("graphs-" + satellite).css("display", show ? 'inline' : 'none')
            for (const graph of Object.values(this.graphPrimitives)) {
                graph.satellites[satellite].show = show
                graph.visible(graph.show)
            }
        }
    }

    // Remove all explanation elements
    dropExplanation() {
        if (!this.explanation) return
        console.log('MultipleGraphs.dropExplanation')
        let $g = this.svgPanel.get("explanation-edges")
        $g.empty()
        if (this.explanation.nodes)
            for (const [g, nodes] of Object.entries(this.explanation.nodes)) {
                if (g in this.graphs)
                    for (const node of Object.keys(nodes))
                        this.graphs[g].nodePrimitives[node].dropColor()
            }
        this.explanation = null
        this.explanationEdges = null
    }
}
