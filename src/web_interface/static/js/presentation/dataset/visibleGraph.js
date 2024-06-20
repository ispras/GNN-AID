const IMPORTANCE_COLORMAP = 'cool'
const PREDICTION_COLORMAP = 'binary'
const EMBEDDING_COLORMAP = 'bwr'
const EDGE_MINIBATCH_SIZE = 10000 // Not necessary
const LIGHT_MODE_SCALE_THRESHOLD_SINGLE = 50 // When Scale < THR, light mode is on to improve visibility (for single graph)
const LIGHT_MODE_SCALE_THRESHOLD_MULTI = 20 // When Scale < THR, light mode is on to improve visibility (for multiple graph)
const EXPLANATION_EDGE_IMPORTANCE_THRESHOLD = 0.5 // When edge importance < THR, it is not drawn

function createColormapImage(element, colormap, width=120, height=25) {
    element.innerHTML = ''
    let canvas = document.createElement('canvas')
    // canvas.setAttribute('id', 'visible-graph-colormap')
    element.appendChild(canvas)
    let ctx = canvas.getContext("2d")
    canvas.setAttribute("width", width)
    canvas.setAttribute("height", height)
    const colormapImageData = ctx.createImageData(width, height)
    const data = colormapImageData.data;
    for (let i = 0; i < width; i++) {
        let [r, g, b] = evaluate_cmap(i / width, colormap, false)
        for (let j = 0; j < height; j++) {
            let ix = 4 * (i + j*width)
            data[ix] = r
            data[ix + 1] = g
            data[ix + 2] = b
            data[ix + 3] = 255 // alpha
        }
    }
    ctx.putImageData(colormapImageData, 0, 0);

    ctx.font = "16px serif"
    ctx.fillText("0", 0, 15)
    ctx.fillText("1", width-10, 15)
}

function createSetOfColors(numColors, $svg) {
    let $defs = $(document.createElementNS("http://www.w3.org/2000/svg", "defs"))

    $svg.prepend($defs)
    for (let i = 0; i < numColors; i++) {
        let $rg = $(document.createElementNS("http://www.w3.org/2000/svg", "radialGradient"))
        $rg.attr("id", "RadialGradient" + i)
        let color = valueToColor(i / numColors, "hsv")
        let $stop = $(document.createElementNS("http://www.w3.org/2000/svg", "stop"))
        $stop.attr("offset", "0%").attr("stop-color", color)
        $rg.append($stop)
        $stop = $(document.createElementNS("http://www.w3.org/2000/svg", "stop"))
        $stop.attr("offset", "100%").attr("stop-color", "white")
        $rg.append($stop)
        $defs.append($rg)
    }
    return true

    // let classColor = {}
    // for (let i = 0; i < numClasses; i++)
    //     classColor[i] = valueToColor(i / numClasses, "hsv")
    // return classColor
}

// Representation of a visible part of dataset - whole graph or a neighborhood.
// Responsible for drawing and interaction with user.
class VisibleGraph {
    static SATELLITES = ["features", "labels", "predictions", "embeddings", "train-test-mask"]

    constructor(datasetInfo, svgPanel) {
        this.datasetInfo = datasetInfo
        this.svgPanel = svgPanel
        this.svgElement = svgPanel.$svg[0]
        this.visView = controller.presenter.visualsView
        this._tag = "vg-" + timeBasedId()
        this._tagVar = "vg-var-" + timeBasedId()
        this.layoutFreezeButtonId = null // to be defined in subclass

        // Constants
        this.nodeRadius = 15
        this.nodeStrokeWidth = 2
        this.nodeExplainedStrokeWidth = 4
        this.edgeStrokeWidth = 1
        this.edgeExplainedStrokeWidth = 4
        this.pad = 30 // additional space around elements to SVG border
        this.scale = 100 // scale to adjust element
        this.scaleMax = 1e4 // maximal scale
        this.scaleMin = 1e0 // minimal scale
        this.zoomFactor = 1.15 // scaling coefficient on zoom
        this.screenPos = new Vec(-300, -400) // left-top of part of SVG visible on screen
        this.svgPos = new Vec(0, 0) // SVG viewBox left-top
        this.nodeColor = '#fff'
        this.edgeColor = '#000'

        // Variables
        this.datasetData = null // Nodes, edges, attributes, don't edit - it is changed from outside
        this.visibleConfig = {center: null, depth: null}
        this.datasetVar = null // Satellites, is set later, don't edit - it is changed from outside
        this.nodePrimitives = null // {node -> primitive} on HTML element
        this.edgePrimitivesBatches = null // {key -> list of SVGs} - path for a batch of edges on HTML element
        // External parameters - TODO must be assigned when switch view neighborhood/graph
        this.labeling = null // which labeling is chosen
        this.oneHotableFeature = null // whether feature is be 1-hot encoded
        this.coloredNodes = null // {class -> node fill color}
        this.explanation = null
        this.explanationEdges = null // {edge -> SVG primitive} for current explanation
        this.layout = null
        this.onNodeClick = null // callback when node is clicked
        this.beforeInit = null // callback when reinit() is called
        this.afterInit = null // callback when reinit() is called
        this.alive = null // needed to break draw cycle when destroying
        this.nodesVisible = true // flag whether to show nodes
        this.nodeGrabbed = null // node which is currently dragged (SVG element)
        this.mousePos = new Vec(0, 0) // mouse position in SVG pixels
        this.svgGrabbed = false // whether user grab svg element
        this.svgGrabbedMousePos = new Vec(0, 0) // mouse pos when SVG was grabbed
        this.svgGrabbedScreenPos = new Vec(0, 0) // screen pos when SVG was grabbed

    }

    async drawCycle() {
        while (true) {
            if (!this.alive) break
            if (this.layout && this.layout.moving) {
                this.draw()
            }
            await sleep(100)
        }
    }

    // Register user control elements listeners associated to dataset
    createListeners() {
        this.visView.addListener(this.layoutFreezeButtonId,
            (_, v) => this.freezeLayout(v), this._tag)
    }

    // Register user control elements listeners associated to dataset var
    createVarListeners() {
        for (const satellite of VisibleGraph.SATELLITES)
            this.visView.addListener(this.visView.satellitesIds[satellite],
                (_, v) => {
                    this.showSatellite(satellite, v)
                    this.draw()
                }, this._tagVar)
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
        window.onmouseup = (e) => {
            this.svgGrabbed = false
            this.nodeGrabbed = null
            this.svgElement.style.cursor = 'default'
            if (this.layout) {
                this.layout.release()
            }
        }
        this.svgElement.onmousemove = (e) => {
            this.mousePos.x = e.layerX
            this.mousePos.y = e.layerY
            if (this.svgGrabbed) {
                this.screenPos.x = this.svgGrabbedScreenPos.x - e.screenX + this.svgGrabbedMousePos.x
                this.screenPos.y = this.svgGrabbedScreenPos.y - e.screenY + this.svgGrabbedMousePos.y
                this.draw()
            }
            else if (this.nodeGrabbed != null) {
                this.layout.lock(this.nodeGrabbed, Vec.add(this.mousePos, this.svgPos).mul(1/this.scale))
                this.layout.startMoving()
                this.draw()
            }
            this.debugInfo()
        }

        // Handle zoom
        // TODO move to SvgPanel
        this.svgElement.onwheel = (e) => {
            if (e.ctrlKey) {
                e.preventDefault()
                let z = e.wheelDelta > 0 ? this.zoomFactor : 1/this.zoomFactor
                if (this.scale * z > this.scaleMax || this.scale * z < this.scaleMin)
                    return
                this.scale *= z

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

    // Initialize elements and start layout
    async init() {
        // Permanent part
        this.handleDragging()
        this.createListeners()
        $(this.svgElement).css("background-color", "#e7e7e7")

        await this._build()
    }

    // Stop showing visible graph and associated menu elements
    drop() {
        // this.dropVar()
        this.visView.removeListeners(this._tag)
        this.labeling = null
        this.coloredNodes = null
        this.onNodeClick = null

        this._drop()

        // *** finally goes some work at subclass
    }

    // Set visible config based on visual elements
    defineVisibleConfig() {
    }

    // Variable part of init - to be overridden
    async _build() {
        // *** some work at subclass (getting graph data), then:

        this.svgElement.parentElement.scrollLeft = 0
        this.svgElement.parentElement.scrollTop = 0
        this.createPrimitives()
        if (this.explanation)
            this.setExplanation(this.explanation)

        this.visView.fireEventsByTag(this._tag) // will call setLayout and others

        if (!this.alive) { // To avoid multiple draw cycles
            this.alive = true
            this.drawCycle()
        }
    }

    // Variable part of drop - to be overridden
    _drop() {
        this.nodePrimitives = null
        this.edgePrimitivesBatches = null
        if (this.layout)
            this.layout.stopMoving()
        this.layout = null
        this.alive = false
        this.svgElement.innerHTML = ''
    }

    initVar(datasetVar, labeling, oneHotableFeature) {
        this.datasetVar = datasetVar
        this.labeling = labeling
        this.oneHotableFeature = oneHotableFeature
        this.createVarListeners()

        this._buildVar()
    }

    // Remove all elements associated with dataset var data
    dropVar() {
        this.visView.removeListeners(this._tagVar)
        this._dropVar()
        this.datasetVar = null
    }

    // Variable part of initVar - to be overridden
    _buildVar() {
        for (const satellite of VisibleGraph.SATELLITES)
            this.setSatellite(satellite)

        this.visView.fireEventsByTag(this._tagVar)
    }

    // Remove all elements associated with dataset var data - to be overridden
    _dropVar() {
        if (this.datasetVar)
            for (const satellite of VisibleGraph.SATELLITES) {
                this.setSatellite(satellite, false)
                if (satellite in this.datasetVar)
                    delete this.datasetVar[satellite]
            }
        this.explanation = null
        this.explanationEdges = null
    }

    // Drops and builds with other parameters, handling Var part properly, keeping listeners
    async _reinit() {
        this._dropVar()
        this._drop()

        if (this.beforeInit)
            await this.beforeInit()

        await this._build()

        if (this.afterInit)
            await this.afterInit()

        if (this.datasetVar) {
            await this._buildVar()
        }
        this.checkLightMode()
    }

    // Check parameters to decide whether to turn on a light mode.
    // Value of 'visible' can be passed from a superclass.
    checkLightMode(visible=null) {
        if (visible == null)
            this.nodesVisible = this.scale >= LIGHT_MODE_SCALE_THRESHOLD_MULTI
        if (this.nodesVisible) {
            this.svgPanel.get("nodes").show()
            for (const satellite of VisibleGraph.SATELLITES) {
                this.svgPanel.get("nodes-" + satellite).show()
                this.showSatellite(satellite, this.visView.getValue(this.visView.satellitesIds[satellite]))
            }
        }
        else {
            this.svgPanel.get("nodes").hide()
            for (const satellite of VisibleGraph.SATELLITES)
                this.svgPanel.get("nodes-" + satellite).hide()
        }
    }

    // Set or Update graph layout
    setLayout() {
        // *** some work at subclass (setting Layout), then:
        this.layout.setVisibleGraph(this)
        this.visView.fireEvent(this.layoutFreezeButtonId)
    }

    freezeLayout(freeze) {
        this.layout.setFreeze(freeze)
    }

    // Get degree of a node of a graph
    getDegree(node, graph) {
        console.error('Not implemented generally')
    }

    // Get a set of all nodes
    getNodes() {
        console.error('Not implemented generally')
    }

    // Get a list of all edges
    getEdges() {
        console.error('Not implemented generally')
    }

    // Get information HTML
    getInfo() {
        return ''
    }

    // Get information about the node
    getNodeInfo(node, graph) {
        let nodeInfo = this.datasetData["node_info"]
        if (nodeInfo == null) return ''
        let res = {}
        if (this.datasetInfo.count > 1) {
            for (const [attr, info] of Object.entries(nodeInfo))
                if (graph in info)
                    res[attr] = info[graph][node]
        }
        else {
            for (const [attr, info] of Object.entries(nodeInfo))
                if (node in info)
                    res[attr] = info[node]
        }
        return JSON.stringify(res)
    }

    // Get a list of attributes values for a node (of a graph)
    getNodeAttrs(node, graph=0) {
        let nodeAttrs = this.datasetData["node_attributes"]
        if (nodeAttrs == null)
            return null
        let res = []
        for (const a of this.datasetInfo["node_attributes"]["names"]) {
            let attrs = nodeAttrs[a][graph]
            res.push(attrs[node])
        }
        return res
    }

    // Create HTML for SVG primitives on the given element
    createPrimitives() {
        // *** some work at subclass, then:

        // Add <g> for explanation edges
        this.svgPanel.add("explanation-edges")
        // Explanation will be set after the layout

        // Add node primitives
        let g = this.svgPanel.add("nodes")
        for (const node of Object.values(this.nodePrimitives)) {
            g.appendChild(node.circle)
            g.appendChild(node.text)
        }

        // Add satellite elements
        for (const satellite of VisibleGraph.SATELLITES)
            this.svgPanel.add("nodes-" + satellite)
    }

    // Create/remove SVG primitives for node satellites: labels, features, predictions, etc
    setSatellite(satellite, on=true) {
        // Replace satellite elements
        let $g = this.svgPanel.get("nodes-" + satellite)
        if (!on) {
            $g.empty()
            for (const node of Object.values(this.nodePrimitives))
                node.satellites[satellite].blocks = null
            if (satellite === 'labels')
                this.showClassAsColor(false)
        }
        else if (satellite in this.datasetVar) {
            let values = this.datasetVar[satellite]
            if (satellite === 'labels') {
                let numClasses = this.datasetInfo["labelings"][this.labeling]
                for (const [i, node] of Object.entries(this.nodePrimitives)) {
                    node.setLabels(values[i], numClasses)
                    for (const e of node.satellites[satellite].blocks)
                        $g.append(e)
                }

                if (numClasses <= 12) {
                    this.coloredNodes = createSetOfColors(numClasses, this.svgPanel.$svg)
                    this.visView.setEnabled(this.visView.singleClassAsColorId, true)
                    // this.showClassAsColor()
                }
                else {
                    this.visView.setValue(this.visView.singleClassAsColorId, false)
                    this.visView.setEnabled(this.visView.singleClassAsColorId, false)
                }
            }
            else {
                for (const [i, node] of Object.entries(this.nodePrimitives)) {
                    if (node.setSatellite(satellite, values[i]))
                        for (const e of node.satellites[satellite].blocks)
                            $g.append(e)
                }
            }
        }
    }

    // Add SVG path for a batch of edges
    addEdgePrimitivesBatch(key, batch, color, width, directed, show) {
        if (this.edgePrimitivesBatches == null)
            this.edgePrimitivesBatches = {}
        if (this.edgePrimitivesBatches[key] == null)
            this.edgePrimitivesBatches[key] = []
        let count = Math.ceil(batch.length / EDGE_MINIBATCH_SIZE)
        for (let i = 0; i < count; i++) {
            let miniBatch = batch.slice(i*EDGE_MINIBATCH_SIZE, (i+1)*EDGE_MINIBATCH_SIZE)
            let svg = new SvgEdgeBatch(miniBatch, color, width, directed, show)
            this.edgePrimitivesBatches[key].push(svg)
            this.svgElement.appendChild(svg.path)
        }
    }

    // Create a node SVG (will be added later together)
    createNodePrimitive(element, i, radius, width, color, show) {
        let node = new SvgNode(0, 0, radius, width, color, i.toString(), show, this.svgPanel.$tip)
        this.nodePrimitives[i] = node

        // Add listeners
        node.circle.onmousedown = (e) => this.nodeGrabbed = i
        if (this.onNodeClick) {
            node.circle.onclick = (e) => this.onNodeClick("left", i)
            node.circle.oncontextmenu = (e) => this.onNodeClick("right", i)
            node.circle.ondblclick = (e) => this.onNodeClick("double", i)
        }
    }

    debugInfo() {
        let html = ""
        html += `scale: ${this.scale.toPrecision(3)}`
        // html += `<br> svg pos: ${this.svgPos.str(5)}`
        // html += `<br> screen pos: ${this.screenPos.str(5)}`
        // html += `<br> mouse pos: ${this.mousePos.str()}`
        // html += `<br> layout pos: ${Vec.add(this.mousePos, this.svgPos).mul(1/this.scale).str(5)}`
        // html += `<br> viewBoxShift: ${this.svgPos.str(4)}`
        // html += `<br> viewBox: ${this.element.getAttribute("viewBox")}`
        // html += `<br> scroll: ${new Vec(this.element.parentElement.scrollLeft, this.element.parentElement.scrollTop).str(4)}`
        // $("#dataset-info-bottomleft").html(html)
        // $("#dataset-info-upright").html(html)
        controller.presenter.datasetView.$upRightInfoDiv.html(html)
    }

    // Compute approximate bounding box using nodes positions
    approxBBox() {
        let pos = this.layout.pos
        let xMin = Infinity
        let yMin = Infinity
        let xMax = -Infinity
        let yMax = -Infinity
        for (const vec of Object.values(pos)) {
            xMin = Math.min(xMin, vec.x)
            yMin = Math.min(yMin, vec.y)
            xMax = Math.max(xMax, vec.x)
            yMax = Math.max(yMax, vec.y)
        }
        xMin *= this.scale
        yMin *= this.scale
        xMax *= this.scale
        yMax *= this.scale
        let r = Math.max(MIN_NODE_RADIUS, Math.ceil(this.nodeRadius * (this.scale/100) ** 0.5))
        let padx = 2.1 * r
        let pady = 2.8 * r
        return {
            x: xMin - padx,
            y: yMin - pady,
            width: xMax-xMin + 2*padx,
            height: yMax-yMin + 2*pady
        }
    }

    // Adjust SVG viewBox, visible view and scroll according to elements positions on SVG
    adjustVisibleArea() {
        // TODO move this to SvgPanel
        let t = performance.now()
        let parent = this.svgElement.parentElement

        // Define new SVG viewBox - minimal rectangle covering SVG bbox and current visible screen
        // let bbox = this.element.getBBox()
        let bbox = this.approxBBox()
        // console.log(`Time of getBBox(): ${performance.now() - t}ms`)
        this.svgPos.x = Math.min(this.screenPos.x, bbox.x)
        this.svgPos.y = Math.min(this.screenPos.y, bbox.y)
        let svgX1 = Math.max(this.screenPos.x + parent.clientWidth, bbox.x + bbox.width)
        let svgY1 = Math.max(this.screenPos.y + parent.clientHeight, bbox.y + bbox.height)
        let w = svgX1 - this.svgPos.x
        let h = svgY1 - this.svgPos.y
        this.svgElement.setAttribute("viewBox", `${this.svgPos.x} ${this.svgPos.y} ${w} ${h}`)
        let dx = parseInt(this.svgElement.style.borderLeftWidth) + parseInt(this.svgElement.style.borderRightWidth) || 0
        let dy = parseInt(this.svgElement.style.borderBottomWidth) + parseInt(this.svgElement.style.borderTopWidth) || 0
        this.svgElement.style.width = `${w-dx}px`
        this.svgElement.style.height = `${h-4-dy}px` // TODO what is the magic number: 4px ? Check other browsers

        // Set scroll after SVG resize
        parent.scrollLeft = Math.max(0, this.screenPos.x - this.svgPos.x)
        parent.scrollTop = Math.max(0, this.screenPos.y - this.svgPos.y)
        // console.log(`Time of adjustVisibleArea(): ${performance.now() - t}ms`)
    }

    // Change SVG elements positions according to layout positions and scale
    draw(adjust=true) {
        console.log('draw')
        let t = performance.now()
        let pos = this.layout.pos
        // Update SVG elements according to layout and scale
        if (this.nodesVisible)
            for (const [n, node] of Object.entries(this.nodePrimitives)) {
                let newPos = Vec.mul(pos[n], this.scale)
                node.moveTo(newPos.x, newPos.y)
                node.scale(this.scale)
            }
        // console.log(`Time of node moving: ${performance.now() - t}ms`)
        let scaledPos = {}
        for (const [n, vec] of Object.entries(pos)) {
            scaledPos[n] = Vec.mul(vec, this.scale)
        }
        for (const batch of Object.values(this.edgePrimitivesBatches))
            for (const svg of batch) {
                svg.setScale(this.scale)
                svg.moveTo(scaledPos)
            }
        // console.log(`Time of node+edge moving: ${performance.now() - t}ms`)

        // Explanation edges
        if (this.explanation)
            for (const [edge, svg] of Object.entries(this.explanationEdges)) {
                let [i, j] = edge.split(',') // TODO simplify: parse at explanation init
                i = parseInt(i)
                j = parseInt(j)
                if (i in pos && j in pos) { // Only edges that can be visible
                    let pos1 = Vec.mul(pos[i], this.scale)
                    let pos2 = Vec.mul(pos[j], this.scale)
                    svg.moveTo(pos1.x, pos1.y, pos2.x, pos2.y)
                }
            }
        if (adjust)
            this.adjustVisibleArea()
        // console.log(`Time of draw(): ${performance.now() - t}ms`) // 310-360 for Cora (with attr and preds)
    }

    // Add (new) explanation
    setExplanation(explanation) {
        if (!explanation) return
        this.explanation = explanation

        console.log('VisibleGraph.drawExplanation')
        let $g = this.svgPanel.get("explanation-edges")
        // $g.empty()
        this.explanationEdges = {}
        if (this.explanation.edges)
            for (const [edge, value] of Object.entries(this.explanation.edges)) {
                // TODO create threshold from interface
                let thr = EXPLANATION_EDGE_IMPORTANCE_THRESHOLD
                if (value < thr) continue
                let color = valueToColor(value, this.explanation.colormap)
                // TODO how to determine width by edge ?
                let svg = new SvgEdge(0, 0, 0, 0, color, this.edgeExplainedStrokeWidth,
                    this.explanation.isDirected(),true)
                this.explanationEdges[edge] = svg
                $g.append(svg.path)
            }

        if (this.explanation.nodes)
            for (const [node, value] of Object.entries(this.explanation.nodes))
                if (node in this.nodePrimitives)
                    this.nodePrimitives[node].setColor(
                        valueToColor(value, this.explanation.colormap), this.nodeExplainedStrokeWidth)
        this.draw()
    }

    // Remove all explanation elements
    dropExplanation() {
        if (!this.explanation) return
        console.log('VisibleGraph.dropExplanation')
        // TODO re-use drawExplanation
        let $g = this.svgPanel.get("explanation-edges")
        $g.empty()
        if (this.explanation.nodes)
            for (const node of Object.keys(this.explanation.nodes))
                if (node in this.nodePrimitives)
                    this.nodePrimitives[node].dropColor()
        this.explanation = null
        this.explanationEdges = null
    }

    showClassAsColor(show) {// TODO unite in subclasses     dropClassAsColor()
        // console.log('showClassAsColor', show)
        if (show) {
            if (this.coloredNodes && this.datasetVar['labels'])
                for (const [n, node] of Object.entries(this.nodePrimitives))
                    node.setFillColorIdx(this.datasetVar['labels'][n])
        }
        else // drop
            for (const node of Object.values(this.nodePrimitives))
                node.dropFillColor()
    }

    // Turn on/off visibility of labels, features, predictions, etc
    showSatellite(satellite, show) {
        // console.log('showSatellite', satellite, show)
        this.svgPanel.get("nodes-" + satellite).css("display", show ? 'inline' : 'none')
        for (const node of Object.values(this.nodePrimitives)) {
            node.satellites[satellite].show = show
            node.visible(node.show)
        }
    }
}