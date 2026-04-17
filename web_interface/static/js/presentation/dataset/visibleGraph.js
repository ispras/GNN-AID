const IMPORTANCE_COLORMAP = 'cool'
const PREDICTION_COLORMAP = 'binary'
const EMBEDDING_COLORMAP = 'bwr'
const CORRELATION_COLORMAP = 'bwr'
// const EDGE_MINIBATCH_SIZE = 10000 // Not necessary
const EXPLANATION_EDGE_IMPORTANCE_THRESHOLD = 0.5 // When edge importance < THR, it is not drawn

class Thresholds {
    static MAX_VISIBLE_SATELLITES_SCALE = 25 // When Scale < THR, satellite are hidden
    static MAX_VISIBLE_SATELLITES_COUNT = 200 // When count > THR, satellite are hidden
    static MAX_VISIBLE_NODES_SCALE = 5 // When Scale < THR, nodes are hidden
    static MAX_VISIBLE_NODES_COUNT = 1000 // When count > THR, nodes are hidden
    static MAX_VISIBLE_EDGES_COUNT = 3000 // Max number of edges to draw
}

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
    static SATELLITES = ["features", "labels", "predictions", "logits", "train-test-mask"]
    static ELEMS = ["node", "edge", "graph"]

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
        this.edgeRadius = 12
        this.nodeStrokeWidth = 2
        this.nodeExplainedStrokeWidth = 4
        this.edgeStrokeWidth = 0.6
        this.edgeExplainedStrokeWidth = 4
        this.pad = 30 // additional space around elements to SVG border
        this.scale = 100 // scale to adjust element
        this.scaleMax = 1e4 // maximal scale
        this.scaleMin = 1e-1 // minimal scale
        this.zoomFactor = 1.15 // scaling coefficient on zoom
        this.screenPos = new Vec(-300, -400) // left-top of part of SVG visible on screen
        this.svgPos = new Vec(0, 0) // SVG viewBox left-top in the world coordinates
        this.nodeColor = '#fff'
        this.edgeColor = '#000'
        this.backgroundColor = "#e7e7e7"

        // ----- Variables

        // Nodes, edges, attributes, don't edit - it is changed from outside
        this.visibleConfig = {center: null, depth: null}
        this.datasetData = null
        this.nodeAttributes = null // node -> dict of attributes
        this.edgeAttributes = null // edge -> dict of attributes

        // Satellites, is set later, don't edit - it is changed from outside
        this.datasetVar = null

        // {node -> primitive} on HTML element
        this.nodePrimitives = null

        // {key -> list of SVGs} - path for a batch of edges on HTML element.
        // Key is: depth for neighborhood, 0 for graph, graph ix for multi graph
        this.edgePrimitivesBatches = null

        // {key -> {edgeKey -> SvgEdge}} for individual edges.
        // Key is: depth for neighborhood, 0 for graph, graph ix for multi graph
        this.edgePrimitives = null

        this._showClassAsColor = null

        // ----- External parameters - TODO must be assigned when switch view neighborhood/graph

        this.task = null // which labeling is chosen
        this.labeling = null // which labeling is chosen
        this.oneHotableFeature = null // whether feature is be 1-hot encoded
        this.numClasses = null
        this.coloredNodes = null // {class -> node fill color}
        this.explanation = null
        this.layout = null
        this.onNodeClick = null // callback when node is clicked
        this.beforeInit = null // callback when reinit() is called
        this.afterInit = null // callback when reinit() is called
        this.ready = false // flag whether structures are ready to be updated, e.g. for setSatellites()
        this.alive = null // needed to break draw cycle when destroying
        this.nodesVisible = true // flag whether to show nodes
        this.nodeGrabbed = null // node which is currently dragged (SVG element)
        this.mousePos = new Vec(0, 0) // mouse position in SVG pixels
        this.svgGrabbed = false // whether user grab svg element
        this.svgGrabbedMousePos = new Vec(0, 0) // mouse pos when SVG was grabbed
        this.svgGrabbedScreenPos = new Vec(0, 0) // screen pos when SVG was grabbed

        // this.frameDt = -1
    }

    async drawCycle() {
        while (true) {
            if (!this.alive) break

            let t = 0
            let layoutUpdated = false
            if (this.layout)
                layoutUpdated = this.layout.consumePositionUpdates()

            let needsRedraw = layoutUpdated || this.needsRedraw

            if (needsRedraw) {
                t = performance.now()
                this.draw()
                this.needsRedraw = false
                t = performance.now() - t
                // console.log(`Time of draw cycle(): ${t}ms`)
            }
            await sleep(Math.max(0, 100 - t))
        }
    }

    // Register user control elements listeners associated to dataset
    createListeners() {
        this.visView.addListener(this.layoutFreezeButtonId,
            (_, v) => this.freezeLayout(v), this._tag)
    }

    // Register user control elements listeners associated to dataset var
    createVarListeners() {
        for (const elem of VisibleGraph.ELEMS)
            for (const satellite of VisibleGraph.SATELLITES)
                this.visView.addListener(this.visView.satellitesIds[satellite],
                    (_, v) => {
                        this.showSatellite(elem, satellite, v)
                        this.draw()
                    }, this._tagVar)
    }

    // Handle nodes and whole SVG drag&drop
    handleDragging() {
        this.svgElement.onmousedown = (e) => {
            // console.log(e.buttons)
            // if (e.buttons === 1 && e.ctrlKey) {
            if (e.buttons === 1 && this.nodeGrabbed === null) {
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
            if (e.buttons !== 1)
                return
            let rect = this.svgElement.getBoundingClientRect();
            this.mousePos.x = e.clientX - rect.left;
            this.mousePos.y = e.clientY - rect.top;
            if (this.svgGrabbed) {
                this.screenPos.x = this.svgGrabbedScreenPos.x - e.screenX + this.svgGrabbedMousePos.x
                this.screenPos.y = this.svgGrabbedScreenPos.y - e.screenY + this.svgGrabbedMousePos.y
                this.needsRedraw = true
            }
            else if (this.nodeGrabbed != null) {
                this.layout.lock(this.nodeGrabbed, Vec.add(this.mousePos, this.screenPos).mul(1/this.scale))
                this.layout.startMoving()
                this.needsRedraw = true
            }
            this._debugInfo()
        }

        // Handle zoom
        this.svgElement.onwheel = (e) => {
            // if (e.ctrlKey) {
            if (true) {
                e.preventDefault()
                const z = e.wheelDelta > 0 ? this.zoomFactor : 1 / this.zoomFactor
                if (this.scale * z > this.scaleMax || this.scale * z < this.scaleMin)
                    return

                const rect = this.svgElement.getBoundingClientRect()
                const mx = e.clientX - rect.left
                const my = e.clientY - rect.top

                this.screenPos.x = (this.screenPos.x + mx) * z - mx
                this.screenPos.y = (this.screenPos.y + my) * z - my
                this.scale *= z

                this.mousePos.x = mx
                this.mousePos.y = my
                this.needsRedraw = true
            }
            this._debugInfo()
        }

        // To avoid computing scroll from screenPos
        this.svgElement.parentElement.onscroll = null

        // Cache SVG size to avoid getting clientWidth at each draw
        this.svgParentSize = new Vec(
            this.svgElement.parentElement.clientWidth, this.svgElement.parentElement.clientHeight)

        this.svgParentSize = new Vec(
            this.svgElement.parentElement.clientWidth,
            this.svgElement.parentElement.clientHeight
        )

        const parent = this.svgElement.parentElement

        // Фиксируем размер родителя через CSS чтобы он не зависел от содержимого
        parent.style.overflow = 'hidden'
        parent.style.position = 'relative' // нужен для minimap

        // Читаем размер один раз
        this.svgParentSize = new Vec(parent.clientWidth, parent.clientHeight)

        // Если нужно реагировать на resize окна или svg
        window.addEventListener('resize', () => {
            // console.log('window - resize')
            this.svgParentSize.x = parent.clientWidth
            this.svgParentSize.y = parent.clientHeight
            this.needsRedraw = true
        })
        let parentResize = () => {
            // console.log('parent - resize')
            this.svgParentSize.x = parent.clientWidth
            this.svgParentSize.y = parent.clientHeight
            this.needsRedraw = true
        }
        parentResize()
        new ResizeObserver(parentResize).observe(parent)
    }

    // Convert node,edge,graph attributes
    _convertDatasetData() {
        this.nodeAttributes = {}
        const attrNames = this.datasetInfo["node_attributes"]["names"]
        let n = 0
        for (const nKey of this.getNodes()) {
            this.nodeAttributes[nKey] = {}
            const attrs = this.datasetData['node_attributes'][n]
            if (attrs)
                for (let i = 0; i < attrNames.length; ++i) {
                    let a = attrNames[i]
                    this.nodeAttributes[nKey][a] = attrs[i]
                }
            ++n
        }
    }

    // Initialize elements and start layout
    async init() {
        // Permanent part
        this.handleDragging()
        this.createListeners()
        // $(this.svgElement).css("background-color", "#e7e7e7")

        await this._build()
    }

    // Stop showing visible graph and associated menu elements
    drop() {
        // this.dropVar()
        this.visView.removeListeners(this._tag)
        this.task = null
        this.labeling = null
        this.coloredNodes = null
        this.onNodeClick = null
        this.nodeAttributes = null
        this.edgeAttributes = null

        this._drop()

        // *** finally goes some work at subclass
    }

    // Set visible config based on visual elements
    defineVisibleConfig() {
    }

    // Variable part of init - to be overridden
    async _build() {
        // *** some work at subclass (getting graph data), then:

        // Convert datasetData
        this._convertDatasetData()

        this.svgElement.parentElement.scrollLeft = 0
        this.svgElement.parentElement.scrollTop  = 0
        this.svgElement.style.width  = '100%'
        this.svgElement.style.height = '100%'

        this.createPrimitives()       // создаёт пул DOM-элементов
        this._initMinimap()           // добавляет canvas

        if (this.explanation)
            this.setExplanation(this.explanation)

        this.visView.fireEventsByTag(this._tag)

        if (!this.alive) {
            this.alive = true
            this.drawCycle()
        }
        $(this.svgElement).css("background-color", this.backgroundColor)
    }

    // Variable part of drop - to be overridden
    _drop() {
        this.nodePrimitives = null
        this.edgePrimitives = null
        this.edgePrimitivesBatches = null
        if (this.layout)
            this.layout.stopMoving()
        this.layout = null
        this.alive = false
        this.svgElement.innerHTML = ''
        this.svgPanel.$tip.hide()
    }

    // Convert new datasetVar format (arrays) to old format (dicts) for compatibility

    _convertDatasetVar(datasetVar) {
        console.error('Not implemented generally')
        return

        if (!datasetVar) return null;
        const converted = {};

        // Get node keys from datasetData
        let nodeKeys

        // Check if nodeKeys is a dict (multiple graphs) or simple list (single graph)
        // const isMultiGraph = !Array.isArray(nodeKeys);
        const isMultiGraph = this.datasetInfo.count > 1

        // Get edge keys from datasetData
        let edgeKeys;
        if (isMultiGraph) {
            nodeKeys = this.nodes
            // Multiple graphs: edges is a dict {graphId -> list of edges}
            const edges = {[this._graphIx]: this.getEdges()}
            edgeKeys = {}
            for (const [graphId, edgeList] of Object.entries(edges)) {
                edgeKeys[graphId] = edgeList.map(([i, j]) => `${i},${j}`)
            }
        } else {
            // Single graph: edges is a simple list
            nodeKeys = this.getNodes()
            const edges = this.getEdges()
            edgeKeys = edges ? edges.map(([i, j]) => `${i},${j}`) : []
        }

        // Convert variables, omit nulls
        for (const elem of VisibleGraph.ELEMS) {
            if (datasetVar[elem]) {
                converted[elem] = {}
                for (const [field, dataArray] of Object.entries(datasetVar[elem])) {
                    if (dataArray && Array.isArray(dataArray)) {
                        if (elem === 'graph') {
                            // For graph-level variables
                            if (isMultiGraph) {
                                // Multiple graphs: create dict (graphId -> value)
                                const varDict = {};
                                const graphIds = Object.keys(nodeKeys);
                                graphIds.forEach((graphId, arrayIdx) => {
                                    if (arrayIdx < dataArray.length && dataArray[arrayIdx] !== null) {
                                        varDict[graphId] = dataArray[arrayIdx];
                                    }
                                });
                                converted[elem][field] = varDict;
                            } else {
                                // Single graph: just keep as array
                                converted[elem][field] = dataArray;
                            }
                        } else if (elem === 'edge') {
                            // For edge-level variables
                            if (isMultiGraph) {
                                // Multiple graphs: create dict of dicts (graphId -> edgeKey -> value)
                                const varDictDict = {};
                                const graphIds = Object.keys(nodeKeys);
                                graphIds.forEach((graphId, arrayIdx) => {
                                    const varDict = {};
                                    const graphEdgeKeys = edgeKeys[graphId] || [];
                                    if (arrayIdx < dataArray.length && Array.isArray(dataArray[arrayIdx])) {
                                        graphEdgeKeys.forEach((key, edgeIdx) => {
                                            if (edgeIdx < dataArray[arrayIdx].length &&
                                                dataArray[arrayIdx][edgeIdx] !== null) {
                                                varDict[key] = dataArray[arrayIdx][edgeIdx];
                                            }
                                        });
                                    }
                                    varDictDict[graphId] = varDict;
                                });
                                converted[elem][field] = varDictDict;
                            } else {
                                // Single graph: create simple dict (edgeKey -> value)
                                const actualArray = (dataArray.length === 1 && Array.isArray(dataArray[0])) ? dataArray[0] : dataArray;
                                const varDict = {};
                                edgeKeys.forEach((key, index) => {
                                    if (index < actualArray.length && actualArray[index] !== null) {
                                        varDict[key] = actualArray[index];
                                    }
                                });
                                converted[elem][field] = varDict;
                            }
                        } else if (isMultiGraph) {
                            // Multiple graphs: create dict of dicts (graphId -> nodeId -> value)
                            const varDictDict = {};
                            const graphIds = Object.keys(nodeKeys);
                            graphIds.forEach((graphId, arrayIdx) => {
                                const graphKeys = nodeKeys[graphId];
                                const varDict = {};
                                if (arrayIdx < dataArray.length && Array.isArray(dataArray[arrayIdx])) {
                                    graphKeys.forEach((key, nodeIdx) => {
                                        if (nodeIdx < dataArray[arrayIdx].length &&
                                            dataArray[arrayIdx][nodeIdx] !== null) {
                                            varDict[key] = dataArray[arrayIdx][nodeIdx];
                                        }
                                    });
                                }
                                varDictDict[graphId] = varDict;
                            });
                            converted[elem][field] = varDictDict;
                        } else {
                            // Single graph: dataArray might be nested [[[...]]] or flat [[...]]
                            const actualArray = (dataArray.length === 1 && Array.isArray(dataArray[0])) ? dataArray[0] : dataArray;
                            const varDict = {};
                            nodeKeys.forEach((key, index) => {
                                if (index < actualArray.length && actualArray[index] !== null) {
                                    varDict[key] = actualArray[index];
                                }
                            });
                            converted[elem][field] = varDict;
                        }
                    }
                }
            }
        }

        return converted;
    }

    initVar(datasetVar, task, labeling, oneHotableFeature) {
        this.datasetVar = this._convertDatasetVar(datasetVar)
        this.task = task
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
        this.needsRedraw = true
    }

    // Variable part of initVar - to be overridden
    _buildVar() {
        this.ready = true
        this.numClasses = this.datasetInfo["labelings"][this.task][this.labeling]
        if (this.numClasses <= 12)
            this.coloredNodes = createSetOfColors(this.numClasses, this.svgPanel.$svg)

        this.visView.fireEventsByTag(this._tagVar)
        this.needsRedraw = true
    }

    // Remove all elements associated with dataset var data - to be overridden
    _dropVar() {
        this.explanation = null
    }

    // Drops and builds with other parameters, handling Var part properly, keeping listeners
    async _reinit() {
        this.ready = false
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
    }

    // Set or Update graph layout
    setLayout() {
        // *** some work at subclass (setting Layout), then:
        this.layout.setVisibleGraph(this)
        this.visView.fireEvent(this.layoutFreezeButtonId)
        this._buildMinimapCache()
    }

    freezeLayout(freeze) {
        this.layout.setFreeze(freeze)
    }

    // Get degree of a node of a graph - for layout
    getDegree(node, graph) {
        console.error('Not implemented generally')
    }

    // Get a list of all nodes - for layout
    getNodes() {
        console.error('Not implemented generally')
    }

    // Get a total number of nodes - for layout
    getNumNodes() {
        console.error('Not implemented generally')
    }

    // Get a list of all edges - for layout
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
        let nodeAttrs = this.nodeAttributes[node]
        return nodeAttrs ?? ''
    }

    // Create HTML for SVG primitives on the given element
    createPrimitives() {
        let gEdge = this.svgPanel.add("edge")
        this.edgePrimitives = {0:{}}
        for (let n=0; n<Thresholds.MAX_VISIBLE_EDGES_COUNT; ++n) {
            let edge = new SvgEdge(0, 0, 0, 0, this.edgeRadius, '#fff',
                this.edgeStrokeWidth, this.datasetInfo.directed, true, this.svgPanel.$tip)
            this.edgePrimitives[0][n] = edge
            gEdge.appendChild(edge.g)
        }

        // Add max slots for nodes
        this.nodePrimitives = {}
        let gNode = this.svgPanel.add("node")
        for (let n=0; n<Thresholds.MAX_VISIBLE_NODES_COUNT; ++n) {
            let node = new SvgNode(0, 0, this.nodeRadius, 'circle', this.nodeStrokeWidth,
                '#fff', '', true, this.svgPanel.$tip)
            this.nodePrimitives[n] = node
            gNode.appendChild(node.g)
        }

        this.visNodes = new Set()
        this.visEdges = []
    }

    _debugInfo() {
        let html = ""
        html += `scale: ${this.scale.toPrecision(3)}`
        html += `<br> svg pos: ${this.svgPos.str(5)}`
        html += `<br> screen pos: ${this.screenPos.str(5)}`
        html += `<br> mouse pos: ${this.mousePos.str(5)}`
        html += `<br> layout pos: ${Vec.add(this.mousePos, this.svgPos).mul(1/this.scale).str(5)}`

        // Drawing times
        // {key -> {edgeKey -> SvgEdge}} for individual edges.
        // Key is: depth for neighborhood, 0 for graph, graph ix for multi graph
        let numEdgePrimitives = 0
        if (this.edgePrimitives) {
            for (const es of Object.values(this.edgePrimitives))
                numEdgePrimitives += Object.keys(es).length
        }

        // let last = 0;
        // const self = this
        // let rafTick = (t) => {
        //     if (last) {
        //         this.frameDt = t - last;
        //         // dt ~ 16.7ms на 60Hz, ~8.3ms на 120Hz, больше = просадки
        //         console.log("frame dt:", this.frameDt.toFixed(2), "ms");
        //     }
        //     last = t;
        //     requestAnimationFrame(rafTick);
        // }
        // requestAnimationFrame(rafTick);

        html += `<br> draw nodes time [${this.visNodes.size}]: ${Debug.DRAW_NODES_TIME}`
        html += `<br> draw edges time [${this.visEdges.length}]: ${Debug.DRAW_EDGES_TIME}`
        html += `<br> adjustVisibleArea time: ${Debug.ADJUST_VIS_AREA_TIME}`
        html += `<br> draw minimap time: ${Debug.DRAW_MINIMAP_TIME}`
        html += `<br> draw total time: ${Debug.DRAW_TOTAL_TIME}`
        // html += `<br> frame dt: ${this.frameDt.toFixed(2)}`
        html += `<br> layout step time: ${Debug.LAYOUT_STEP_TIME}`

        // html += `<br> viewBoxShift: ${this.svgPos.str(4)}`
        // html += `<br> viewBox: ${this.element.getAttribute("viewBox")}`
        // html += `<br> scroll: ${new Vec(this.element.parentElement.scrollLeft, this.element.parentElement.scrollTop).str(4)}`
        // $("#dataset-info-bottomleft").html(html)
        $("#dataset-info-upright").html(html)
        // controller.presenter.datasetView.$upRightInfoDiv.html(html)
    }

    // Adjust SVG viewBox
    _adjustVisibleArea() {
        const t = performance.now()
        this.svgElement.setAttribute('viewBox',
            `${this.screenPos.x} ${this.screenPos.y} ${this.svgParentSize.x} ${this.svgParentSize.y}`)

        Debug.ADJUST_VIS_AREA_TIME = performance.now() - t
    }

    // Minimap setup (вызвать один раз в init)
    _initMinimap() {
        // Создаём canvas поверх SVG-контейнера
        const wrap = this.svgElement.parentElement
        const mc = document.createElement('canvas')
        mc.style.cssText = `
            position:absolute; bottom:12px; right:12px;
            width:160px; height:100px; border-radius:8px;
            background:rgba(15,17,23,0.85);
            border:1px solid rgba(100,120,255,0.25);
            pointer-events:none; z-index:10;
        `
        // parentElement должен быть position:relative
        wrap.style.position = 'relative'
        wrap.appendChild(mc)
        this._minimapCanvas = mc
        this._minimapCtx   = mc.getContext('2d')

        // Предвычисляем фиксированную выборку и bounding box
        this._minimapSample = null
        this._minimapBBox   = null
    }

    _buildMinimapCache() {
        const pos     = this.layout.pos
        const allKeys = Object.keys(pos)
        if (!allKeys.length) return

        // Bounding box по всем точкам + находим крайние ключи
        let xMin = Infinity, yMin = Infinity, xMax = -Infinity, yMax = -Infinity
        let kXMin, kYMin, kXMax, kYMax
        for (const k of allKeys) {
            const v = pos[k]
            if (v.x < xMin) { xMin = v.x; kXMin = k }
            if (v.x > xMax) { xMax = v.x; kXMax = k }
            if (v.y < yMin) { yMin = v.y; kYMin = k }
            if (v.y > yMax) { yMax = v.y; kYMax = k }
        }
        this._minimapBBox = { xMin, yMin, xMax, yMax }

        const SAMPLE = 500
        if (allKeys.length <= SAMPLE) {
            // Все точки
            this._minimapSample = allKeys
        } else {
            // Случайная выборка с фиксированным сидом
            let seed = 42
            const rand = () => {
                seed = (seed * 1664525 + 1013904223) & 0xffffffff
                return (seed >>> 0) / 0xffffffff
            }
            const sampled = Array.from(
                {length: SAMPLE - 4},
                () => allKeys[Math.floor(rand() * allKeys.length)]
            )
            // Добавляем крайние точки гарантированно
            this._minimapSample = [...new Set([...sampled, kXMin, kXMax, kYMin, kYMax])]
        }
    }

    _drawMinimap() {
        const mc  = this._minimapCanvas
        const ctx = this._minimapCtx
        if (!mc || !ctx || !this.layout || !this._minimapSample) return

        let t = performance.now()
        const MW = mc.offsetWidth  || 160
        const MH = mc.offsetHeight || 100
        mc.width  = MW
        mc.height = MH

        const pos = this.layout.pos

        // Bbox считаем по всем точкам сэмпла каждый раз —
        // layout мог сдвинуться после _buildMinimapCache
        let xMin = Infinity, yMin = Infinity, xMax = -Infinity, yMax = -Infinity
        for (const k of this._minimapSample) {
            const v = pos[k]; if (!v) continue
            if (v.x < xMin) xMin = v.x; if (v.x > xMax) xMax = v.x
            if (v.y < yMin) yMin = v.y; if (v.y > yMax) yMax = v.y
        }
        // Крайние точки из кэша — гарантируют что bbox не уже реального графа
        if (this._minimapBBox) {
            const b = this._minimapBBox
            if (b.xMin < xMin) xMin = b.xMin
            if (b.yMin < yMin) yMin = b.yMin
            if (b.xMax > xMax) xMax = b.xMax
            if (b.yMax > yMax) yMax = b.yMax
        }

        const ww = xMax - xMin || 1
        const wh = yMax - yMin || 1

        const PAD    = 4
        const availW = MW - PAD * 2
        const availH = MH - PAD * 2
        const scale  = Math.min(availW / ww, availH / wh)

        const offX = PAD + (availW - ww * scale) / 2
        const offY = PAD + (availH - wh * scale) / 2

        const toMX = (wx) => (wx - xMin) * scale + offX
        const toMY = (wy) => (wy - yMin) * scale + offY

        ctx.fillStyle = '#0f1117'
        ctx.fillRect(0, 0, MW, MH)

        ctx.fillStyle = '#6070ff'
        for (const k of this._minimapSample) {
            const v = pos[k]; if (!v) continue
            ctx.beginPath()
            ctx.arc(toMX(v.x), toMY(v.y), 1.5, 0, Math.PI * 2)
            ctx.fill()
        }

        // Viewport
        const vx = toMX(this.screenPos.x / this.scale)
        const vy = toMY(this.screenPos.y / this.scale)
        const vw = (this.svgParentSize.x / this.scale) * scale
        const vh = (this.svgParentSize.y / this.scale) * scale
        ctx.strokeStyle = 'rgba(160,170,255,0.85)'
        ctx.lineWidth = 1
        ctx.strokeRect(vx, vy, vw, vh)

        Debug.DRAW_MINIMAP_TIME = performance.now() - t
    }

    // Set attributes specific to this node
    setNodeAttributes(node, n) {
        // *** to be defined in subclasses
    }

    // Set attributes specific to this edge
    setEdgeAttributes(edge, i, j) {
        // *** to be defined in subclasses
    }

    getVar(elem, satellite) {
        return this.datasetVar[elem][satellite]
    }

    setNodeColor(node, n) {
        if (this._showClassAsColor && this.numClasses && this.numClasses <= 12)
            node.setFillColorIdx(this.getVar('node', 'labels')?.[n])
        else
            node.dropFillColor()
    }

    drawNode(node, n, drawSatellites) {
        // Set attributes specific to this node
        this.setNodeAttributes(node, parseInt(n))

        // Позиция через transform — один setAttribute вместо cx/cy
        const pos = this.layout.pos
        node.translate(pos[n].x * this.scale, pos[n].y * this.scale)
        node.scale(this.scale)

        // Текст — номер узла
        if (node.text.textContent !== n)
            node.text.textContent = n

        // Set explanation color
        this.setNodeExplanationColor(node, n)

        // Set var
        if (this.datasetVar) {
            // Set color
            this.setNodeColor(node, n)

            // Set satellites
            if (!drawSatellites) {
                node.removeSatellites()
                node.text.textContent = ''
            }
            else
                for (const satellite of VisibleGraph.SATELLITES) {
                    let values = this.getVar('node', satellite)
                    if (values) {
                        if (satellite === "labels") {
                            node.setLabels(values[n], this.numClasses)
                        } else
                            node.setSatellite(satellite, values[n])
                    }
                }
        }
        else {
            node.removeSatellites()
            node.dropFillColor()
        }

        // Drag listener: передаём реальный ключ узла
        n = parseInt(n)
        node.body.onmousedown = (e) => this.nodeGrabbed = n
        if (this.onNodeClick) {
            node.body.onclick = (e) => this.onNodeClick("left", n)
            node.body.oncontextmenu = (e) => this.onNodeClick("right", n)
            node.body.ondblclick = (e) => this.onNodeClick("double", n)
        }
        let tip = this.datasetData?.node_info?.[n]
        if (tip)
            node.addTip(JSON_stringify(tip, 1))
        // node.addTip(JSON_stringify(this.getNodeAttrs(n), 1))
    }

    setNodeExplanationColor(node, n) {
        if (this.explanation && n in this.explanation.nodes) {
            const value = this.explanation.nodes[n]
            node.setColor(valueToColor(value, this.explanation.colormap), this.nodeExplainedStrokeWidth)
        }
        else
            node.dropColor()
    }

    setEdgeExplanationColor(edge, eKey) {
        if (this.explanation && eKey in this.explanation.edges) {
            const value = this.explanation.edges[eKey]
            if (value >= EXPLANATION_EDGE_IMPORTANCE_THRESHOLD) {
                let color = valueToColor(value, this.explanation.colormap)
                edge.setColor(color)
            }
        }
        else
            edge.dropColor()
    }

    drawEdge(edge, i, j, drawSatellites) {
        const pos = this.layout.pos
        const x1e = pos[i].x * this.scale, y1e = pos[i].y * this.scale
        const x2e = pos[j].x * this.scale, y2e = pos[j].y * this.scale

        edge.scale(this.scale)
        edge.moveTo(x1e, y1e, x2e, y2e) // fixme only when node is moving
        // if (this.layout && this.layout.moving)
            // edge.moveTo(pos[i].x, pos[i].y, pos[j].x, pos[j].y) // fixme only when node is moving

        // Set attributes specific to this node
        this.setEdgeAttributes(edge, i, j)

        let eKey = `${i},${j}`

        // Set explanation color
        this.setEdgeExplanationColor(edge, eKey)

        // Set satellites
        if (this.datasetVar)
            for (const satellite of VisibleGraph.SATELLITES) {
                if (!drawSatellites) {
                    edge.removeSatellites()
                    continue
                }
                let values = this.datasetVar?.edge?.satellite
                if (values) {
                    if (satellite === "labels") {
                        edge.setLabels(values[eKey], this.numClasses)
                    }
                    else
                        edge.setSatellite(satellite, values[eKey])
                }
            }
    }

    drawGraphs() {
    }

    // Change SVG elements positions according to layout positions and scale
    draw(adjust = true) {
        let t = performance.now()

        const pos = this.layout.pos
        const scale = this.scale

        // ── 1. Viewport в мировых координатах
        const x0 = this.screenPos.x / scale
        const y0 = this.screenPos.y / scale
        const x1 = (this.screenPos.x + this.svgParentSize.x) / scale
        const y1 = (this.screenPos.y + this.svgParentSize.y) / scale

        // ── 2. Culling узлов
        const visNodeKeys = []
        for (const n in pos) {
            const v = pos[n]
            if (v.x >= x0 && v.x <= x1 && v.y >= y0 && v.y <= y1)
                visNodeKeys.push(n)
        }

        // ── 3. Culling рёбер (хотя бы один конец виден)
        const visNodeSet = new Set(visNodeKeys.map(Number))
        let visEdges   = []
        let e = 0
        const allEdges = this.getEdges() // fixme keep this ?
        for (const [i, j] of allEdges) {
            if (visNodeSet.has(i) || visNodeSet.has(j))
                visEdges.push(e)
            ++e
            // if (visEdges.length >= Thresholds.MAX_VISIBLE_NODES_COUNT) break
        }

        // Прореживание ребер
        if (visEdges.length > Thresholds.MAX_VISIBLE_EDGES_COUNT) {
            let newArray = []
            let r = visEdges.length / Thresholds.MAX_VISIBLE_EDGES_COUNT
            for (let i=0; i<Thresholds.MAX_VISIBLE_EDGES_COUNT; ++i) {
                newArray.push(visEdges[parseInt(i*r)])
            }
            visEdges = newArray
        }

        // ── 4. Обновляем пул узлов
        let drawSatellites = (visNodeKeys.length <= Thresholds.MAX_VISIBLE_SATELLITES_COUNT) &&
            (this.scale > Thresholds.MAX_VISIBLE_SATELLITES_SCALE)

        let drawNodes = (visNodeKeys.length <= Thresholds.MAX_VISIBLE_NODES_COUNT) &&
            (this.scale > Thresholds.MAX_VISIBLE_NODES_SCALE)
        console.log('drawNodes', drawNodes)

        if (!drawNodes) {
            // Hide all nodes
            this.svgPanel.get("node").hide()
            this.visNodes = new Set()
        }
        else {
            let gNode = this.svgPanel.get("node")
            gNode.show()

            for (let n = 0; n < Thresholds.MAX_VISIBLE_NODES_COUNT; n++) {
                const node = this.nodePrimitives[n]

                if (n >= visNodeKeys.length) {
                    // Скрываем неиспользуемые слоты
                    node.visible(false)
                    continue
                }
                node.visible(true)

                this.drawNode(node, visNodeKeys[n], drawSatellites)
            }
        }
        Debug.DRAW_NODES_TIME = performance.now() - t

        // ── 5. Обновляем пул рёбер
        for (let ei = 0; ei < Thresholds.MAX_VISIBLE_EDGES_COUNT; ei++) {
            const edge = this.edgePrimitives[0][ei]  // TODO

            if (ei >= visEdges.length) {
                edge.visible(false)
                continue
            }

            const e = visEdges[ei]
            const [i, j] = allEdges[e]
            if (!(i in pos) || !(j in pos)) {
                edge.visible(false)
                continue
            }
            edge.visible(true)

            this.drawEdge(edge, i, j, drawSatellites)
        }

        Debug.DRAW_EDGES_TIME = performance.now() - t - Debug.DRAW_NODES_TIME

        // ── 6. Draw graphs frames and satellites
        this.drawGraphs(drawSatellites)

        // ── 7. Minimap
        this._drawMinimap()

        // ── 8. adjustVisibleArea
        if (adjust)
            this._adjustVisibleArea()

        this.visEdges  = visEdges
        this.visNodes  = visNodeSet
        Debug.DRAW_TOTAL_TIME = performance.now() - t
        this._debugInfo()
    }

    // Add (new) explanation
    setExplanation(explanation) {
        if (!explanation) return
        console.log('VisibleGraph.drawExplanation')
        this.explanation = explanation
        this.needsRedraw = true

        // Change keys to sorted(i,j) for undirected graphs
        if (this.explanation.edges && !this.datasetInfo.directed) {
            const edges = {}
            for (const [key, value] of Object.entries(this.explanation.edges)) {
                let [i, j] = key.split(',')
                let sortedKey = `${Math.min(i,j)},${Math.max(i,j)}`
                edges[sortedKey] = value
            }
            this.explanation.edges = edges
        }
    }

    // Remove all explanation elements
    dropExplanation() {
        if (!this.explanation) return
        console.log('VisibleGraph.dropExplanation')
        this.explanation = null
        this.needsRedraw = true
    }

    showClassAsColor(show) {
        this._showClassAsColor = show
        this.needsRedraw = true
    }

    // Turn on/off visibility of labels, features, predictions, etc
    showSatellite(elem, satellite, show) {
        this.svgPanel.getSatellite(elem, satellite).css("display", show ? 'inline' : 'none')
    }
}