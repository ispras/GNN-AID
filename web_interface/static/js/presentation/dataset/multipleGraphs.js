function un1hot(arr) {
    for (let i = 0; i < arr.length; i++)
        if (arr[i] === 1)
            return i
    return -1
}

class MultipleGraphs extends Graph {
    constructor(datasetInfo, svgPanel) {
        super(datasetInfo, svgPanel)

        this.layoutFreezeButtonId = this.visView.multiLayoutFreezeId

        // All-graphs storage (incoming format)
        this._graphs = []           // [graphId...]
        this._nodesByGraph = []     // [numNodes...]
        this._edgesByGraph = []     // [edgeList...]
        this._graphIdToIx = {}      // graphId -> ix

        // Active graph
        this._graph = null
        this._graphIx = null
        this._layout = null

        // Graph frame + satellites
        this.graphPrimitive = null

        // Optional feature-as-color (one-hot)
        this._showNodeTypeAsColor = false
        this._featureColors = null
    }

    /* -------------------------- listeners / visible config -------------------------- */

    defineVisibleConfig() {
        this.visibleConfig["center"] = parseInt(this.visView.getValue(this.visView.multiGraphId))
    }

    createListeners() {
        this.visView.addListener(
            this.visView.multiGraphId,
            async (_, v) => await this.setGraph(parseInt(v)),
            this._tag
        )

        this.visView.addListener(
            this.visView.multiLayoutId,
            (_, v) => this.setLayout(v),
            this._tag
        )

        this.visView._getById(this.visView.multiGraphId).attr("max", this.datasetInfo.count - 1)

        // IMPORTANT: bypass Graph.createListeners() (it registers singleGraph UI)
        VisibleGraph.prototype.createListeners.call(this)
    }

    createVarListeners() {
        // super.createVarListeners()
        VisibleGraph.prototype.createVarListeners.call(this)
        this.visView.addListener(
            this.visView.multiNodeTypeAsColorId,
            (_, v) => this.showNodeTypeAsColor(v),
            this._tagVar
        )
    }

    /* ---------------------------------- build/drop --------------------------------- */

    async _build() {
        // Cache datasetData (avoid refetch on setGraph())
        if (!this.datasetData) {
            this.datasetData = await controller.ajaxRequest('/dataset', { get: "data" })
        }

        // multipleGraphs: nodes/edges/graphs are arrays for a set of graphs
        this._nodesByGraph = this.datasetData.nodes
        this._edgesByGraph = this.datasetData.edges
        this._graphs = this.datasetData.graphs
        this.nodes = Object.fromEntries(this._graphs.map(g => [g, Array.from(Array(this._nodesByGraph[g]).keys())]))
        this.edges = this._edgesByGraph

        this._graphIdToIx = {}
        for (let i = 0; i < this._graphs.length; i++) this._graphIdToIx[this._graphs[i]] = i

        const fallbackGraphId = (this._graphs.length ? this._graphs[0] : 0)
        const requested = (this.visibleConfig["center"] !== null && this.visibleConfig["center"] !== undefined)
            ? this.visibleConfig["center"]
            : fallbackGraphId

        this._setActiveGraphInternal(requested)

        // Omit Graph._build()
        await VisibleGraph.prototype._build.call(this)

        $(this.svgElement).css("background-color", "#404040")
    }

    _convertDatasetVar(datasetVar) {
        if (!datasetVar) return null;
        const converted = {};

        const nodeKeys = this.nodes
        const edges = { [this._graphIx]: this.getEdges() }
        const edgeKeys = {}

        for (const [graphId, edgeList] of Object.entries(edges)) {
            edgeKeys[graphId] = edgeList.map(([i, j]) => `${i},${j}`)
        }

        for (const elem of VisibleGraph.ELEMS) {
          if (!datasetVar[elem]) continue;

          converted[elem] = {};
          for (const [field, dataArray] of Object.entries(datasetVar[elem])) {
            if (!dataArray || !Array.isArray(dataArray)) continue;

            if (elem === "graph") {
              const varDict = {};
              const graphIds = Object.keys(nodeKeys);

              graphIds.forEach((graphId, arrayIdx) => {
                if (arrayIdx < dataArray.length && dataArray[arrayIdx] !== null) {
                  varDict[graphId] = dataArray[arrayIdx];
                }
              });

              converted[elem][field] = varDict;
              continue;
            }

            if (elem === "edge") {
              const varDictDict = {};
              const graphIds = Object.keys(nodeKeys);

              graphIds.forEach((graphId, arrayIdx) => {
                const varDict = {};
                const graphEdgeKeys = edgeKeys[graphId] || [];

                if (arrayIdx < dataArray.length && Array.isArray(dataArray[arrayIdx])) {
                  graphEdgeKeys.forEach((key, edgeIdx) => {
                    if (edgeIdx < dataArray[arrayIdx].length && dataArray[arrayIdx][edgeIdx] !== null) {
                      varDict[key] = dataArray[arrayIdx][edgeIdx];
                    }
                  });
                }

                varDictDict[graphId] = varDict;
              });

              converted[elem][field] = varDictDict;
              continue;
            }

            const varDictDict = {};
            const graphIds = Object.keys(nodeKeys);

            graphIds.forEach((graphId, arrayIdx) => {
              const graphKeys = nodeKeys[graphId];
              const varDict = {};

              if (arrayIdx < dataArray.length && Array.isArray(dataArray[arrayIdx])) {
                graphKeys.forEach((key, nodeIdx) => {
                  if (nodeIdx < dataArray[arrayIdx].length && dataArray[arrayIdx][nodeIdx] !== null) {
                    varDict[key] = dataArray[arrayIdx][nodeIdx];
                  }
                });
              }

              varDictDict[graphId] = varDict;
            });

            converted[elem][field] = varDictDict;
          }

          if (Object.keys(converted[elem]).length === 0) delete converted[elem];
        }

        return converted;
    }

    _drop() {
        // IMPORTANT: bypass Graph._drop() (it resets background to light)
        VisibleGraph.prototype._drop.call(this)

        this.graphPrimitive = null
        // this.allDatasetVar = null
        this._layout = null
        // this._featureColorKey = null
        this._featureColors = null
    }

    _buildVar() {
        // Prepare one-hot feature palette if present
        // this._featureColorKey = null
        this._featureColors = null

        if (this.oneHotableFeature && this.datasetVar?.node?.features) {
            const keys = Object.keys(this.datasetVar.node.features)
            if (keys.length) {
                const key = keys[0]
                const first = this.datasetVar.node.features[key]?.[0]
                if (Array.isArray(first)) {
                    // this._featureColorKey = key
                    this._featureColors = createSetOfColors(first.length, this.svgPanel.$svg)
                }
            }
        }

        this.visView.setEnabled(this.visView.multiNodeTypeAsColorId, !!this._featureColors)

        super._buildVar()
    }

    /* --------------------------------- public actions -------------------------------- */

    getNodes() {
        return this.nodesList
    }

    getNumNodes() {
        return this.numNodes
    }

    getEdges() {
        return this.edgesList
    }

    async setGraph(graph) {
        if (graph == null) graph = parseInt(this.visView.getValue(this.visView.multiGraphId))
        if (graph === this._graph) return

        this.visibleConfig["center"] = graph
        this._graph = graph
        this._layout = null

        await this._reinit()
    }

    setLayout(layout) {
        if (layout == null) layout = this.visView.getValue(this.visView.multiLayoutId)
        if (layout === this._layout && this.layout) return

        if (this.layout) this.layout.stopMoving()

        switch (layout) {
            case "random":
                this.layout = new Layout()
                break
            case "force":
                this.layout = new ForceLayout()
                break
            case "forceAtlas2":
                this.layout = new ForceAtlas2Layout()
                this.layout.rad = 0.08
                break
            default:
                console.error('Unknown layout', layout)
                this.layout = new Layout()
        }

        this.layout.setVisibleGraph(this)
        this._layout = layout
        this.needsRedraw = true
    }

    showNodeTypeAsColor(show) {
        this._showNodeTypeAsColor = !!show
        this.needsRedraw = true
    }

    /* -------------------- primitives: add graph frame + satellites groups -------------------- */

    createPrimitives() {
        // this.svgElement.innerHTML = ''
        super.createPrimitives()

        const gGraph = this.svgPanel.add("graph")
        this.graphPrimitive = new SvgGraph(0, 0, 1, 1, "#ffffff",
            `Graph ${this._graph}`, true, this.svgPanel.$tip)
        gGraph.appendChild(this.graphPrimitive.g)
    }

    /* ----------------------------- satellites (graph-level) ----------------------------- */

    setSatellite(elem, satellite, on = true) {
        if (!this.ready) return

        if (elem !== 'graph') {
            this.showSatellite(elem, satellite, on)
            this.needsRedraw = true
            return
        }

        const $g = this.svgPanel.get("graph-" + satellite)

        if (!on) {
            $g.empty()
            if (this.graphPrimitive?.satellites?.[satellite])
                this.graphPrimitive.satellites[satellite].blocks = null
            return
        }

        $g.empty()
        if (!this.datasetVar?.graph?.[satellite]) return

        const numClasses = this.datasetInfo["labelings"][this.task][this.labeling]
        const ok = this.graphPrimitive.setSatellite(satellite, this.datasetVar.graph[satellite], numClasses)

        if (ok && this.graphPrimitive?.satellites?.[satellite]?.blocks) {
            for (const e of this.graphPrimitive.satellites[satellite].blocks) $g.append(e)
        }
    }

    /* ------------------------------------ draw ------------------------------------ */

    getVar(elem, satellite) {
        return this.datasetVar?.[elem]?.[satellite]?.[this._graphIx]
    }

    setNodeColor(node, n) {
        if (this._showNodeTypeAsColor && this._featureColors && this.datasetVar?.node?.features) {
            const values = this.getVar('node', 'features')?.[n]

            const cls = un1hot(values)
            if (cls >= 0)
                node.setFillColorIdx(cls)
        }
        else
            node.dropFillColor()
    }

    drawGraphs(drawSatellites) {
        if (!(this.graphPrimitive && this.layout?.pos))
            return
        const bbox = this._getWorldBBoxFromLayout(this.layout.pos)

        this.graphPrimitive.scale(this.scale) // SCALE APPLIED ON PRIMITIVE
        // this.graphPrimitive.moveTo(bbox.x, bbox.y, bbox.width, bbox.height) // WORLD
        this.graphPrimitive.moveTo(
            bbox.x * this.scale,
            bbox.y * this.scale,
            bbox.width * this.scale,
            bbox.height * this.scale
        )

        this.graphPrimitive.text.textContent = `Graph ${this._graph}`

        // Set var
        if (this.datasetVar) {
            // Set color
            // TODO

            // Set satellites
            if (!drawSatellites) {
                this.graphPrimitive.removeSatellites()
            }
            else
                for (const satellite of VisibleGraph.SATELLITES) {
                    let values = this.getVar('graph', satellite)
                    if (values) {
                        if (satellite === "labels") {
                            this.graphPrimitive.setLabels(values, this.numClasses)
                        } else
                            this.graphPrimitive.setSatellite(satellite, values)
                    }
                }
        }
        else {
            this.graphPrimitive.removeSatellites()
            // this.graphPrimitive.dropFillColor()
        }

    }

    setNodeExplanationColor(node, n) {
        if (this.explanation && this._graphIx in this.explanation.nodes && n in this.explanation.nodes[this._graphIx]) {
            const value = this.explanation.nodes[this._graphIx][n]
            node.setColor(valueToColor(value, this.explanation.colormap), this.nodeExplainedStrokeWidth)
        }
        else
            node.dropColor()
    }

    setEdgeExplanationColor(edge, eKey) {
        if (this.explanation && this._graphIx in this.explanation.edges && eKey in this.explanation.edges[this._graphIx]) {
            const value = this.explanation.edges[this._graphIx][eKey]
            if (value >= EXPLANATION_EDGE_IMPORTANCE_THRESHOLD) {
                let color = valueToColor(value, this.explanation.colormap)
                edge.setColor(color)
            }
        }
        else
            edge.dropColor()
    }

    // Add (new) explanation
    setExplanation(explanation) {
        if (!explanation) return
        this.explanation = explanation
        this.needsRedraw = true

        // Change keys to sorted(i,j) for undirected graphs
        if (this.explanation.edges && !this.datasetInfo.directed) {
            const edges = {}
            for (const [g, eEdges] of Object.entries(this.explanation.edges)) {
                edges[g] = {}
                for (const [key, value] of Object.entries(eEdges)) {
                    let [i, j] = key.split(',')
                    let sortedKey = `${Math.min(i, j)},${Math.max(i, j)}`
                    edges[g][sortedKey] = value
                }
            }
            this.explanation.edges = edges
        }
    }

    /* ---------------------------------- helpers ---------------------------------- */

    _setActiveGraphInternal(graphId) {
        const ix = (graphId in this._graphIdToIx) ? this._graphIdToIx[graphId] : 0
        this._graph = this._graphs.length ? this._graphs[ix] : graphId
        this._graphIx = ix

        this.visibleConfig["center"] = this._graph

        this.numNodes = this._nodesByGraph[ix] ?? 0
        // this.nodes = Array.from(Array(this.numNodes).keys())
        this.nodesList = this.nodes[this._graphIx]
        this.edgesList = this.edges[this._graphIx]
    }

    _getWorldBBoxFromLayout(pos) {
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
        for (const k in pos) {
            const v = pos[k]
            if (!v) continue
            if (v.x < minX) minX = v.x
            if (v.y < minY) minY = v.y
            if (v.x > maxX) maxX = v.x
            if (v.y > maxY) maxY = v.y
        }
        if (!isFinite(minX)) {
            minX = minY = -1
            maxX = maxY = 1
        }
        return { x: minX, y: minY, width: maxX - minX, height: maxY - minY }
    }
}