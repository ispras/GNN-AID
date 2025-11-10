class ExplanationGraphs extends MultipleGraphs {
    constructor(datasetInfo, svgPanel, prots, classes) {
        console.assert(datasetInfo.count > 0)
        super(datasetInfo, svgPanel)

        this.baseGraphs = null // Graphs numbers
        this.satelliteShowInputs = {}
        this.prots = prots // rows
        this.classes = classes // columns
    }

    handleDragging () {
        // TODO add zoom possibility later
        // // Handle zoom
        // this.element.onwheel = (e) => {
        //     if (e.ctrlKey) {
        //         e.preventDefault()
        //         let z = e.wheelDelta > 0 ? this.zoomFactor : 1/this.zoomFactor
        //         if (this.scale * z > this.scaleMax || this.scale * z < this.scaleMin)
        //             return
        //         this.scale *= z
        //         for (const graph of Object.values(this.graphs))
        //             graph.scale = this.scale
        //
        //         // Compute SVG and screen new positions
        //         this.screenPos.x += (z-1)*(this.mousePos.x + this.svgPos.x)
        //         this.screenPos.y += (z-1)*(this.mousePos.y + this.svgPos.y)
        //
        //         this.checkLightMode()
        //         this.draw()
        //         // Update mouse pos AFTER draw
        //         this.mousePos.x = e.layerX
        //         this.mousePos.y = e.layerY
        //     }
        // }

        // // To avoid computing scroll from screenPos
        // this.element.parentElement.onscroll = (e) => {
        //     this.screenPos.x = this.element.parentElement.scrollLeft + this.svgPos.x
        //     this.screenPos.y = this.element.parentElement.scrollTop + this.svgPos.y
        // }
    }

    createListeners() {
        // Create listeners for layout and arrange
        this.visView.addListener(this.visView.multiLayoutId,
            (_, v) => this.setLayout(v), this._tag)
        this.visView.addListener(this.visView.multiArrangeId,
            (_, v) => this.setArrange(v), this._tag)
    }

    createVarListeners() {
        for (let [satellite, $showInput] of Object.entries(this.satelliteShowInputs)) {
            $showInput.off()
            $showInput.change(() => this.showSatellite(satellite, $showInput.is(':checked')))
            $showInput.change()
        }
    }

    async init(nodes) {
        await super.init()

        // Induce subgraphs
        let _nodes = {}
        let _edges = {}
        for (const [g, graph] of Object.entries(this.graphs)) {
            _nodes[g] = Object.fromEntries(nodes[g].map(n => [n, 1]))
            let edgeList = graph.getInducedSubgraph(new Set(nodes[g]))
            _edges[g] = Object.fromEntries(edgeList.map(e => [e, 1]))
        }
        let explanationData = {
            info: {
                type: "subgraph",
                directed: false,
            },
            data: {
                nodes: _nodes,
                edges: _edges,
            }
        }
        console.log(explanationData)
        this.setExplanation(new SubgraphExplanation(explanationData))
    }

    async _build() {
        await super._build()

        // Change graphs' display names
        for (const [graphIx, graph] of Object.entries(this.graphPrimitives)) {
            let p = graphIx % this.prots
            let c = Math.floor(graphIx / this.prots)
            // graph.text.textContent = 'Prot ' + p + ' Class ' + c + ' Graph ' + this.graphsChosen[graphIx]
            graph.text.textContent = 'p' + p + ' c' + c + ' g' + this.baseGraphs[graphIx]
        }
    }

    setLayout () {
        super.setLayout("force")
    }

    setArrange() {
        super.setArrange("grid")
    }

    // Turn on/off visibility of labels, features, predictions, etc
    showSatellite(satellite, show) {
        super.showSatellite(satellite, show)
        this.draw()
    }
}