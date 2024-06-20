// Main Graph view
class DatasetView extends View {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)
        this.visView = controller.presenter.visualsView
        this._tag = "dv-" + timeBasedId()

        // main SVG
        let $svgDiv = $("<div></div>").attr("id", "dataset-svg")
            .attr("style", "top: 0; bottom: 0; left: 0; right: 0")
            .css("position", "absolute")
            .css("overflow", "scroll")
        this.$div.append($svgDiv)
        this.svgPanel = new SvgPanel($svgDiv[0])

        // Info panels
        this.$upLeftInfoDiv = $("<div></div>").attr("id", "dataset-info-upleft")
            .attr("style", "position: fixed; background: #eeeeeeb0; padding: 5px; pointer-events: none;")
        this.$div.append(this.$upLeftInfoDiv)
        this.$bottomLeftInfoDiv = $("<div></div>").attr("id", "dataset-info-bottomleft")
            .attr("style", "position: absolute; bottom: 1px; left: 2px; background: #eeeeeeb0; padding: 5px; pointer-events: none;")
        this.$div.append(this.$bottomLeftInfoDiv)
        this.$upRightInfoDiv = $("<div></div>").attr("id", "dataset-info-upright")
            .attr("style", "position: absolute; top: 1px; right: 2px; background: #eeeeeeb0; padding: 5px; pointer-events: none;")
        this.$div.append(this.$upRightInfoDiv)
        this.$bottomRightInfoDiv = $("<div></div>").attr("id", "dataset-info-bottomright")
            .attr("style", "position: absolute; bottom: 1px; right: 2px; background: #eeeeeeb0; padding: 5px; pointer-events: none;")
        this.$div.append(this.$bottomRightInfoDiv)

        // Draw colormap for attributes
        createColormapImage(this.$bottomLeftInfoDiv[0], IMPORTANCE_COLORMAP)

        this._downloadButton(this.$bottomRightInfoDiv)

        this.visView.addListener(this.visView.showModeId,
            async (_, v) => await this.setDataset(v), this._tag)

        // Variables
        this.datasetInfo = null
        this.datasetVar = null
        this.explanation = null
        this.visibleGraph = null
        this.labeling = null
        this.oneHotableFeature = null
    }

    async onInit(block, data) {
        await super.onInit(block, data)
        if (block === "dvc") {
            this.datasetInfo = data
            if (controller.presenter.menuDatasetView.state === MVState.LOCKED) {
                // this.dataset = new Dataset(datasetData)
                // console.log(this.dataset)
                this.visView.fireEvent(this.visView.showModeId) // will call setDataset
            }
        }
    }

    async onSubmit(block, data) {
        super.onSubmit(block, data)
        if (block === "dc") {
            await this.init()
        }
        else if (block === "dvc") {
            [this.labeling, this.oneHotableFeature] = data
            this.datasetVar = await Controller.ajaxRequest('/dataset', {get: "var_data"})
            this.setDatasetVar()
        }
        else if (block === "el") {
            if ("explanation_data" in data) {
                this.setExplanation(data["explanation_data"])
            }
        }
    }

    onUnlock(block) {
        super.onUnlock(block)
        if (block === "dc") {
            this.dropDatasetVar()
            this.dropDataset()
            this.datasetInfo = null
        }
        else if (block === "dvc") {
            this.dropDatasetVar()
            this.datasetVar = null
            this.labeling = null
            this.oneHotableFeature = null
        }
        else if (block === "mmc") {
            if (this.datasetVar) {
                // TODO other model vars
                this.datasetVar['train-test-mask'] = null
                this.visibleGraph.setSatellite('train-test-mask', false)
            }
        }
        else if (block === "el" || block === "ei") {
            if (this.visibleGraph)
                this.visibleGraph.dropExplanation()
            this.explanation = null
        }
    }

    onReceive(block, data) {
        // super.onReceive(block, args)
        if (block === "mmc" || block === "mt") {
            for (const satellite of VisibleGraph.SATELLITES) {
                if (satellite in data) {
                    this.datasetVar[satellite] = data[satellite]
                    this.visibleGraph.setSatellite(satellite)
                }
            }
        }
        else if (block === "er") {
            if ("explanation_data" in data) {
                this.setExplanation(data["explanation_data"])
            }
        }
    }

    // Update Node info
    setNodeInfo(node, graph) {
        // TODO extend to graph
        let html = ''
        if (this.visibleGraph) {
            html += this.visibleGraph.getInfo() + '<br>'
        }
        if (node != null) {
            html += "<b>Selected</b>"
            if (graph != null)
                html += ` graph: ${graph}, `
            html += ` node: ${node}`
            html += ' ' + this.visibleGraph.getNodeInfo(node, graph)
            // html += `<br>Degree: ${this.visibleGraph.getDegree(node, graph)}`
            let attrString = JSON.stringify(this.visibleGraph.getNodeAttrs(node, graph))
            if (attrString.length > 120)
                attrString = attrString.slice(0, 120) + '...'
            html += `<br>Attributes: ${attrString}`
            // html += `<br>pos: ${this.visibleGraph.layout.pos[node].str(5)}`
            // if (this.embeddings)
            //     html += `<br>embedding: ${this.embeddings[node]}`
        }
        this.$upLeftInfoDiv.html(html)
    }

    async onNodeClick(event, node, graph) {
        if (event === "left") {
            this.setNodeInfo(node, graph)
        }
        else if (event === "double") {
            if (this.visibleGraph instanceof Neighborhood) {
                await this.visibleGraph.setNode(node)
            }
        }
        else if (event === "right") {
            // TODO show popup-menu
        }
    }

    // Create visible graph based on current dataset and show parameter
    async setDataset(showMode) {
        blockLeftMenu(true)
        this.dropDatasetVar()
        this.dropDataset()

        if (this.datasetInfo.count > 1) {
            this.visibleGraph = new MultipleGraphs(this.datasetInfo, this.svgPanel)
        }
        else {
            // Choose depending on showMode
            switch (showMode) {
                case "neighborhood":
                    this.visibleGraph = new Neighborhood(this.datasetInfo, this.svgPanel)
                    break
                case "whole-graph":
                    this.visibleGraph = new Graph(this.datasetInfo, this.svgPanel)
            }
        }
        this.visibleGraph.beforeInit = this.beforeInit.bind(this)
        this.visibleGraph.afterInit = this.afterInit.bind(this)
        this.visibleGraph.onNodeClick = this.onNodeClick.bind(this)

        this.visibleGraph.defineVisibleConfig()

        await this.beforeInit()
        await this.visibleGraph.init()
        await this.afterInit()

        blockLeftMenu(false)

        if (this.datasetVar)
            this.setDatasetVar()
    }

    dropDataset() {
        if (this.visibleGraph)
            this.visibleGraph.drop()
        this.visibleGraph = null
    }

    setDatasetVar() {
        // console.log('setDatasetVar()')
        this.visibleGraph.initVar(this.datasetVar, this.labeling, this.oneHotableFeature)
    }

    dropDatasetVar() {
        if (this.visibleGraph)
            this.visibleGraph.dropVar()
    }

    setExplanation(explanationData) {
        // fixme duplicates expl from datasetView
        let explanation
        switch (explanationData["info"]["type"]) {
            case "subgraph":
                explanation = new SubgraphExplanation(explanationData)
                break
            case "prototype":
                explanation = new PrototypeExplanation(explanationData)
                break
            case "string":
                explanation = new StringExplanation(explanationData)
                break
            default:
                console.error("Unknown explanation type")
        }
        this.explanation = explanation
        this.visibleGraph.setExplanation(this.explanation)
    }

    async beforeInit() {
        await Controller.ajaxRequest('/dataset',
            {set: "visible_part", part: JSON_stringify(this.visibleGraph.visibleConfig)})

        let data = await Controller.ajaxRequest('/dataset', {get: "data"})
        this.visibleGraph.datasetData = data
    }

    async afterInit() {
        // Ask for features and labels
        let data = await Controller.ajaxRequest('/dataset', {get: "var_data"})
        if (data !== '') {
            for (const satellite of VisibleGraph.SATELLITES) {
                if (satellite in data) {
                    this.datasetVar[satellite] = data[satellite]
                }
            }
        }

        // Ask for model satellites: masks, preds and embeds
        data = await Controller.ajaxRequest('/model', {get: "satellites"})
            for (const satellite of VisibleGraph.SATELLITES) {
                if (data !== '')
                    if (satellite in data) {
                        this.datasetVar[satellite] = data[satellite]
                    }
            }

        if (this.explanation)
            this.visibleGraph.setExplanation(this.explanation)

        this.visibleGraph.checkLightMode()
        this.visibleGraph.draw()
        this.setNodeInfo()
    }

    _downloadButton($div) {
        $div.empty()
        let $button = $("<button></button>").append($("<img>")
            .attr("src", "../static/icons/download-svgrepo-com.svg")
            .attr("height", "20px"))
        $div.append($button)
        $button.css("pointer-events", "auto")
        $button.click(() => {
            let svgData = this.svgPanel.svg.outerHTML
            let svgBlob = new Blob([svgData], {type:"image/svg+xml;charset=utf-8"})
            let svgUrl = URL.createObjectURL(svgBlob)
            let downloadLink = document.createElement("a")
            downloadLink.href = svgUrl
            downloadLink.download = "dataset.svg"
            document.body.appendChild(downloadLink)
            downloadLink.click()
            document.body.removeChild(downloadLink)
        })
    }
}
