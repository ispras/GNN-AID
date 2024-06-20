class Explanation {
    constructor() {
        this.info = null
        this.data = null
    }

    reduce(graph) {
        // TODO
    }

    // Explanation information to show on the panel
    addInfo($panel) {
    }
}

class StringExplanation extends Explanation {
    constructor(explanationData) {
        super()
        this.info = explanationData["info"]
        this.data = explanationData["data"]
    }

    addInfo($panel, params) {
        let html = ''
        html += '<br>Type: ' + this.info["type"]
        html += '<br>Locality: ' + (this.info["local"] ? 'local' : 'global')
        html += '<br>' + '<pre>' + JSON_stringify(this.data, 1).replaceAll('\n', '<br>') + '</pre>'
        $panel.append(html)
    }
}

class PrototypeExplanation extends Explanation {
    constructor(explanationData) {
        super()
        this.info = explanationData["info"]
        this.data = explanationData["data"]

        this.numClasses = explanationData["info"]["meta"]["num_classes"]
        this.protPerClass = explanationData["info"]["meta"]["num_prot_per_class"]
        this.baseGraphs = explanationData["data"]["base_graphs"]
        this.nodes = explanationData["data"]["nodes"]

        this.colormap = IMPORTANCE_COLORMAP
    }

    async addInfo($panel, params) {
        let html = ''
        html += '<br>Type: ' + this.info["type"]
        html += '<br>Locality: ' + (this.info["local"] ? 'local' : 'global')

        let meta = this.info["meta"]
        if (meta) {
            html += `<br>Num classes: ${this.numClasses}`
            html += `<br>Prototypes per class: ${this.protPerClass}`
        }
        $panel.append(html)

        // SVG with prototype graphs
        let $div = $("<div></div>").css("overflow", "scroll")
        $panel.append($div)
        let svgPanel = new SvgPanel($div[0], 300, 400)
        svgPanel.$svg.css("border", "1px solid #22ee22")

        // Checkboxes
        let id = "explanation-checkbox-showfeatures"
        let $checkboxDiv = $("<div></div>").attr("class", "control-block")
        $panel.append($checkboxDiv)
        let $showFeaturesCheckbox = $("<input>").attr("id", id)
            .attr("type", "checkbox").prop("checked", false)
        $checkboxDiv.append($showFeaturesCheckbox)
        $checkboxDiv.append($("<label></label>").text("Show features").attr("for", id))

        // Extract graphs for prototypes
        let datasetView = controller.presenter.datasetView
        this.prototypeGraphs = new ExplanationGraphs(
            datasetView.datasetInfo, svgPanel, this.protPerClass, this.numClasses)
        this.prototypeGraphs.scale = 30
        this.prototypeGraphs.satelliteShowInputs = {
            'features': $showFeaturesCheckbox,
        }

        // Set datasetData and datasetVar
        let datasetData = await Controller.ajaxRequest('/dataset', {
            get: "data", part: JSON_stringify({center: this.baseGraphs})})
        datasetData.graphs = Array.from(Array(this.baseGraphs.length).keys())
        this.prototypeGraphs.datasetData = datasetData
        this.prototypeGraphs.baseGraphs = this.baseGraphs
        await this.prototypeGraphs.init(this.nodes)

        // Ask for model satellites: masks, preds and embeds
        let datasetVar = {}
        let data = await Controller.ajaxRequest('/model', {
            get: "satellites", part: JSON_stringify({center: this.baseGraphs})})
        if (data !== '')
            for (const satellite of VisibleGraph.SATELLITES)
                if (satellite in data)
                    datasetVar[satellite] = data[satellite]

        data = await Controller.ajaxRequest('/dataset', {
            get: "var_data", part: JSON_stringify({center: this.baseGraphs})})
        if (data !== '')
            for (const satellite of VisibleGraph.SATELLITES)
                if (satellite in data)
                    datasetVar[satellite] = data[satellite]

        // Maps datasetVar mapped version, also prevent from changing
        // Expect data as {key -> {ix-to-be-mapped -> value}}
        let remap = (data) => {
            let dataMapped = {}
            for (let ix = 0; ix < this.baseGraphs.length; ix++) {
                let g = this.baseGraphs[ix]
                for (const [key, val] of Object.entries(data)) {
                    if (!(key in dataMapped))
                        dataMapped[key] = {}
                    dataMapped[key][ix] = val[g]
                }
            }
            return dataMapped
        }

        // NOTE: after remapping, datasetVar is not editable from outside
        this.prototypeGraphs.initVar(remap(datasetVar),
            datasetView.labeling, datasetView.oneHotableFeature)

        let protScores = Array.from(Array(this.baseGraphs.length), () => [])
        for (const cs of this.data["class_connection"])
            for (let i = 0; i < cs.length; i++)
                protScores[i].push(cs[i])
        this.prototypeGraphs.datasetVar['scores'] = protScores
        this.prototypeGraphs.setSatellite('scores', true)
        this.prototypeGraphs.showSatellite('labels', true)
        this.prototypeGraphs.showSatellite('scores', true)
        this.prototypeGraphs.showClassAsColor(true)
    }
}

class SubgraphExplanation extends Explanation {
    constructor(explanationData) {
        super()
        this.info = explanationData["info"]
        this.data = explanationData["data"]

        this.nodes = explanationData["data"]["nodes"]
        this.edges = explanationData["data"]["edges"]
        this.features = explanationData["data"]["features"]

        this.colormap = IMPORTANCE_COLORMAP
    }

    isDirected() {
        return "directed" in this.info ? this.info["directed"] : true
    }

    // Get an explanation for a particular graph
    reduce(graphIx) {
        let copy = new SubgraphExplanation({
            info: this.info,
            data: this.data,
            nodes: null,
            edges: null,
            features: this.features,
        })
        if (this.nodes)
            copy.nodes = Object.assign({}, this.nodes[graphIx])
        if (this.edges)
            copy.edges = Object.assign({}, this.edges[graphIx])
        return copy
    }

    addInfo($panel, params) {
        let html = ''
        html += '<br>Type: ' + this.info["type"]
        html += '<br>Locality: ' + (this.info["local"] ? 'local' : 'global')

        let meta = this.info["meta"]
        if (meta) {
            html += `<br> Properties:<br> - nodes: ${meta["nodes"]}`
            html += `<br> - edges: ${meta["edges"]}`
            html += `<br> - features: ${meta["features"]}`
        }
        $panel.append(html)

        // Show interactive table
        let [num_feats, multi] = params
        if (this.data) this.createTables($panel, num_feats, multi)
    }

    // Represent explanation data as table
    createTables($panel, numFeatures, multi) {
        let $div = $("<div></div>").addClass('explanation-table')
        $panel.append($div)

        let nodes = this.data["nodes"]
        if (nodes) {
            $div.append($("<label></label>").html("<h3>Nodes importance</h3>"))
            let $table = $("<table></table>")
            $table.addClass("display")
            let $thead = $("<thead></thead>")
            $table.append($thead)
            let $tr = $("<tr></tr>")
            $thead.append($tr)
            if (multi)
                $tr.append($("<th>Graph</th>"))
            $tr.append($("<th>Node</th>"))
            $tr.append($("<th>Importance</th>"))
            let $tbody = $("<tbody></tbody>")
            $table.append($tbody)
            if (multi)
                for (const [g, gNodes] of Object.entries(nodes))
                    for (const [n, val] of Object.entries(gNodes)) {
                        let $tr = $("<tr></tr>")
                        $tbody.append($tr)
                        $tr.append($("<td></td>").text(g))
                        $tr.append($("<td></td>").text(n))
                        $tr.append($("<td></td>").text(val))
                    }
            else
                for (const [n, val] of Object.entries(nodes)) {
                    let $tr = $("<tr></tr>")
                    $tbody.append($tr)
                    $tr.append($("<td></td>").text(n))
                    $tr.append($("<td></td>").text(val))
                }
            $div.append($table)
            $table.DataTable({
                scrollY: 300,
                paging: false,
                searching: false,
                scrollCollapse: true,
                order: multi ? [[0, "asc"], [1, "asc"], [2, "desc"]] : [[1, "desc"]],
                info: false,
            })
        }

        let edges = this.data["edges"]
        if (edges) {
            $div.append($("<label></label>").html("<h3>Edges importance</h3>"))
            let $table = $("<table></table>")
            let $thead = $("<thead></thead>")
            $table.append($thead)
            let $tr = $("<tr></tr>")
            $thead.append($tr)
            if (multi)
                $tr.append($("<th>Graph</th>"))
            $tr.append($("<th>Edge</th>"))
            $tr.append($("<th>Importance</th>"))
            let $tbody = $("<tbody></tbody>")
            $table.append($tbody)
            if (multi)
                for (const [g, gEdges] of Object.entries(edges))
                    for (const [e, val] of Object.entries(gEdges)) {
                        let $tr = $("<tr></tr>")
                        $tbody.append($tr)
                        $tr.append($("<td></td>").text(g))
                        $tr.append($("<td></td>").text(e))
                        $tr.append($("<td></td>").text(val))
                    }
            else
                for (const [e, val] of Object.entries(edges)) {
                    let $tr = $("<tr></tr>")
                    $tbody.append($tr)
                    $tr.append($("<td></td>").text(e))
                    $tr.append($("<td></td>").text(val))
                }
            $div.append($table)
            $table.DataTable({
                scrollY: 300,
                paging: false,
                searching: false,
                order: multi ? [[0, "asc"], [2, "desc"]] : [[1, "desc"]],
                info: false,
            })
        }

        let features = this.data["features"]
        if (features) {
            $div.append($("<label></label>").html("<h3>Features importance</h3>"))

            let $tip = $("<span></span>").addClass("tooltip-text")
            $panel.append($tip)

            // Adds mouse listener for all elements which shows a tip with given text
            let _addTip = (element, text) => {
                element.onmousemove = (e) => {
                    $tip.show()
                    $tip.css("left", e.pageX + 10)
                    $tip.css("top", e.pageY + 15)
                    $tip.html(text)
                }
                element.onmouseout = (e) => {
                    $tip.hide()
                }
            }

            // Checkbox switcher "as color table"
            let id = "explanation-features-switch"
            let $switchDiv = $("<div></div>")
                .css("display", "flex")
                .css("margin-bottom", "8px")
            $div.append($switchDiv)
            let $switch = $("<input>").attr("id", id)
                .attr("type", "checkbox").prop("checked", false)
            $switchDiv.append($switch)
            $switchDiv.append($("<label></label>").text("as color table").attr("for", id))

            // Table body
            let $table = $("<table></table>")
            let $thead = $("<thead></thead>")
            $table.append($thead)
            let $tr = $("<tr></tr>")
            $thead.append($tr)
            $tr.append($("<th>Feature</th>"))
            $tr.append($("<th>Importance</th>"))
            let $tbody = $("<tbody></tbody>")
            $table.append($tbody)
            for (const [f, val] of Object.entries(features)) {
                let $tr = $("<tr></tr>")
                $tbody.append($tr)
                $tr.append($("<td></td>").text(f))
                $tr.append($("<td></td>").text(val))
            }
            $div.append($table)
            $table.DataTable({
                scrollY: 300,
                paging: false,
                searching: false,
                order: [[1, "desc"]],
                info: false,
            })

            // SVG with features
            let width = Math.min(30, Math.floor(numFeatures**0.5))
            let height = Math.ceil(numFeatures / width)
            let size = Math.min(30, Math.floor(300 / width))
            let svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
            let $svg = $(svg)
                .css("background-color", "#e7e7e7")
                .css("flex-shrink", "0")
                .css("display", "none")
                .css("width", (width * size) + "px")
                .css("height", (height * size) + "px")
            $div.append($svg)
            let ix = 0
            for (let j = 0; j < height; j++) {
                for (let i = 0; i < width; i++) {
                    let rect = document.createElementNS("http://www.w3.org/2000/svg", "rect")
                    rect.setAttribute('x', size * i)
                    rect.setAttribute('y', size * j)
                    rect.setAttribute('width', size)
                    rect.setAttribute('height', size)
                    let color = "#c4c4c4"
                    if (ix in features) {
                        color = valueToColor(features[ix], this.colormap)
                        _addTip(rect, ix + ': ' + features[ix])
                    }
                    rect.setAttribute('fill', color)
                    rect.setAttribute('stroke', '#e7e7e7')
                    rect.setAttribute('stroke-width', 1)
                    $svg.append(rect)
                    ix++
                    if (ix === numFeatures) break
                }
            }

            // Checkbox switcher listener
            $table = $table.parent().parent().parent()
            $switch.change(() => {
                if ($svg.css('display') === 'none') {
                    $table.hide()
                    $svg.show()
                }
                else {
                    $table.show()
                    $svg.hide()
                }
            })
        }
    }
}