class PanelDatasetView extends PanelView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)
        this.$infoDiv = null
        this.$statsDiv = null
        this.$varStatsDiv = null

        this.init()

        // Variables
        this.datasetInfo = null
        this.labeling = null
    }

    init() {
        super.init("Dataset panel")
        this.$div.css("background", '#d1e8ff')
    }

    async onInit(block, args) {
        super.onInit(block, args)
        if (block === "dvc") {
            this.datasetInfo = args
            this.update()
        }
    }

    onUnlock(block) {
        super.onUnlock(block)
        if (block === "dvc") {
            this.labeling = null
            this.$varStatsDiv.empty()
        }
    }

    onSubmit(block, data) {
        // No super call
        if (block === "dvc") {
            this.labeling = data[0]
            this.updateVar()
        }
    }

    /// Called at each submit
    addNumericStat($whereDiv, name, stat, fracFlag) {
        let $div = $("<div></div>")
        $whereDiv.append($div)
        let $button = $("<button></button>").text("get")
        $div.append(name + ': ')
        $div.append($button)
        $button.click(async () => {
            $button.prop("disabled", true)
            let res = await Controller.ajaxRequest('/dataset', {get: "stat", stat: stat})
            // console.log(res)
            $div.empty()
            if (res.constructor === Object) { // dict
                let str = ""
                for (let [k, v] of Object.entries(res)) {
                    str += '<div>' + k + ': ' + (fracFlag ? parseFloat(v).toFixed(4) : v) + '</div>'
                }
                $div.append(name + ': ' + str)
            }
            else
                $div.append(name + ': ' + (fracFlag ? parseFloat(res).toFixed(4) : res))
        })
    }

    plotDistribution($div, name, st, lbl, oX, oY, dictFlag) {
        let $ddDiv = $("<div></div>")
        $div.append($ddDiv)
        let $button = $("<button></button>").text("get")
        $ddDiv.append(name + ': ')
        $ddDiv.append($button)
        $button.click(async () => {
            $button.prop("disabled", true)
            let data = await Controller.ajaxRequest('/dataset', {get: "stat",stat: st})
            // console.log(data)
            $ddDiv.empty()
            if (!dictFlag)
                data = {"": data}
            for (let [k, v] of Object.entries(data)) {
                let scale = 'linear'
                let type = 'bar'
                if (Object.keys(v).length > 20) {
                    scale = 'logarithmic'
                    type = 'scatter'
                    delete v[0]
                }
                let $canvas = $("<canvas></canvas>").css("height", "300px")
                $ddDiv.append($canvas)
                const ctx = $canvas[0].getContext('2d')
                new Chart(ctx, {
                    type: type,
                    data: {
                        datasets: [{
                            label: lbl,
                            data: v,
                            backgroundColor: 'rgb(52, 132, 246, 0.6)',
                            // borderColor: borderColor,
                            borderWidth: 1,
                            barPercentage: 1,
                            categoryPercentage: 1,
                            borderRadius: 0,
                        }]
                    },
                    options: {
                        // responsive: false,
                        // maintainAspectRatio: true,
                        // aspectRatio: 3,
                        scales: {
                            x: {
                                type: scale,
                                beginAtZero: false,
                                // offset: false,
                                // grid: {
                                //     offset: false
                                // },
                                ticks: {stepSize: 1},
                                title: {
                                    display: true,
                                    text: oX,
                                    font: {size: 14}
                                }
                            },
                            y: {
                                type: scale,
                                suggestedMin: 1,
                                title: {
                                    display: true,
                                    text: oY,
                                    font: {size: 14}
                                }
                            }
                        },
                        plugins: {
                            title: {
                                display: true,
                                text: k + ' ' + name,
                                font: {size: 16}
                            },
                            legend: {display: false},
                        }
                    }
                })
            }
        })
    }

    // Get information HTML
    getInfo() {
        // TODO extend
        // let N = this.numNodes
        // let E = this.numEdges
        let html = ''
        html += `Name: ${this.datasetInfo["name"]}`
        // html += `<br>Size: ${N} nodes, ${E} edges`
        html += `<br>Directed: ${this.datasetInfo.directed}`
        // html += `<br>Weighted: ${this.weighted}`
        html += `<br>Attributes:`

        // List attributes or their number
        let attributesNames = this.datasetInfo["node_attributes"]["names"]
        let attributesTypes = this.datasetInfo["node_attributes"]["types"]
        let attributesValues = this.datasetInfo["node_attributes"]["values"]
        if (attributesNames.length > 30)
            html += ` ${attributesNames.length} (not shown)`
        else {
            let $attrList = $("<ul></ul>")
            for (let i = 0; i < Math.min(10, attributesNames.length); i++) {
                let item = '"' + attributesNames[i] + '"'
                item += ' - ' + attributesTypes[i]
                item += ', values: [' + attributesValues[i] + ']'
                let $item = $("<li></li>").text(item)
                $attrList.append($item)
            }
            html += $attrList.prop('outerHTML')
        }
        return html
    }

    // Update a dataset info panel
    update() {
        this._collapse(false)

        this.$infoDiv = $("<div></div>")
        this.$statsDiv = $("<div></div>")
        this.$varStatsDiv = $("<div></div>")
        this.$body.empty()
        this.$body.append(this.$infoDiv)
        this.$body.append(this.$statsDiv)
        this.$body.append(this.$varStatsDiv)

        if (this.datasetInfo == null) {
            this.$infoDiv.append('No dataset specified')
            return
        }

        // Info
        let html = '<u><b>Info</b></u>'
        html += '<br>' + this.getInfo()
        this.$infoDiv.append(html)

        // Stats
        let multi = this.datasetInfo.count > 1
        this.$statsDiv.append('<u><b>Degree statistics</b></u><br>')

        if (multi) {
            this.$statsDiv.append('Graphs: ' + this.datasetInfo.count + '<br>')
            this.$statsDiv.append('Nodes: ' + Math.min(...this.datasetInfo.nodes)
                + ' — ' + Math.max(...this.datasetInfo.nodes) + '<br>')
        } else {
            this.$statsDiv.append('Nodes: ' + this.datasetInfo.nodes[0] + '<br>')
            this.addNumericStat(this.$statsDiv, "Edges", "num_edges", false)
            this.addNumericStat(this.$statsDiv, "Average degree", "avg_degree", true)
        }

        if (!multi) {
            this.addNumericStat(this.$statsDiv, "Clustering", "clustering_coeff", true)
            this.addNumericStat(this.$statsDiv, "Triangles", "num_triangles", false)
            this.addNumericStat(this.$statsDiv, "Number of connected components", "num_cc", false)
            this.addNumericStat(this.$statsDiv, "Giant connected component (GCC) size", "gcc_size", false)
            this.addNumericStat(this.$statsDiv, "GCC relative size (relative size)", "gcc_rel_size", true)
            this.addNumericStat(this.$statsDiv, "GCC diameter", "gcc_diam", false)
            // this.addNumericStat(this.$statsDiv, "GCC 90% effective diameter", "gcc_diam90", true)
            this.addNumericStat(this.$statsDiv, "Degree assortativity", "degree_assort", true)
        }

        if (multi) {
            this.plotDistribution(this.$statsDiv, 'Distribution of number of nodes', 'num_nodes', 'Number of graphs', 'Nodes', 'Number of graphs', false)
            this.plotDistribution(this.$statsDiv, 'Distribution of number of edges', 'num_edges', 'Number of graphs', 'Edges', 'Number of graphs', false)
            // this.plotDistribution(this.$statsDiv, 'Distribution of average degree', 'avg_degree', 'Highest average: ', 'Number of graphs', 'Average degree', 'Number of graphs', false)
        }
        else {
            this.plotDistribution(this.$statsDiv, 'Degree distribution', 'degree_distr', 'Degree', 'Degree', 'Nodes', this.datasetInfo.directed)

            let name1 = 'Attributes assortativity'
            let $acDiv = $("<div></div>")
            this.$statsDiv.append($acDiv)
            let $button1 = $("<button></button>").text("get")
            $acDiv.append(name1 + ': ')
            $acDiv.append($button1)
            $button1.click(async () => {
                $button1.prop("disabled", true)
                let data = Controller.ajaxRequest('/dataset', {get: "stat", stat: "attr_corr"})

                let attrs = data['attributes']
                let correlations = data['correlations']
                $acDiv.empty()
                $acDiv.append(name1 + ':<br>')

                // Adds mouse listener for all elements which shows a tip with given text
                let $tip = $("<span></span>").addClass("tooltip-text")
                $acDiv.append($tip)
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

                // SVG with table
                let count = attrs.length
                let size = Math.min(30, Math.floor(300 / count))
                let svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
                let $svg = $(svg)
                    .css("background-color", "#e7e7e7")
                    // .css("flex-shrink", "0")
                    .css("margin", "5px")
                    .css("width", (count * size) + "px")
                    .css("height", (count * size) + "px")
                $acDiv.append($svg)
                for (let j = 0; j < count; j++) {
                    for (let i = 0; i < count; i++) {
                        let rect = document.createElementNS("http://www.w3.org/2000/svg", "rect")
                        rect.setAttribute('x', size * i)
                        rect.setAttribute('y', size * j)
                        rect.setAttribute('width', size)
                        rect.setAttribute('height', size)
                        let color = valueToColor(correlations[i][j], CORRELATION_COLORMAP, -1, 1)
                        _addTip(rect, `Corr[${attrs[i]}][${attrs[j]}]=` + correlations[i][j])
                        rect.setAttribute('fill', color)
                        rect.setAttribute('stroke', '#e7e7e7')
                        rect.setAttribute('stroke-width', 1)
                        $svg.append(rect)
                    }
                }
            })
        }
    }

    // Update a dataset info panel when Var data is known
    updateVar() {
        this._collapse(false)
        this.$varStatsDiv.empty()
        this.$varStatsDiv.append('<u><b>Variable data statistics</b></u>')
        let multi = this.datasetInfo.count > 1

        this.plotDistribution(this.$varStatsDiv, 'Labels distribution', 'label_distr', 'Items', 'Class', 'Count', false)

        if (!multi)
            this.addNumericStat(this.$varStatsDiv,'Labels assortativity', 'label_assort', true)
    }

    break() {
        super.break()
        this.datasetInfo = null
        this.$body.append("No Dataset selected")
    }
}