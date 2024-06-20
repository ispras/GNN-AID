class PanelDatasetView extends PanelView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        this.init()

        // Variables
        this.datasetInfo = null
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

    onSubmit(block, data) {
        // Do nothing
    }

    addNumericStat(name, stat, fracFlag) {
        let $div = $("<div></div>")
        this.$body.append($div)
        let $button = $("<button></button>").text("get")
        $div.append(name + ': ')
        $div.append($button)
        $button.click(async () => {
            $button.prop("disabled", true)
            let res = await Controller.ajaxRequest('/dataset', {get: "stat", stat: stat})
            $div.empty()
            $div.append(name + ': ' + (fracFlag ? parseFloat(res).toFixed(4) : res))
        })
    }

    plotDistribution(name, st, txt, lbl, oX, oY) {
        let $ddDiv = $("<div></div>")
        this.$body.append($ddDiv)
        let $button = $("<button></button>").text("get")
        $ddDiv.append(name + ': ')
        $ddDiv.append($button)
        $button.click(async () => {
            $button.prop("disabled", true)
            await $.ajax({
                type: 'POST',
                url: '/dataset',
                data: {
                    get: "stat",
                    stat: st,
                },
                success: function (data) {
                    data = JSON_parse(data)
                    // console.log(data)
                    $ddDiv.empty()
                    let scale = 'linear'
                    let type = 'bar'
                    if (Object.keys(data).length > 20) {
                        scale = 'logarithmic'
                        type = 'scatter'
                        delete data[0]
                    }
                    let $canvas = $("<canvas></canvas>").css("height", "300px")
                    $ddDiv.append($canvas)
                    const ctx = $canvas[0].getContext('2d')
                    new Chart(ctx, {
                        type: type,
                        data: {
                            datasets: [{
                                label: lbl,
                                data: data,
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
                                    text: name,
                                    font: {size: 16}
                                },
                                legend: {display: false},
                            }
                        }
                    })
                }
            })
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
        // this.updateArgs = arguments
        // if (this.collapsed) {
        //     return
        // }
        // if (dataset === this.dataset) return

        // this.dataset = dataset
        this.$body.empty()

        if (this.datasetInfo == null) {
            this.$body.append('No dataset specified')
            return
        }

        // Info
        let html = '<u><b>Info</b></u>'
        html += '<br>' + this.getInfo()
        this.$body.append(html)

        // Stats
        let multi = this.datasetInfo.count > 1
        this.$body.append('<u><b>Statistics</b></u><br>')

        if (multi) {
            this.$body.append('Graphs: ' + this.datasetInfo.count + '<br>')
            this.$body.append('Nodes: ' + Math.min(...this.datasetInfo.nodes)
                + ' — ' + Math.max(...this.datasetInfo.nodes) + '<br>')
        } else {
            this.$body.append('Nodes: ' + this.datasetInfo.nodes[0] + '<br>')
        }
        this.addNumericStat("Edges", "num_edges", false)
        this.addNumericStat("Average degree", "avg_deg", true)

        if (!multi) {
            this.addNumericStat("Clustering", "CC", true)
            this.addNumericStat("Triangles", "triangles", false)
            this.addNumericStat("Diameter", "diameter", false)
            this.addNumericStat("Number of connected components", "cc", false)
            this.addNumericStat("Largest connected component size", "lcc", false)
            this.addNumericStat("Degree assortativity", "degree_assortativity", true)
        }

        if (multi) {
            this.plotDistribution('Distribution of number of nodes', 'num_nodes_distr', 'Maximum nodes: ', 'Number of graphs', 'Nodes',
                'Number of graphs', multi, false)
            this.plotDistribution('Distribution of average degree', 'avg_degree_distr', 'Highest average: ', 'Number of graphs',
                'Average degree', 'Number of graphs', multi, false)
        }
        else {
            this.plotDistribution(
                'Degree distribution', 'DD', 'Maximum degree: ', 'Degree',
                'Nodes', 'Degree', multi, true)

            let name1 = 'Attributes assortativity'
            let $acDiv = $("<div></div>")
            this.$body.append($acDiv)
            let $button1 = $("<button></button>").text("get")
            $acDiv.append(name1 + ': ')
            $acDiv.append($button1)
            $button1.click(async () => {
                $button1.prop("disabled", true)
                await $.ajax({
                    type: 'POST',
                    url: '/dataset',
                    data: {
                        get: "stat",
                        stat: "attr_corr",
                    },
                    success: function (data) {
                        data = JSON_parse(data)
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

                    }
                })
            })
        }
    }

    break() {
        super.break()
        this.$body.append("No Dataset selected")
    }
}