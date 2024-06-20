class PanelModelStatView extends PanelView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)
        this.$metricsDiv = null
        this.ctx = null
        this.loss = {} // values over epoch
        // metrics values over epoch: {part -> {metric -> [values list]}}
        this.metricsValues = {"train": {}, "val": {}, "test": {}}
        this._lastUpdated = null
        this._waitingUpdate = null
        this.inited = false
    }

    init() {
        super.init("Metrics")
        // this._init()
    }

    _init() {
        // this.$element.append($("<label></label>").html("<u><b>Model statistics</u></b><br>"))
        this.$metricsDiv = $("<div></div>")
            .css("display", "flex")
            .css("flex-flow", "column")
        this.$body.append(this.$metricsDiv)

        let $div = $("<div></div>")
            .css("height", "330px")
            .css("flex-shrink", 0)
        this.$body.append($div)
        let $canvas = $("<canvas></canvas>").css("background-color", "#f1f1f1")
        $div.append($canvas)
        this.ctx = $canvas[0].getContext('2d')
        // this.ctx.style.backgroundColor = 'rgba(255,0,0,255)'
        this.chart = new Chart(this.ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Loss',
                        data: this.loss,
                        pointStyle: 'circle',
                        radius: 0.5,
                        borderColor: '#000000',
                        borderWidth: 1,
                    }
                    ]
            },
            options: {
                maintainAspectRatio: false,
                chartArea: {
                    backgroundColor: 'rgba(255, 255, 255, 1)'
                },
                scales: {
                    x: {
                        type: 'linear',
                        // ticks: {stepSize: 1}, // NOTE: slows down at large number of epochs
                        title: {
                            display: true,
                            text: 'Epochs',
                            font: {size: 14}
                        }
                    },
                    y: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Metric value',
                            font: {size: 14}
                        }
                    }
                },
                elements: {
                    point: {
                        pointStyle: false,
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: name,
                        font: {size: 16}
                    },
                    legend: {
                        display: true,
                        labels: {
                            usePointStyle: true,
                        }
                    },
                }
            }
        })
        // this.lastUpdate = performance.now()
        this._collapse(false)
        this.inited = true
    }

    onSubmit(block, data) {
        super.onSubmit(block, data)
        if (block === "mmc" || block === "mconstr" || block === "mload" || block === "mcustom") {
            if (!this.inited)
                this._init()
        }
    }

    onReceive(block, data) {
        if (block === "mt") {
            if ("metrics" in data) {

                this.update(data["metrics"])
            }
        }
    }

    _reset() {
        console.log('PanelModelStatView._reset()')
        this.loss = {}
        this.metricsValues = {"train": {}, "val": {}, "test": {}}
        this.chart.data.labels= []
        this.chart.data.datasets = []
    }

    // Update figures according to modelResult
    update(data) {
        this.$metricsDiv.empty()
        let epoch = data["epochs"]
        if (epoch === 0 && Object.keys(this.loss).length > 0)
            this._reset()

        this.$metricsDiv.append($("<label></label>").text('Epochs: ' + epoch))
        if (data["loss"]) {
            this.loss[epoch] = data["loss"]
            this.$metricsDiv.append($("<label></label>").text('Loss: ' + data["loss"]))
        }
        if (data["metrics_values"]) {
            this.$metricsDiv.append($("<label></label>").text('Metrics:'))
            for (const [part, metricValues] of Object.entries(data["metrics_values"])) {
                let thisMetricsValues = this.metricsValues[part]
                for (const [metric, value] of Object.entries(metricValues)) {
                    if (!(metric in thisMetricsValues)) {
                        thisMetricsValues[metric] = {}
                        this.chart.data.labels.push(part + '-' + metric);
                        this.chart.data.datasets.push({
                            label: part + ' ' + metric,
                            data: thisMetricsValues[metric],
                            pointStyle: 'circle',
                            radius: 0.5,
                            borderColor: randomColor(0.99, 0.4),
                            borderWidth: 1,
                        })
                    }
                    thisMetricsValues[metric][epoch] = value
                    this.$metricsDiv.append($("<label></label>").text(part + ' ' + metric + ' = ' + value))
                }
            }
        }
        this._updateArgs = arguments
        if (this.collapsed) {
            return
        }
        // for (const [key, value] of Object.entries(data)) {
        //     this.$metricsDiv.append($("<label></label>").text(`${key} = ${value}`))
        // }

        this._updateCycle()
    }

    // Chart is updated not more frequent than once in 100ms
    async _updateCycle() {
        if (this._waitingUpdate) return
        let t = performance.now()
        if (t - this._lastUpdated < 100) {
            this._waitingUpdate = true
            await sleep(100 - t + this._lastUpdated)
        }
        this._waitingUpdate = false
        this._lastUpdated = performance.now()
        this.chart.update('none')
    }

    // Drop all figures
    break() {
        super.break()
        this.loss = {}
        this.metricsValues = {"train": {}, "val": {}, "test": {}}
        this.inited = false
        // if (this.chart) {
        //     this.chart.data.labels= []
        //     this.chart.data.datasets = []
        //     this.chart.update()
        // }
    }
}