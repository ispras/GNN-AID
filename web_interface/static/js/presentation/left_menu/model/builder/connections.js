class Connections {
    constructor(layerType, layerIx, nodeCount, graphCount, idPrefix) {
        this.layerType = layerType // current layer type ('n' or 'g')
        this.layerIx = layerIx // current layer index
        this.nodeCount = nodeCount // number of node layers
        this.graphCount = graphCount // number of graph layers

        this.$div = $("<div></div>").css("margin-left", LayerBlock.leftMargin + "px")
        this.idPrefix = idPrefix ? idPrefix : timeBasedId() + '-'
        this.nodeChecks = null
        this.graphChecks = null
        this.nodeDivs = null // div for node label and params selectors
        this.graphDivs = null // div for graph label and params selectors
        this.poolSelects = null // graph layer ix -> pool type (only if from 'n' to 'g')
        this.aggrSelects = null // graph layer ix -> aggregation type
    }

    // Update constraints when index or number of layers has changed
    update(layerIx, nodeCount, graphCount) {
        this.layerIx = layerIx
        this.nodeCount = nodeCount
        this.graphCount = graphCount
        this.$div[0].outerHtml = ''

        // Create checkboxes with params for nodes, keep existing ones
        let self = this
        let i = this.layerIx + 2
        // Remove preceding
        for (let j = 0; j < i; j++) {
            if (j in this.nodeChecks) {
                this.nodeDivs[j].remove()
                delete this.nodeDivs[j]
                delete this.nodeChecks[j]
            }
        }
        // Add new ones
        if (this.layerType === 'n') {
            for (; i < this.nodeCount; i++) {
                if (i in this.nodeChecks) {
                    this.$div.append(this.nodeDivs[i])
                    continue
                }

                let $div = this.nodeDivs[i] = $("<div></div>")
                this.$div.append($div)
                let $cb = $("<div></div>").attr("class", "control-block")
                $div.append($cb)
                let id = this.idPrefix + "label-node-" + i
                $cb.append($("<label></label>").text("to Node layer " + i).attr("for", id))
                let $nodeCheck = $("<input>").attr("id", id).attr("type", "checkbox")
                $cb.append($nodeCheck)
                this.nodeChecks[i] = $nodeCheck
                let $paramsDiv = self._addAggr(i).hide()
                    .css("background-color", LayerBlock.connectionColor)
                $cb.after($paramsDiv)
                $nodeCheck.change(function () {
                    if (this.checked)
                        $paramsDiv.show()
                    else
                        $paramsDiv.hide()
                })
            }
            // Remove excessive if exists
            if (i in this.nodeChecks) {
                self.nodeDivs[i].remove()
                delete self.nodeDivs[i]
                delete self.nodeChecks[i]
            }
        }

        // Create checkboxes with params for graphs, keep existing ones
        // Possible to connect to layers with index >= (this_layer_ix + 2)
        let start = this.layerType === 'g' ? this.layerIx + 2 : 0
        // Remove preceding
        for (let j = 0; j < start; j++) {
            if (j in this.graphChecks) {
                this.graphDivs[j].remove()
                delete this.graphDivs[j]
                delete this.graphChecks[j]
            }
        }
        for (i = start; i < this.graphCount; i++) {
            if (i in this.graphChecks) {
                this.$div.append(this.graphDivs[i])
                continue
            }

            let $div = this.graphDivs[i] = $("<div></div>")
            this.$div.append($div)
            let $cb = $("<div></div>").attr("class", "control-block")
            $div.append($cb)
            let id = this.idPrefix + "label-graph-" + i
            $cb.append($("<label></label>").text("to Graph layer " + i).attr("for", id))
            let $graphCheck = $("<input>").attr("id", id).attr("type", "checkbox")
            $cb.append($graphCheck)
            this.graphChecks[i] = $graphCheck
            let $paramsDiv = $("<div></div>").hide()
                .css("background-color", LayerBlock.connectionColor)
            $paramsDiv.append(self._addAggr(i))
            if (self.layerType === 'n')
                $paramsDiv.append(self._addPool(i))
            $cb.after($paramsDiv)
            $graphCheck.change(function () {
                if (this.checked)
                    $paramsDiv.show()
                else
                    $paramsDiv.hide()
            })
        }
        // Remove excessive if exists
        if (i in this.graphChecks) {
            self.graphDivs[i].remove()
            delete self.graphDivs[i]
            delete this.graphChecks[i]
        }
    }

    setAsLast() {
        // If this is node layer before graph layers
        if (this.layerType === 'n' && Object.values(this.graphChecks).length > 0) {
            // for (const $graphCheck of Object.values(this.graphChecks))
            //     if ($graphCheck.is(':checked')) return

            // Force connection to 0-th graph layer
            if (!this.graphChecks[0].is(':checked'))
                this.graphChecks[0].click()
        }
    }

    _addPool(ix) {
        // Pool type
        let $cb = $("<div></div>").attr("class", "control-block")
            .css("margin-left", LayerBlock.leftMargin + "px")
        let id = this.idPrefix + "pool-" + ix
        $cb.append($("<label></label>").text("Pool type").attr("for", id))
            // .css("background-color", LayerBlock.skipColor)
        let $poolSelect = $("<select></select>").attr("id", id)
        $cb.append($poolSelect)
        $poolSelect.append($("<option></option>").val("global_add_pool").text("global add"))
        this.poolSelects[ix] = $poolSelect
        return $cb
    }

    _addAggr(ix) {
        // Aggregation type
        let $cb = $("<div></div>").attr("class", "control-block")
            .css("margin-left", LayerBlock.leftMargin + "px")
        let id = this.idPrefix + "aggr-" + ix
        $cb.append($("<label></label>").text("Aggregation type").attr("for", id))
        let $aggrSelect = $("<select></select>").attr("id", id)
        $cb.append($aggrSelect)
        $aggrSelect.append($("<option></option>").val("cat").text("cat"))
        this.aggrSelects[ix] = $aggrSelect
        return $cb
    }

    build() {
        this.nodeChecks = {}
        this.graphChecks = {}
        this.nodeDivs = {}
        this.graphDivs = {}
        this.poolSelects = {}
        this.aggrSelects = {}
        this.update(this.layerIx, this.nodeCount, this.graphCount)
    }

    constructConfig() {
        let config = []
        let additionalNodeIxes = []
        for (const [i, $nodeCheck] of Object.entries(this.nodeChecks)) {
            if ($nodeCheck.is(':checked')) {
                let aggr = this.aggrSelects[i].val()
                if (aggr === 'cat') // TODO what else?
                    additionalNodeIxes.push(i)
                config.push({
                    'into_layer': parseInt(i),
                    'connection_kwargs': {
                        'aggregation_type': aggr
                    }
                })
            }
        }
        let additionalGraphIxes = []
        for (const [i, $graphCheck] of Object.entries(this.graphChecks)) {
            if ($graphCheck.is(':checked')) {
                let aggr = this.aggrSelects[i].val()
                if (aggr === 'cat') // TODO what else?
                    additionalGraphIxes.push(i)
                config.push({
                    'into_layer': parseInt(i) + this.nodeCount,
                    'connection_kwargs': {
                        'pool': {
                            'pool_type': this.poolSelects[i].val(),
                        },
                        'aggregation_type': aggr
                    }
                })
            }
        }
        return [config, additionalNodeIxes, additionalGraphIxes]
    }
}
