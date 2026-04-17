class MenuDatasetVarView extends MenuView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        this.datasetInfo = null

        // Selectors
        this.task = null // which task is chosen
        this.labeling = null // which labeling is chosen
        this.$oneHotNodeInput = null
        this.$tenOnesNodeInput = null
        this.$nodeClusteringInput = null
        this.$nodeDegreeInput = null
    }

    init(datasetInfo) {
        super.init()
        this.datasetInfo = datasetInfo

        let $cc, $cb, id, size
        $cc = $("<div></div>")
        this.$mainDiv.append($cc)

        // 1. Input features
        $cc.append($("<label></label>").html("<h3>Features constructor</h3>"))

        // $cc.append($("<label></label>").html("<h4>Local structural</h4>"))
        //
        // let $cb = $("<div></div>").attr("class", "control-block")
        // $cc.append($cb)
        // let id = "dataset-variable-local-cc"
        // this.$nodeClusteringInput = $("<input>").attr("type", "checkbox").attr("id", id)
        // $cb.append(this.$nodeClusteringInput)
        // $cb.append($("<label></label>").text(`node clustering (size=${1})`).attr("for", id))
        // this.$nodeClusteringInput.prop("disabled", true)
        //
        // $cb = $("<div></div>").attr("class", "control-block")
        // $cc.append($cb)
        // id = "dataset-variable-local-deg"
        // this.$nodeDegreeInput = $("<input>").attr("type", "checkbox").attr("id", id)
        // $cb.append(this.$nodeDegreeInput)
        // $cb.append($("<label></label>").text(`node degree (size=${1})`).attr("for", id))
        // this.$nodeDegreeInput.prop("disabled", true)

        if (this.datasetInfo.hetero) {
            this.$oneHotNodeInput = {}
            let nodeTypes = Object.keys(this.datasetInfo["nodes"][0])

            // Global
            $cc.append($("<label></label>").html("<h4>Global structural</h4>"))
            for (const nt of nodeTypes) {
                $cc.append($("<label></label>").html("<h5>" + nt + "</h5>"))

                $cb = $("<div></div>").attr("class", "control-block")
                $cc.append($cb)
                id = this.idPrefix  + '-' + nt + "-node-1hot-input"
                size = this.datasetInfo["nodes"][0][nt]
                this.$oneHotNodeInput[nt] = $("<input>").attr("type", "checkbox").attr("id", id)
                $cb.append(this.$oneHotNodeInput[nt])
                $cb.append($("<label></label>").text(`1-hot input (size=${size})`).attr("for", id))
            }

            // Node attributes
            $cc.append($("<label></label>").html("<h4>Node attributes</h4>"))
            for (const nt of nodeTypes) {
                $cc.append($("<label></label>").html("<h5>" + nt + "</h5>"))

                let attrs = this.datasetInfo["node_attributes"][nt]["names"]
                let values = this.datasetInfo["node_attributes"][nt]["values"]
                if (this.$oneHotNodeInput[nt] && attrs.length === 0) {
                    this.$oneHotNodeInput[nt].prop("checked", true)
                    this.$oneHotNodeInput[nt].click((e) => e.preventDefault())
                } else {
                    let i = 0
                    for (const attr of attrs) {
                        size = 1
                        switch (this.datasetInfo["node_attributes"][nt]["types"][i]) {
                            case "continuous":
                                size = 1
                                break
                            case "categorical":
                                size = values[i].length
                                break
                            case "vector":
                                size = values[i]
                                break
                            case "other":
                                size = 0  // FIXME check
                        }
                        ++i
                        if (size === 0)
                            continue
                        let $cb = $("<div></div>").attr("class", "control-block")
                        $cc.append($cb)
                        id = this.idPrefix + '-' + nt + "-attribute-" + nameToId(attr)
                        $cb.append($("<input>").attr("type", "checkbox").attr("id", id))
                        $cb.append($("<label></label>").text(attr + ` (size=${size})`).attr("for", id))
                    }
                    $('#' + this.idPrefix + "-attribute-" + nameToId(attrs[0])).prop('checked', true)
                }
            }
        }
        else {
            // Global if single graph
            if (this.datasetInfo.count === 1) {
                $cc.append($("<label></label>").html("<h4>Global structural</h4>"))

                // 1-hot over nodes
                $cb = $("<div></div>").attr("class", "control-block")
                $cc.append($cb)
                id = this.idPrefix + "-node-1hot-input"
                size = this.datasetInfo["nodes"].sum()
                this.$oneHotNodeInput = $("<input>")
                    .attr("type", "checkbox").attr("id", id)
                $cb.append(this.$oneHotNodeInput)
                $cb.append($("<label></label>")
                    .text(`1-hot ovre nodes (size=${size})`).attr("for", id))

                // 10 ones
                $cb = $("<div></div>").attr("class", "control-block")
                $cc.append($cb)
                id = this.idPrefix + "-node-10ones-input"
                this.$tenOnesNodeInput = $("<input>")
                    .attr("type", "checkbox").attr("id", id)
                $cb.append(this.$tenOnesNodeInput)
                $cb.append($("<label></label>")
                    .text("10 ones (size=10)").attr("for", id))

                // Degree
                $cb = $("<div></div>").attr("class", "control-block")
                $cc.append($cb)
                id = this.idPrefix + "-node-degree-input"
                this.$nodeDegreeInput = $("<input>")
                    .attr("type", "checkbox").attr("id", id)
                $cb.append(this.$nodeDegreeInput)
                $cb.append($("<label></label>")
                    .text("degree (size=1)").attr("for", id))

                // Clustering
                $cb = $("<div></div>").attr("class", "control-block")
                $cc.append($cb)
                id = this.idPrefix + "-node-clustering-input"
                this.$nodeClusteringInput = $("<input>")
                    .attr("type", "checkbox").attr("id", id)
                $cb.append(this.$nodeClusteringInput)
                $cb.append($("<label></label>")
                    .text("clustering (size=1)").attr("for", id))
            }
            else {
                this.$oneHotNodeInput = null
                this.$tenOnesNodeInput = null
                this.$nodeClusteringInput = null
                this.$nodeDegreeInput = null
            }

            // Node attributes
            $cc.append($("<label></label>").html("<h4>Node attributes</h4>"))
            let attrs = this.datasetInfo["node_attributes"]["names"]
            let values = this.datasetInfo["node_attributes"]["values"]
            if (this.$oneHotNodeInput && attrs.length === 0) {
                this.$oneHotNodeInput.prop("checked", true)
                this.$oneHotNodeInput.click((e) => e.preventDefault())
            } else {
                let i = 0
                for (const attr of attrs) {
                    let $cb = $("<div></div>").attr("class", "control-block")
                    $cc.append($cb)
                    id = this.idPrefix + "-attribute-" + nameToId(attr)
                    size = 1
                    switch (this.datasetInfo["node_attributes"]["types"][i]) {
                        case "continuous":
                            size = 1
                            break
                        case "categorical":
                            size = values[i].length
                            break
                        case "vector":
                            size = values[i]
                    }
                    $cb.append($("<input>").attr("type", "checkbox").attr("id", id))
                    $cb.append($("<label></label>").text(attr + ` (size=${size})`).attr("for", id))
                    ++i
                }
                $('#' + this.idPrefix + "-attribute-" + nameToId(attrs[0])).prop('checked', true)
            }
        }
        // 2. class labels
        $cc.append($("<div></div>").attr("class", "menu-separator"))

        $cc.append($("<label></label>").html("<h3>Task & labeling</h3>"))
        let taskLabelings = this.datasetInfo["labelings"]
        // Add edge prediciton
        taskLabelings["edge-prediction"] = {"default": null}
        let checked = false
        for (const [task, labelingValue] of Object.entries(taskLabelings)) {
            $cc.append($("<label></label>").html(`<h4>${task}</h4>`))
            for (const [labeling, value] of Object.entries(labelingValue)) {
                let $cb = $("<div></div>").attr("class", "control-block")
                $cc.append($cb)
                let id = this.idPrefix + "-labelings-" + nameToId(task + labeling)
                let $input = $("<input>").attr("type", "radio")
                    .attr("name", "dataset-variable-labelings")
                    .attr("id", id).attr("value", labeling)
                $cb.append($input)
                let text = labeling
                if (task.endsWith("classification"))
                    text += ` (${value} classes)`

                $cb.append($("<label></label>").text(text)
                    .attr("for", id))
                $input.change(() => {
                    this.task = task
                    this.labeling = labeling
                })

                // Check the first option
                if (!checked) {
                    $input.change()
                    $input.prop('checked', true)
                    checked = true
                }
            }
        }
        // this.task = Object.keys(taskLabelings)[0]
        // this.labeling = Object.keys(taskLabelings)[0]
        // $('#' + this.idPrefix + "-labelings-" + nameToId(this.labeling)).prop("checked", true)

        this.appendAcceptBreakButtons()
        // this.$acceptDiv.hide()
    }

    async _accept() {
        // Construct config
        let attrsChecked = []
        let attrs = this.datasetInfo["node_attributes"]["names"]
        for (const attr of attrs) {
            let id = this.idPrefix + "-attribute-" + nameToId(attr)
            attrsChecked.push($('#' + id).is(':checked'))
        }

        // Fill features according to format
        let features = {"node_struct": [], "node_attr": []}
        // If no attributes , check oneHotNodeInput
        if (attrs.length === 0 || (attrs.length > 0 && !attrsChecked.reduce((a, v) => a || v, false))) {
            if (this.$tenOnesNodeInput == null)
                console.error("No attributes and no node 1-hot - features will be null!")
            else
                this.$tenOnesNodeInput.prop('checked', true)
        }

        if (this.$oneHotNodeInput && this.$oneHotNodeInput.is(":checked"))
            features["node_struct"].push("one_hot")
        if (this.$tenOnesNodeInput && this.$tenOnesNodeInput.is(":checked"))
            features["node_struct"].push("10-ones")
        if (this.$nodeClusteringInput && this.$nodeClusteringInput.is(":checked"))
            features["node_struct"].push("clustering")
        if (this.$nodeDegreeInput && this.$nodeDegreeInput.is(":checked"))
            features["node_struct"].push("degree")
        for (let i = 0; i < attrs.length; i++) {
            if (attrsChecked[i])
                features["node_attr"].push(attrs[i])
        }

        let datasetVarConfig = {
            task: this.task,
            labeling: this.labeling,
            features: features,
            dataset_ver_ind: 0, // TODO check
        }
        await controller.blockRequest(this.requestBlock, 'modify', datasetVarConfig)
    }

}

