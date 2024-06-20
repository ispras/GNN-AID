class MenuDatasetVarView extends MenuView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        this.datasetInfo = null

        // Selectors
        this.labeling = null // which labeling is chosen
        this.$nodeClusteringInput = null
        this.$nodeDegreeInput = null
        this.$oneHotNodeInput = null
        this.$attackTypeSelect = null
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

        if (this.datasetInfo.count === 1) {
            $cc.append($("<label></label>").html("<h4>Global structural</h4>"))

            $cb = $("<div></div>").attr("class", "control-block")
            $cc.append($cb)
            id = this.idPrefix + "-node-1hot-input"
            size = this.datasetInfo["nodes"].sum()
            this.$oneHotNodeInput = $("<input>").attr("type", "checkbox").attr("id", id)
            $cb.append(this.$oneHotNodeInput)
            $cb.append($("<label></label>").text(`1-hot input (size=${size})`).attr("for", id))
        }
        else
            this.$oneHotNodeInput = null

        $cc.append($("<label></label>").html("<h4>Node attributes</h4>"))
        let attrs = this.datasetInfo["node_attributes"]["names"]
        let values = this.datasetInfo["node_attributes"]["values"]
        if (this.$oneHotNodeInput && attrs.length === 0) {
            this.$oneHotNodeInput.prop("checked", true)
            this.$oneHotNodeInput.click((e) => e.preventDefault())
        }
        else {
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
                    case "other":
                        size = values[i]
                }
                $cb.append($("<input>").attr("type", "checkbox").attr("id", id))
                $cb.append($("<label></label>").text(attr + ` (size=${size})`).attr("for", id))
                ++i
            }
            $('#' + this.idPrefix + "-attribute-" + nameToId(attrs[0])).prop('checked', true)
        }

        // 2. class labels
        $cc.append($("<div></div>").attr("class", "menu-separator"))

        $cc.append($("<label></label>").html("<h3>Class labeling</h3>"))
        // TODO 2 cases
        let labelingClasses = this.datasetInfo["labelings"]
        for (const [labeling, classes] of Object.entries(labelingClasses)) {
            let $cb = $("<div></div>").attr("class", "control-block")
            $cc.append($cb)
            let id = this.idPrefix + "-labelings-" + nameToId(labeling)
            let $input = $("<input>").attr("type", "radio").attr("name", "dataset-variable-labelings").attr("id", id).attr("value", labeling)
            $cb.append($input)
            $cb.append($("<label></label>").text(labeling + ` (${classes} classes)`).attr("for", id))
            // TODO can do this if graph is small
            // $input.change(() => this.setLabels(labeling))
        }
        this.labeling = Object.keys(labelingClasses)[0]
        $('#' + this.idPrefix + "-labelings-" + nameToId(this.labeling)).prop("checked", true)

        // 3. Add options for attack type
        $cc.append($("<div></div>").attr("class", "menu-separator"))
        $cb = $("<div></div>").attr("class", "control-block")
        $cc.append($cb)
        $cb.append($("<label></label>").text("Attack"))
        this.$attackTypeSelect = $("<select></select>").attr("id", this.idPrefix + "-attack")
        $cb.append(this.$attackTypeSelect)
        // TODO need possible attack types here
        let attack = "original"
        this.$attackTypeSelect.append($("<option></option>").text(attack))

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

        // If no attributes checked, check oneHotNodeInput
        if (attrs.length > 0 && !attrsChecked.reduce((a, v) => a || v, false)) {
            if (this.$oneHotNodeInput == null)
                console.error("No attributes and no node 1-hot - features will be null!")
            else
                this.$oneHotNodeInput.prop('checked', true)
        }

        // Fill features according to format
        let features = {"attr": {}}
        // if (this.$nodeClusteringInput.is(":checked"))
        //     features["str_f"] = "c"
        // if (this.$nodeDegreeInput.is(":checked"))
        //     features["str_f"] = "d"
        if (this.$oneHotNodeInput && this.$oneHotNodeInput.is(":checked"))
            features["str_g"] = "one_hot"
        for (let i = 0; i < attrs.length; i++) {
            if (attrsChecked[i])
                features["attr"][attrs[i]] = this.datasetInfo["node_attributes"]["types"][i]
        }
        this.labeling = $("input[name='dataset-variable-labelings']:checked").val()

        let datasetVarConfig = {
            features: features,
            labeling: this.labeling,
            dataset_attack_type: this.$attackTypeSelect.val(),
            dataset_ver_ind: 0, // TODO check
        }
        await Controller.blockRequest(this.requestBlock, 'modify', datasetVarConfig)
    }

}

