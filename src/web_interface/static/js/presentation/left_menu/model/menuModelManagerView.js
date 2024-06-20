class MenuModelManagerView extends MenuView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        // Variables
        this.mm_info = null // Dict of appropriate managers, {manager -> info}

        // Training params
        this.managerParamsBuilder = null
        this.$optimizerSelect = null
        this.optimizerParamsBuilder = null
        this.$lossSelect = null
        this.lossParamsBuilder = null
        this.$trainRatioInput = null
        this.$valRatioInput = null
        this.$testRatioInput = null
        this.$trainMaskFlag = null
    }

    async init(arg) {
        super.init()
        this.appendAcceptBreakButtons()
        // this.$acceptDiv.hide()

        this.mm_info = arg

        await this.buildManager()
    }

    async _accept() {
        let mmc = this.constructManagerConfig()
        // console.log("MM config", mmc)
        await Controller.blockRequest(this.requestBlock, 'modify', mmc)
    }

    // Model manager parameters
    async buildManager() {
        let $cc = this.$configManagerParamsDiv = $("<div></div>")
        this.$mainDiv.append(this.$configManagerParamsDiv)
        let $cb
        $cc.append($("<label></label>").html("<h3>Manager</h3>"))

        // Specific manager params
        let paramsType = "FW"
        // paramsType = "FW_custom"
        ParamsBuilder.addParams(paramsType, this.mm_info)
        this.mm_info = Object.keys(this.mm_info)

        // Manager params
        let $managerParamsDiv
        [$cb, this.$managerSelect, $managerParamsDiv, this.managerParamsBuilder]
            = await addOptionsWithParams("menu-model-constructor-manager",
            "Model manager", this.mm_info.map((x) => [x, x]), paramsType)
        $cc.append($cb)
        $cc.append($managerParamsDiv)

        // Optimizer
        let $optimizerParamsDiv
        [$cb, this.$optimizerSelect, $optimizerParamsDiv, this.optimizerParamsBuilder]
            = await addOptionsWithParams("menu-model-constructor-optimizer",
            "Optimizer", null, "O")
        $cc.append($cb)
        $cc.append($optimizerParamsDiv)

        // Loss function
        let $lossParamsDiv
        [$cb, this.$lossSelect, $lossParamsDiv, this.lossParamsBuilder]
            = await addOptionsWithParams("menu-model-constructor-loss",
            "Loss function", [["NLLLoss", "NLL"], ["CrossEntropyLoss", "CE"]], "F")
        $cc.append($cb)
        $cc.append($lossParamsDiv)

        // // Batch
        // $cb = $("<div></div>").attr("class", "control-block")
        // $cc.append($cb)
        // $cb.append($("<label></label>").text("Batch")) // TODO assoc to select ?
        // this.$batch = $("<input>").attr("type", "number").attr("min", "1")
        //     .attr("max", "100000000").attr("step", "1").attr("value", "10000")
        // $cb.append(this.$batch)

        // Train-val-test block
        $cb = $("<div></div>").attr("class", "control-block").css("display", "block")
        $cc.append($cb)
        $cb.append($("<label></label>").text("Train/validation/test ratio"))
        let $div3 = $("<div></div>").css("display", "flex")
        $cb.append($div3)
        this.$trainRatioInput = $("<input>").attr("type", "number").attr("min", "0")
            .attr("max", "1").attr("step", "0.01").attr("value", "0.6")
        $div3.append(this.$trainRatioInput)

        this.$valRatioInput = $("<input>").attr("type", "number").attr("min", "0")
            .attr("max", "0.99").attr("step", "0.01").attr("value", "0")
        $div3.append(this.$valRatioInput)

        this.$testRatioInput = $("<input>").attr("type", "number").attr("min", "0")
            .attr("max", "1").attr("step", "0.01").attr("value", "0.4")
        $div3.append(this.$testRatioInput)

        // Balance between 3 fields
        this.$trainRatioInput.change((e) => { // val, test
            let val = 1 - e.target.valueAsNumber - this.$testRatioInput.val()
            if (val > 0)
                this.$valRatioInput.val(Math.round(val*1e4)/1e4)
            else {
                this.$valRatioInput.val(0)
                val = 1 - e.target.valueAsNumber
                this.$testRatioInput.val(Math.round(val*1e4)/1e4)
            }
        })
        this.$valRatioInput.change((e) => { // train, test
            let val = 1 - e.target.valueAsNumber - this.$testRatioInput.val()
            if (val > 0)
                this.$trainRatioInput.val(Math.round(val*1e4)/1e4)
            else {
                this.$trainRatioInput.val(0)
                val = 1 - e.target.valueAsNumber
                this.$testRatioInput.val(Math.round(val*1e4)/1e4)
            }
        })
        this.$testRatioInput.change((e) => { // val, train
            let val = 1 - e.target.valueAsNumber - this.$trainRatioInput.val()
            if (val > 0)
                this.$valRatioInput.val(Math.round(val*1e4)/1e4)
            else {
                this.$valRatioInput.val(0)
                val = 1 - e.target.valueAsNumber
                this.$trainRatioInput.val(Math.round(val*1e4)/1e4)
            }
        })

        // // Train mask flag
        // $cb = $("<div></div>").attr("class", "control-block")
        // $cc.append($cb)
        // let id = "menu-model-constructor-trainMaskFlag"
        // $cb.append($("<label></label>").text("Train mask flag").attr("for", id))
        // this.$trainMaskFlag = $("<input>").attr("id", id).attr("type", "checkbox").prop("checked", false)
        // $cb.append(this.$trainMaskFlag)

        // $cc.append($("<div></div>").attr("class", "menu-separator"))

    }

    // drop() {
    //     this.$configManagerDiv.empty()
    //     this.managerActive = false
    // }

    // Form manager config from selectors values
    constructManagerConfig() {
        let managerConfig = {
            class: this.$managerSelect.val(),
            optimizer: {
                _class_name: this.$optimizerSelect.val(),
                _config_kwargs: Object.assign({}, this.optimizerParamsBuilder.kwArgs)
            },
            loss_function: {
                _class_name: this.$lossSelect.val(),
                _config_kwargs: Object.assign({}, this.lossParamsBuilder.kwArgs)
            },
            // optimizer: Object.assign(
            //     {_class_name: this.$optimizerSelect.val()},
            //     this.optimizerParamsBuilder.kwArgs),
            // loss_function: Object.assign(
            //     {_class_name: this.$lossSelect.val()},
            //     this.lossParamsBuilder.kwArgs),
            train_test_split: [
                parseFloat(this.$trainRatioInput.val()),
                // parseFloat(this.$valRatioInput.val()),
                parseFloat(this.$testRatioInput.val())],
        }

        Object.assign(managerConfig, this.managerParamsBuilder.kwArgs)
        return managerConfig
    }
}