class MenuExplainerRunView extends MenuView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        this.$localRunParamsConstructorDiv = null
        this.$globalRunParamsConstructorDiv = null
        this.$constructorDiv = null
        this.$globalConstructorDiv = null
        this.$localConstructorDiv = null
        this.$builButtonsConstructorDiv = null
        this.$build = null
        this.$runGlobal = null
        this.$runLocal = null

        // Variables
        this.multi = null
        this.visibleGraph = null
        this.globalRunning = null
        this.localRunning = null
        this.paramsLocalRunBuilder = null
        this.paramsGlobalRunBuilder = null
    }

    async init(args) {
        super.init()
        console.log('MenuExplainerRunView', args)
        this.multi = args[1]
        this.algorithm = args[2]

        await this.buildMenu()
    }

    onBreak(block) {
        super.onBreak(block)
        if (block === 'er')
            this.visibleGraph.dropExplanation()
    }

    onReceive(block, args) {
        // super.onReceive(block, args)
        if (block === 'er') {
            if ("progress" in args) {
                let progressBar = this.globalRunning ? this.globalProgressBar : this.localProgressBar
                let load = args["progress"]["load"]
                let text = args["progress"]["text"]
                if (load) {
                    console.assert(load >= 0 && load <= 1)
                    progressBar.setLoad(load)
                }
                if (text)
                    progressBar.setText(text)
            }
        }
    }

    // Recreate menu for explainer constructor
    async buildMenu() {
        this.active = true
        this.$mainDiv.children().remove()

        this.$constructorDiv = $("<div></div>").attr(
            "id", this.idPrefix + "-params")
        this.$globalConstructorDiv = $("<div></div>").attr(
            "id", this.idPrefix + "-global")
        this.$localConstructorDiv = $("<div></div>").attr(
            "id", this.idPrefix + "-local")
        this.$localRunParamsConstructorDiv = $("<div></div>").attr(
            "id", this.idPrefix + "-local-run-params")
        this.$globalRunParamsConstructorDiv = $("<div></div>").attr(
            "id", this.idPrefix + "-global-run-params")
        this.$builButtonsConstructorDiv = $("<div></div>").attr(
            "id", this.idPrefix + "-build-buttons")

        this.$mainDiv.append(this.$constructorDiv)
        this.$constructorDiv.append(this.$globalConstructorDiv)
        this.$globalConstructorDiv.append($("<label></label>").html("<h4>Global level</h4>"))
        this.$globalConstructorDiv.append(this.$globalRunParamsConstructorDiv)
        this.$globalRunParamsConstructorDiv.append($("<div></div>").attr("class", "menu-separator"))

        this.$constructorDiv.append(this.$localConstructorDiv)
        this.$localConstructorDiv.append($("<label></label>").html("<h4>Local level</h4>"))
        this.$localConstructorDiv.append(this.$localRunParamsConstructorDiv)
        this.$localRunParamsConstructorDiv.append($("<div></div>").attr("class", "menu-separator"))

        await this.buildConfiguration()
        this.buildButtons()
    }

    async onrun(mode) {
        console.assert(["local", "global"].includes(mode))
        // Form explainer run config from selectors values
        let explainerRunConfig = {
            _config_kwargs: {
                mode: mode,
                kwargs: {
                    _class_name: this.algorithm,
                    _config_kwargs: Object.assign({}, mode === "local"
                        ? this.paramsLocalRunBuilder.kwArgs : this.paramsGlobalRunBuilder.kwArgs),
                }
            }
        }
        // let explainerRunConfig = {
        //     mode: mode,
        //     class_name: this.algorithm,
        //     kwargs: Object.assign(mode === "local"
        //         ? this.paramsLocalRunBuilder.kwArgs : this.paramsGlobalRunBuilder.kwArgs),
        // }

        // self.explainerInfoPanel.explainer.update(null, self.explainerRunConfig) // FIXME
        // if (self.explanation) // Drop current local explanation // FIXME
        this.visibleGraph.dropExplanation()
        let $btn = mode === "local" ? this.$runLocal : this.$runGlobal
        $btn.prop("disabled", true)
        await Controller.ajaxRequest('/explainer', {
            do: "run", explainerRunConfig: JSON_stringify(explainerRunConfig)})
        $btn.prop("disabled", false)
    }
    async onstop(mode) {
        // TODO
        await $.ajax({
            type: 'POST',
            url: '/explainer',
            data: {do: "stop", mode: mode},
        })
    }
    // async onsave() {
    //     // TODO
    //     await $.ajax({
    //         type: 'POST',
    //         url: '/explainer',
    //         data: {do: "save"},
    //         success: (data) => {
    //             console.log("explanation saved at", data)
    //         }
    //     })
    // }

    // Common part of config
    async buildConfiguration() {
        this.$globalConstructorDiv.hide()
        this.$localConstructorDiv.hide()

        this.paramsLocalRunBuilder = new ParamsBuilder(this.$localRunParamsConstructorDiv,
            'ELR', "menu-explainer-constructor-algorithm-paramrun-")
        this.paramsGlobalRunBuilder = new ParamsBuilder(this.$globalRunParamsConstructorDiv,
            'EGR', "menu-explainer-constructor-algorithm-paramrun-")

        // Ad-hoc defined values
        let localRunPostFunction = (algorithm) => {
            // if ("element_idx" in this.paramsLocalRunBuilder.selectors)
            this.paramsLocalRunBuilder.renameParam("element_idx", this.multi ? "Graph" : "Node")
            this.$localConstructorDiv.show()
        }
        let globalRunPostFunction = (algorithm) => {
            this.$globalConstructorDiv.show()
        }

        this.visibleGraph = controller.presenter.datasetView.visibleGraph
        await this.paramsLocalRunBuilder.build(this.algorithm, localRunPostFunction)
        await this.paramsGlobalRunBuilder.build(this.algorithm, globalRunPostFunction)
        this.$builButtonsConstructorDiv.show()

        // Set center node chosen
        this.paramsLocalRunBuilder.setValue(
            "element_idx", this.visibleGraph.visibleConfig["center"])

        // $cb = $("<div></div>").attr("class", "control-block")
        // $cc.append($cb)
        // $cb.append($("<label></label>").text("version")) // TODO assoc to select ?
        // this.$explainVersionInput = $("<input>").attr("type", "number").attr("min", "0")
        //     .attr("step", "1").attr("value", "0")
        // $cb.append(this.$explainVersionInput)
    }

    // Buttons and actions listeners
    buildButtons() {
        // Build button
        let $cb = $("<div></div>").attr("class", "control-block")
        this.$builButtonsConstructorDiv.append($cb)
        this.$build = $("<button></button>").attr("id", "explainer-button-constructor")
            .text("Build").css("margin-right", "12px")
        $cb.append(this.$build)

        // Global buttons
        $cb = $("<div></div>").attr("class", "control-block")
        this.$globalConstructorDiv.append($cb)
        this.$runGlobal= $("<button></button>")
            .attr("id", this.idPrefix + "-button-global-run").text("Run")
        $cb.append(this.$runGlobal)

        // Progress bar
        this.globalProgressBar = new ProgressBar()
        this.globalProgressBar.visible(true)
        this.$globalConstructorDiv.append(this.globalProgressBar.$div)

        // $cb = $("<div></div>").attr("class", "control-block")
        // this.$globalButtonsConstructorDiv.append($cb)
        // let $save = $("<button></button>").attr("id", "explainer-save-constructor").text("Save").prop("disabled", true)
        // $cb.append($save)

        this.$runGlobal.click(async () => {
            if (this.globalRunning) { // Stop running
                this.$runGlobal.prop("disabled", true)
                await this.onstop("global")
                // this.$localConstructorDiv.removeClass("disabledDiv")
            }
            else { // Run
                this.$build.prop("disabled", true)
                this.$runGlobal.prop("disabled", true)
                this.$runGlobal.text("Stop")
                blockDiv(this.$globalRunParamsConstructorDiv, true)
                // $save.prop("disabled", true)
                this.globalRunning = true
                this.globalProgressBar.start()
                await this.onrun("global")
                // Nothing is called after since it can fail - FIXME move to ...?
                blockDiv(this.$globalRunParamsConstructorDiv, false)
                this.$runGlobal.text("Run")
                this.globalRunning = false
            }
        })

        // $save.click(async () => {
        //     await onsave()
        // })

        // Local run button
        $cb = $("<div></div>").attr("class", "control-block")
        this.$localConstructorDiv.append($cb)
        this.$runLocal= $("<button></button>")
            .attr("id", this.idPrefix + "-button-local-run").text("Run")
        $cb.append(this.$runLocal)

        // Progress bar
        this.localProgressBar = new ProgressBar()
        this.$localConstructorDiv.append(this.localProgressBar.$div)
        this.localProgressBar.visible(true)

        // Action listeners
        this.$runLocal.click(async () => {
            if (this.localRunning) { // Stop running
                this.$runLocal.prop("disabled", true)
                await this.onstop("local")
            }
            else { // Run
                this.$build.prop("disabled", true)
                this.$runLocal.prop("disabled", true)
                this.$runLocal.text("Stop")
                blockDiv(this.$localRunParamsConstructorDiv, true)
                // $save.prop("disabled", true)
                this.localRunning = true
                this.localProgressBar.start()
                await this.onrun("local")
                // fixme - Nothing is called after since it can fail
                blockDiv(this.$localRunParamsConstructorDiv, false)
                this.$runLocal.text("Run")
                this.localRunning = false
            }
        })
    }
}
