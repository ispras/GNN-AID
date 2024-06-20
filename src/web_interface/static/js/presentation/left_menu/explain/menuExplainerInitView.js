class MenuExplainerInitView extends MenuView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        this.$initConfigConstructorDiv = null
        this.$initParamsConstructorDiv = null

        // Variables
        this.availableExplainers = null // list of available algorithms
        this.paramsInitBuilder = null
    }

    async init(args) {
        super.init()
        this.appendAcceptBreakButtons()
        this.$acceptDiv.hide()

        this.availableExplainers = args // .toSorted()
        await this.updateMenu()
    }

    async _accept() {
        // Form explainer init config from selectors values
        let explainerInitConfig = {
            _class_name: this.$algorithmSelect.val(),
            _config_kwargs: Object.assign({}, this.paramsInitBuilder.kwArgs)
        }
        // let explainerInitConfig = {
        //     class_name: this.$algorithmSelect.val(),
        //     // explainer_ver_ind: null, // Create a new version
        //     // explainer_ver_ind: parseInt(this.$explainVersionInput.val()),
        //     // kwargs: Object.assign({}, this.paramsInitBuilder.kwArgs),
        // }
        // Object.assign(explainerInitConfig, this.paramsInitBuilder.kwArgs)
        console.log("explainerInitConfig", explainerInitConfig)
        await Controller.blockRequest(this.requestBlock, 'modify', explainerInitConfig)
    }

    // Recreate menu for explainer constructor
    async updateMenu() {
        this.$mainDiv.empty()

        this.$initConfigConstructorDiv = $("<div></div>").attr(
            "id", this.idPrefix + "-init-config")
        this.$initParamsConstructorDiv = $("<div></div>").attr(
            "id", this.idPrefix + "-init-params")

        this.$mainDiv.append(this.$initConfigConstructorDiv)

        let $cb = $("<div></div>").attr("class", "control-block")
        this.$initConfigConstructorDiv.append($cb)
        let id = this.idPrefix + "-algorithm"
        $cb.append($("<label></label>").text("Algorithm").attr("for", id))
        this.$algorithmSelect = $("<select></select>").attr("id", id)
        $cb.append(this.$algorithmSelect)
        this.$algorithmSelect.append($("<option></option>").attr("selected", "true")
            .attr("value", "").attr("disabled", "").text(`<select algorithm>`))

        for (const key of this.availableExplainers)
            this.$algorithmSelect.append($("<option></option>").text(key))

        this.paramsInitBuilder = new ParamsBuilder(this.$initParamsConstructorDiv,
            'EI', this.idPrefix + "-algorithm-paraminit-")

        // Ad-hoc defined values
        let initPostFunction = (algorithm) => {}

        let self = this
        this.$algorithmSelect.change(async function () {
            self.paramsInitBuilder.drop()
            // await self.unlock(true) // Clear chosen values
            await self.paramsInitBuilder.build(this.value, initPostFunction)
            self.$acceptDiv.show()
        })

        this.$initConfigConstructorDiv.append(this.$initParamsConstructorDiv)

        // $cb = $("<div></div>").attr("class", "control-block")
        // $cc.append($cb)
        // $cb.append($("<label></label>").text("version")) // TODO assoc to select ?
        // this.$explainVersionInput = $("<input>").attr("type", "number").attr("min", "0")
        //     .attr("step", "1").attr("value", "0")
        // $cb.append(this.$explainVersionInput)
    }
}
