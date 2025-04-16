class MenuAfterTrainView extends MenuView {
    static names = ["AD-ea", "AD-ma"]
    static leftMargin = 8
    static paramsAttackColor = "#ff8"

    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        // Variables
        this.availableMethods = null

        //Elements
        this.paramsBuilders = Object.fromEntries(MenuAfterTrainView.names.map(x => [x, null]))
        this.$checkboxes = Object.fromEntries(MenuAfterTrainView.names.map(x => [x, null]))
        this.$methodSelects = Object.fromEntries(MenuAfterTrainView.names.map(x => [x, null]))

        // Buttons
        this.$run = null
        // this.$reset = null
    }

    async init(arg) {
        super.init()
        // this.appendAcceptBreakButtons()

        this.availableMethods = arg
        await this.addConfigMenu()
    }

    _getConfigs() {
        // Form configs from selectors values where checkbox is checked
        let paramConfigs = {}
        for (let name of MenuAfterTrainView.names) {
            if (this.$checkboxes[name].is(':checked'))
                paramConfigs[name] = {
                    _class_name: this.$methodSelects[name].val(),
                    _config_kwargs: Object.assign({}, this.paramsBuilders[name].kwArgs)
                }
        }
        console.log("AD at paramConfigs", paramConfigs)
        return paramConfigs
    }

    async onrun() {
        let configs = this._getConfigs()
        this.$acceptDiv.find('button').prop("disabled", true)
        this.$run.prop("disabled", true)
        await controller.ajaxRequest('/model',
            {do: "run with attacks", configs: JSON_stringify(configs)})
        this.$run.prop("disabled", false)
        this.$acceptDiv.find('button').prop("disabled", false)
    }

    // Build buttons for model training process in model menu
    async addConfigMenu() {
        console.log('addConfigMenu')

        this.$mainDiv.append($("<label></label>").html("<h3>After training</h3>"))

        for (let name of MenuAfterTrainView.names) {
            let $cb = $("<div></div>").attr("class", "control-block")
                .css("padding-top", "12px")
            this.$mainDiv.append($cb)

            let checkId = this.idPrefix + "-AD-check-" + name
            this.$checkboxes[name] = $("<input>").attr("id", checkId)
                .attr("text", "Add " + name)
                .attr("type", "checkbox").prop('checked', false)
            $cb.append(this.$checkboxes[name])
            $cb.append($("<label></label>").html("add <b>" + {
                "AD-ea": "Evasion attack",
                "AD-ma": "MI attack",
            }[name] + "</b>").attr("for", checkId))

            let $methodDiv = $("<div></div>").hide()
            this.$mainDiv.append($methodDiv)

            this.$checkboxes[name].change(() => {
                if (this.$checkboxes[name].is(':checked'))
                    $methodDiv.show()
                else
                    $methodDiv.hide()
            })

            // Method selector
            let $methodSelectDiv = $("<div></div>").attr("class", "control-block")
            $methodDiv.append($methodSelectDiv)
            let id = this.idPrefix + "-AD-select-" + name
            $methodSelectDiv.append($("<label></label>").text("Method").attr("for", id))
            this.$methodSelects[name] = $("<select></select>").attr("id", id)
            $methodSelectDiv.append(this.$methodSelects[name])

            for (const key of this.availableMethods[name])
                this.$methodSelects[name].append($("<option></option>").text(key))

            // Method params
            let $methodParamsDiv = $("<div></div>")
                .css("margin-left", MenuAfterTrainView.leftMargin + "px")
                .css("background-color", name[1] === 'a' ? MenuAfterTrainView.paramsAttackColor : MenuBeforeTrainView.paramsDefenseColor)
            $methodDiv.append($methodParamsDiv)
            this.paramsBuilders[name] = new ParamsBuilder($methodParamsDiv,
                name, this.idPrefix + "-AD-param-" + name + '-')

            let self = this
            this.$methodSelects[name].change(async function () {
                self.paramsBuilders[name].drop()
                // await self.unlock(true) // Clear chosen values
                self.$acceptDiv.hide()
                await self.paramsBuilders[name].build(this.value)
                self.$acceptDiv.show()
            })

            // Select the first method from the list
            this.$methodSelects[name].val(this.availableMethods[name][0]).change()
        }

        let $cb = $("<div></div>").attr("class", "control-block")
        this.$mainDiv.append($cb)
        this.$run = $("<button></button>")
            .attr("id", "model-button-reset").text("Run with attacks")
            .css("margin-right", "5px")
            .attr("title", "Run model with all attacks and defenses specified earlier and compute metrics")
        $cb.append(this.$run)

        this.$run.click(async () => {
            this.onrun() // No await
        })

    }
}
