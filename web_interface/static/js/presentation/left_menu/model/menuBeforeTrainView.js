class MenuBeforeTrainView extends MenuView {
    static names = ["AD-pa", "AD-pd", "AD-ed", "AD-md"]
    static leftMargin = 8
    static paramsAttackColor = "#ff8"
    static paramsDefenseColor = "#bfb"

    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        // Variables
        this.availableMethods = null
        this.isEdgeLevel = null

        //Elements
        this.paramsBuilders = Object.fromEntries(MenuBeforeTrainView.names.map(x => [x, null]))
        this.$checkboxes = Object.fromEntries(MenuBeforeTrainView.names.map(x => [x, null]))
        this.$methodSelects = Object.fromEntries(MenuBeforeTrainView.names.map(x => [x, null]))
    }

    async init(args) {
        super.init()
        this.appendAcceptBreakButtons()

        this.availableMethods = args[0]
        this.isEdgeLevel = args[1]
        await this.addConfigMenu()
    }

    async _accept() {
        // Form configs from selectors values where checkbox is checked
        let paramConfigs = {}
        for (let name of MenuBeforeTrainView.names) {
            if (this.$checkboxes[name].is(':checked'))
                paramConfigs[name] = {
                    _class_name: this.$methodSelects[name].val(),
                    _config_kwargs: Object.assign({}, this.paramsBuilders[name].kwArgs)
                }
        }
        console.log("AD paramConfigs", paramConfigs)
        await controller.blockRequest(this.requestBlock, 'modify', paramConfigs)
    }

    // Build buttons for model training process in model menu
    async addConfigMenu() {
        console.log('addConfigMenu')

        this.$mainDiv.append($("<label></label>").html("<h3>Before training</h3>"))

        for (let name of MenuBeforeTrainView.names) {
            let $cb = $("<div></div>").attr("class", "control-block")
                .css("padding-top", "12px")
            this.$mainDiv.append($cb)

            let checkId = this.idPrefix + "-AD-check-" + name
            this.$checkboxes[name] = $("<input>").attr("id", checkId)
                .attr("text", "Add " + name)
                .attr("type", "checkbox").prop('checked', false)
            $cb.append(this.$checkboxes[name])
            $cb.append($("<label></label>").html("add <b>" + {
                "AD-pa": "Poison attack",
                "AD-pd": "Poison defense",
                "AD-ed": "Evasion defense",
                "AD-md": "MI defense",
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
                .css("margin-left", MenuBeforeTrainView.leftMargin + "px")
                .css("background-color", name[1] === 'a' ? MenuBeforeTrainView.paramsAttackColor : MenuBeforeTrainView.paramsDefenseColor)
            $methodDiv.append($methodParamsDiv)
            this.paramsBuilders[name] = new ParamsBuilder($methodParamsDiv,
                name, this.idPrefix + "-AD-param-" + name + '-')

            // For params builder, to replace param name "node" with "edge" for edge task
            let localPostFunction = (algorithm) => {
                if ("element_idx" in this.paramsBuilders[name].selectors) {
                    let elemName = "Node"
                    if (this.multi)
                        elemName = "Graph"
                    if (this.isEdgeLevel)
                        elemName = "Pair"
                    this.paramsBuilders[name].renameParam("element_idx", elemName)
                }
            }

            let self = this
            this.$methodSelects[name].change(async function () {
                self.paramsBuilders[name].drop()
                // await self.unlock(true) // Clear chosen values
                self.$acceptDiv.hide()
                await self.paramsBuilders[name].build(this.value, localPostFunction, self.isEdgeLevel)
                self.$acceptDiv.show()
            })

            // Select the first method from the list
            this.$methodSelects[name].val(this.availableMethods[name][0]).change()
        }
    }
}
