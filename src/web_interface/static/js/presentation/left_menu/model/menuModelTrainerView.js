class MenuModelTrainerView extends MenuView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        // Variables
        this.trainStepFlag = null // Flag whether train_1_step is available
        this.trainFullFlag = null // Flag whether train_full is available

        // Training params
        this.$epochsInput = null
        this.metricFlags = null

        // Buttons
        this.$run = null
        this.$reset = null
        this.$trainByOne = null
        this.$trainFull = null
        this.$save = null
    }

    async init(arg) {
        super.init()
        this.appendAcceptBreakButtons()
        // this.$acceptDiv.hide()

        this.trainStepFlag = arg["functions"].includes("train_1_step")
        this.trainFullFlag = arg["functions"].includes("train_full")

        await this.updateTrainerMenu()
    }

    onReceive(block, args) {
        // super.onReceive(block, args)
        if (block === 'mt') {
            if ("progress" in args) {
                let load = args["progress"]["load"]
                if (load) {
                    console.assert(load >= 0 && load <= 1)
                    this.progressBar.setLoad(load)
                }
                let text = args["progress"]["text"]
                if (text)
                    this.progressBar.setText(text)
            }
        }
    }

    async onrun(metrics) {
        this.$acceptDiv.find('button').prop("disabled", true)
        this.$run.prop("disabled", true)
        await Controller.ajaxRequest('/model',
            {do: "run", metrics: JSON_stringify(metrics)})
        this.$run.prop("disabled", false)
        this.$acceptDiv.find('button').prop("disabled", false)
    }

    async onreset() {
        this.$acceptDiv.find('button').prop("disabled", true)
        await Controller.ajaxRequest('/model', {do: "reset"})
        this.$acceptDiv.find('button').prop("disabled", false)
    }

    async ontrain(mode, steps, metrics) {
        this.$acceptDiv.find('button').prop("disabled", true)
        await Controller.ajaxRequest('/model',
            {do: "train", mode: mode, steps: steps, metrics: JSON_stringify(metrics)})
        this.$acceptDiv.find('button').prop("disabled", false)
    }

    async onstop() {
        await Controller.ajaxRequest('/model', {do: "stop"})
    }

    async onsave() {
        let path = await Controller.ajaxRequest('/model', {do: "save"})
        console.log("model saved at", path)
        // TODO Re-index models storage
    }

    // Build buttons for model training process in model menu
    async updateTrainerMenu() {
        console.log('updateModelManagerMenu')

        let $cb
        this.$mainDiv.append($("<label></label>").html("<h3>Training & metrics</h3>"))

        // run & reset Buttons
        $cb = $("<div></div>").attr("class", "control-block")
        this.$mainDiv.append($cb)
        this.$run = $("<button></button>")
            .attr("id", "model-button-reset").text("Run")
            .css("margin-right", "5px")
            .attr("title", "Run model on the dataset and compute metrics")
        $cb.append(this.$run)
        // $cb = $("<div></div>").attr("class", "control-block")
        // this.$mainDiv.append($cb)
        this.$reset = $("<button></button>")
            .attr("id", "model-button-reset").text("Reset")
            .css("margin-right", "5px")
            .attr("title", "Drop model weights and dataset train/test split, creating them newly random")
        $cb.append(this.$reset)

        // Epochs and metrics
        $cb = $("<div></div>").attr("class", "control-block")
            .css('display', 'block')
        this.$mainDiv.append($cb)
        // $cb.append($("<label></label>").text("Metrics:"))
        this.metricFlags = {"train": [], "val": [], "test": []}

        let $table = $("<table></table>").css("width", "100%")
        $table.addClass("display")
        $cb.append($table)
        let $thead = $("<thead></thead>")
        $table.append($thead)
        let $tr = $("<tr></tr>")
        $thead.append($tr)
        $tr.append($("<th>Metric</th>").css("width", "70%"))
        $tr.append($("<th>train</th>").css("writing-mode", "sideways-lr"))
        $tr.append($("<th>val</th>").css("writing-mode", "sideways-lr"))
        $tr.append($("<th>test</th>").css("writing-mode", "sideways-lr"))
        let $tbody = $("<tbody></tbody>")
        $table.append($tbody)
        let metrics = ["Accuracy", "BalancedAccuracy", "Precision", "Recall", "F1", "Jaccard"]
        for (const metric of metrics) {
            let $tr = $("<tr></tr>")
            $tbody.append($tr)
            $tr.append($("<td></td>").text(metric))
            for (const mask of ["train", "val", "test"]) {
                let id = "menu-model-constructor-metric-" + metric + '-' + mask
                // $cb.append($("<label></label>").text(metric).attr("for", id).css('margin-left', '8px'))
                let $metricFlag = $("<input>").attr("id", id).attr("name", metric)
                    .attr("type", "checkbox").prop("checked", false)
                this.metricFlags[mask].push($metricFlag)
                let $td = $("<td></td>")
                $tr.append($td)
                $td.append($metricFlag)
            }
        }
        this.metricFlags["train"][0].prop("checked", true)
        this.metricFlags["test"][0].prop("checked", true)

        $cb = $("<div></div>").attr("class", "control-block")
        this.$mainDiv.append($cb)
        $cb.append($("<label></label>").text("Epochs to train")) // TODO assoc to select ?
        this.$epochsInput = $("<input>").attr("type", "number").attr("min", "1")
            .attr("max", "3000").attr("step", "1").attr("value", "10")
        $cb.append(this.$epochsInput)

        // for (const metric of ["Accuracy", "BalancedAccuracy", "Precision", "Recall", "F1", "Jaccard"]) {
        //     let id = "menu-model-constructor-metric-" + metric
        //     $cb.append($("<label></label>").text(metric).attr("for", id).css('margin-left', '8px'))
        //     let $metricFlag = $("<input>").attr("id", id).attr("name", metric)
        //         .attr("type", "checkbox").prop("checked", false)
        //     this.metricFlags.push($metricFlag)
        //     $cb.append($metricFlag)
        // }
        // this.metricFlags[0].prop("checked", true)

        // train & save Buttons
        $cb = $("<div></div>").attr("class", "control-block")
        this.$mainDiv.append($cb)
        this.$trainByOne = $("<button></button>")
            .attr("id", "model-button-trainByOne").text("Train by 1 step")
            .css("margin-right", "5px")
            .attr("title", "Train the model using 'train_1_step' function by calling it number of epochs times")
            .prop("disabled", !this.trainStepFlag)
        $cb.append(this.$trainByOne)
        this.$trainFull = $("<button></button>")
            .attr("id", "model-button-trainFull").text("Train full")
            .css("margin-right", "5px")
            .attr("title", "Train model by calling 'train_full' function")
            .prop("disabled", !this.trainFullFlag)
        $cb.append(this.$trainFull)

        this.progressBar = new ProgressBar()
        this.progressBar.visible(true)
        this.$mainDiv.append(this.progressBar.$div)

        $cb = $("<div></div>").attr("class", "control-block")
        this.$mainDiv.append($cb)
        this.$save = $("<button></button>").text("Save")
            .attr("id", "model-save-constructor")
            .attr("title", "Save current model weights as a new version of the model")
        $cb.append(this.$save)

        this.$run.click(async () => {
            this.onrun(this.getMetrics()) // No await
        })

        this.$reset.click(async () => {
            this.$reset.prop("disabled", true)
            await this.onreset()
            this.$reset.prop("disabled", false)
            this.progressBar.drop()
        })

        let trainFunc = async ($button, text, mode) => {
            if (this.training != null) { // stop training - fixme
                $button.prop("disabled", true)
                await this.onstop()
                $button.prop("disabled", false)
                $button.text(text)
                this.$save.prop("disabled", false)
                this.$reset.prop("disabled", false)
                this.training = null
                // this.dropModel(true)
            }
            else { // run training
                $button.prop("disabled", true)
                this.$save.prop("disabled", true)
                this.$reset.prop("disabled", true)
                this.training = mode
                this.progressBar.visible(true)
                this.progressBar.start()
                blockLeftMenu(true)
                await this.ontrain(mode, parseInt(this.$epochsInput.val()), this.getMetrics())
                blockLeftMenu(false)
                $button.prop("disabled", false)
                this.$save.prop("disabled", false)
                this.$reset.prop("disabled", false)
                this.training = null
            }
        }

        this.$trainByOne.click(() => trainFunc(this.$trainByOne, "Train by 1 step", '1_step'))
        this.$trainFull.click(() => trainFunc(this.$trainFull, "Train full", 'full'))

        this.$save.click(async () => await this.onsave())
    }

    getMetrics() {
        let metrics = []
        for (const [mask, flags] of Object.entries(this.metricFlags))
            for (const $metricFlag of flags)
                if ($metricFlag.is(':checked'))
                    metrics.push({mask: mask, name: $metricFlag.attr("name")})
        return metrics
    }
}
