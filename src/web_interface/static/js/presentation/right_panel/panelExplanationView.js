class PanelExplanationView extends PanelView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        this.explainer = null
        this.explanation = null

        this.init()
        this._collapse(true)
    }

    init() {
        super.init("Explanation panel")
        this.$div.css("background", '#ffd9fe')

        let $explainerDiv = $("<div></div>")
        this.$body.append($explainerDiv)
        this.explainer = new PanelExplainerInfoSubView(
            $explainerDiv, null, ["ei", "el"])
        this.explainer.init()

        let $explanationDiv = $("<div></div>")
        this.$body.append($explanationDiv)
        this.explanation = new PanelExplanationInfoSubView(
            $explanationDiv, null, ["el", "er"])
        this.explanation.init()
    }

    // Drop panel body contents
    break() {
        this.explainer.break()
        this.explanation.break()
        this._collapse(true)
    }
}

class PanelExplainerInfoSubView extends PanelView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        this.explainerPath = null
        this.explainerConfig = null
    }

    init() {
        super.init("Explainer configuration")
    }

    onSubmit(block, data) {
        super.onSubmit(block, data)
        if (block === "ei") {
            if ("config" in data)
                this.update(data["config"])
        }
        else if (block === "el") {
            if ("path" in data)
                this.update(data["path"])
        }
    }

    onUnlock(block, args) {
        super.onUnlock(block, args)
        if (block === "ei") {
            this.explainerPath = null
            this.explainerConfig = null
            this.break()
        }
    }

    // Update info
    update(dict) {
        this._updateArgs = arguments
        this._collapse(false)

        let html = JSON_stringify(dict, 2)
        html = '<pre>' + html.replaceAll('\n', '<br>') + '</pre>'
        this.$body.html(html)
    }

    break() {
        super.break()
        this.$body.html('No explainer specified')
    }
}

class PanelExplanationInfoSubView extends PanelView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        this.num_feats = null
        this.num_classes = null
        this.multi = null
        this.explanation = null
        this.$globalDiv = null // div for global explanation
        this.$localDiv = null // div for local explanation
    }

    init() {
        super.init("Explanation results")

        this.$body.empty()
        this.$globalDiv = $("<div></div>")
        this.$localDiv = $("<div></div>")
        this.$body.append($("<label></label>").html("<h2>Global</h2>"))
        this.$body.append(this.$globalDiv)
        this.$body.append($("<label></label>").html("<h2>Local</h2>"))
        this.$body.append(this.$localDiv)
    }

    async onInit(block, args) {
        await super.onInit(block, args)
        if (block === "el" || block === "er") {
            this.num_feats = args[0]
            this.multi = args[1]
        }
    }

    onUnlock(block, args) {
        super.onUnlock(block, args);
    }

    onSubmit(block, data) {
        super.onSubmit(block, data)
        if (block === "el") {
            if ("explanation_data" in data) {
                this.setExplanation(data["explanation_data"])
            }
        }
    }

    async onReceive(block, data) {
        super.onReceive(block, data)
        if (block === "er") {
            if ("explanation_data" in data) {
                await this.setExplanation(data["explanation_data"])
            }
        }
    }

    async setExplanation(explanationData) {
        let explanation
        switch (explanationData["info"]["type"]) {
            case "subgraph":
                explanation = new SubgraphExplanation(explanationData)
                break
            case "prototype":
                explanation = new PrototypeExplanation(explanationData)
                break
            case "string":
                explanation = new StringExplanation(explanationData)
                break
            default:
                console.error("Unknown explanation type")
        }
        this.explanation = explanation // fixme duplicates expl from datasetView
        await this.update()
    }

    // Update explanation info div
    async update() {
        this._updateArgs = arguments
        this._collapse(false)

        let $panel = this.explanation.info["local"] ? this.$localDiv : this.$globalDiv
        $panel.empty()
        await this.explanation.addInfo($panel, [this.num_feats, this.multi])
    }

    break() {
        // this.$body.html('No explanation')
        this.$globalDiv.html('No explanation')
        this.$localDiv.html('No explanation')
    }
}