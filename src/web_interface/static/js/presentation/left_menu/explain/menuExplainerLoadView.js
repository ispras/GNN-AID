class MenuExplainerLoadView extends MenuView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        this.$contentDiv = null
        this.prefixStorage = null
        this.info = null
    }

    async init(args) {
        super.init(args)

        if (typeof args[2] === "string") {
            this.$mainDiv.html(args[2])
        }
        else {
            this.prefixStorage = PrefixStorage.fromJSON(args[2][0])
            this.info = JSON_parse(args[2][1])

            // Add refresh button
            let $div = $("<div></div>").css("display", "flex")
            this.$mainDiv.append($div)
            let $refreshButton = $("<button></button>")
                .attr("class", "load-update").css("margin-left", "5px")
            $div.append($("<h3></h3>").text("Choose explanation"))
            $div.append($refreshButton)
            $refreshButton.append($("<img>")
                .attr("src", "../static/icons/refresh-svgrepo-com.svg")
                .attr("alt", "Refresh").attr("height", "18px").css("align-self", "center"))
            $refreshButton.attr("title", "Rescan saved explanations")
            $refreshButton.click(async () => await this._refresh($refreshButton))

            this.$contentDiv = $("<div></div>")
            this.$mainDiv.append(this.$contentDiv)

            this._build()

            this.appendAcceptBreakButtons()
            this.$acceptDiv.hide()
        }
    }

    async _accept() {
        let ep = this.prefixStorage.getConfig()
        await Controller.blockRequest(this.requestBlock, 'modify', ep)
    }

    // Build cascade menu
    _build() {
        let drop = () => this.$acceptDiv.hide()
        let set = async () => this.$acceptDiv.show()
        this.$contentDiv.empty()
        this.prefixStorage.buildCascadeMenu(this.$contentDiv, drop, set, this.info)
    }

    async _refresh() {
        blockDiv(this.$div, true)
        let [ps, info] = await Controller.ajaxRequest('/explainer', {do: "index"})
        this.prefixStorage = PrefixStorage.fromJSON(ps)
        this.info = JSON_parse(info)
        this._build()
        blockDiv(this.$div, false)
    }
}
