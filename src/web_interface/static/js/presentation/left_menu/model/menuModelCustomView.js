class MenuModelCustomView extends MenuView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

    }

    init(args) {
        super.init(args)
        this.prefixStorage = PrefixStorage.fromJSON(args[0])
        this.info = JSON_parse(args[1])

        this.$mainDiv.append($("<h3></h3>").text("Choose custom model"))

        this.$contentDiv = $("<div></div>")
        this.$mainDiv.append(this.$contentDiv)

        this._build()

        this.appendAcceptBreakButtons()
        this.$acceptDiv.hide()
    }

    async _accept() {
        await Controller.blockRequest(this.requestBlock, 'modify',
            this.prefixStorage.getConfig())
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
        let [ps, info] = await Controller.ajaxRequest('/model', {do: "index", type: "custom"})
        this.prefixStorage = PrefixStorage.fromJSON(ps)
        this.info = JSON_parse(info)
        this._build()
        blockDiv(this.$div, false)
    }
}
