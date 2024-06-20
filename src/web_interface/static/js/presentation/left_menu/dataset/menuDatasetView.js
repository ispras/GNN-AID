class MenuDatasetView extends MenuView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        this.prefixStorage = null
    }

    async init() {
        super.init()

        // Start with dataset config
        let [ps, info] = await Controller.ajaxRequest('/dataset', {get: "index"})
        this.prefixStorage = PrefixStorage.fromJSON(ps)

        this.$mainDiv.append($("<h3></h3>").text("Choose raw data"))

        let drop = () => {
            this.$acceptDiv.hide()
            // this.break()
        }

        let set = async () => {
            this.$acceptDiv.show()
            // await this.accept()
        }
        this.prefixStorage.buildCascadeMenu(this.$mainDiv, drop, set)

        this.appendAcceptBreakButtons()
        this.$acceptDiv.hide()
    }

    async _accept() {
        let dc = this.prefixStorage.getConfig()
        await Controller.blockRequest(this.requestBlock, 'modify', dc)
    }
}
