class MenuDatasetView extends MenuView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        this.tuplePrefixStorage = null
    }

    async init() {
        super.init()

        // Start with dataset config
        let [ps, info] = await controller.ajaxRequest('/dataset', {get: "index"})
        // this.prefixStorage = PrefixStorage.fromJSON(ps)
        this.tuplePrefixStorage = TuplePrefixStorage.fromJSON(ps)

        this.$mainDiv.append($("<h3></h3>").text("Choose raw data"))

        let drop = () => {
            this.$acceptDiv.hide()
            // this.break()
        }

        let set = async () => {
            this.$acceptDiv.show()
            // await this.accept()
        }
        this.tuplePrefixStorage.buildCascadeMenu(this.$mainDiv, drop, set)

        this.appendAcceptBreakButtons()
        this.$acceptDiv.hide()
    }

    async _accept() {
        let dc = {'full_name': this.tuplePrefixStorage.getConfig()}
        await controller.blockRequest(this.requestBlock, 'modify', dc)
    }
}
