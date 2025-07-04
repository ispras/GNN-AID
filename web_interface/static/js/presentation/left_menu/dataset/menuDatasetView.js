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

        this.$optionsDiv = $("<div></div>")
            .attr("style", "border: 2px solid #bbb; background: #eee;")
        this.$selectedDiv = $("<div></div>")
            .css("user-select", "none")
            .css("display", "grid")
            .css("padding", "3px")
        this.$mainDiv.append(this.$optionsDiv)
        this.$mainDiv.append(this.$selectedDiv)

        this.$selectedDiv.append($("<label></label>").text("Selected:"))
        let $selected = $("<input>").attr("type", "text").val('not selected')
            .attr("readonly", true)
        this.$selectedDiv.append($selected)

        let drop = () => {
            $selected.val('not selected')
            this.$acceptDiv.hide()
            // this.break()
        }
        let set = async () => {
            let text = this.tuplePrefixStorage.getConfig()
            $selected.val(text.join('/'))
            this.$acceptDiv.show()
            // await this.accept()
        }
        this.tuplePrefixStorage.buildCascadeMenu(this.$optionsDiv, drop, set)

        this.appendAcceptBreakButtons()
        this.$acceptDiv.hide()
    }

    async _accept() {
        let dc = {'full_name': this.tuplePrefixStorage.getConfig()}
        await controller.blockRequest(this.requestBlock, 'modify', dc)
    }
}
