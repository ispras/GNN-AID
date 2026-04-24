class MenuDatasetView extends MenuView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        this.tuplePrefixStorage = null
    }

    async init() {
        super.init()

        this.$mainDiv.append($("<h3></h3>").text("Choose raw data"))

        let $uploadBtn = $("<button></button>").text("Upload")
        $uploadBtn.click(async () => {
            const dialog = new DatasetUploadDialog()
            dialog.onSuccess = this._buildDatasetsIndex.bind(this)
            dialog.show()
        })
        this.$mainDiv.append($("<div></div>").attr("class", "control-block").append($uploadBtn))

        // Start with dataset config

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

        this._drop = () => {
            $selected.val('not selected')
            this.$acceptDiv.hide()
            // this.break()
        }
        this._set = async () => {
            let text = this.tuplePrefixStorage.getConfig()
            $selected.val(text.join('/'))
            this.$acceptDiv.show()
            // await this.accept()
        }

        await this._buildDatasetsIndex()

        this.appendAcceptBreakButtons()
        this.$acceptDiv.hide()
    }

    async _buildDatasetsIndex() {
        // console.log('_buildDatasetsIndex')
        let [ps, info] = await controller.ajaxRequest('/dataset', {get: "index"})
        this.tuplePrefixStorage = PrefixStorage.fromJSON(ps)

        this.$optionsDiv.empty()
        this.tuplePrefixStorage.buildPopupMenu(this.$optionsDiv, this._drop, this._set)
    }

    async _accept() {
        let dc = {'full_name': this.tuplePrefixStorage.getConfig()}
        await controller.blockRequest(this.requestBlock, 'modify', dc)
    }
}
