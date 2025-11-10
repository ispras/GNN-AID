class PanelModelConfigView extends PanelView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        this.inited = false
    }

    init() {
        super.init("Configuration")
    }

    _init() {
        this.$body.empty()
        this.$structureDiv = $("<div></div>")
        this.$body.append($("<label></label>").html("<h4>Structure</h4>"))
        this.$body.append(this.$structureDiv)
        this.$managerDiv = $("<div></div>")
        this.$body.append($("<label></label>").html("<h4>Manager parameters</h4>"))
        this.$body.append(this.$managerDiv)
        this.$modificationDiv = $("<div></div>")
        this.$body.append($("<label></label>").html("<h4>Modification</h4>"))
        this.$body.append(this.$modificationDiv)
        this.$epochsDiv = $("<div></div>")
        this.$body.append($("<label></label>").html("<h4>Training</h4>"))
        this.$body.append(this.$epochsDiv)
        this.inited = true
    }

    onSubmit(block, data) {
        super.onSubmit(block, data)
        if (!this.inited)
            this._init()
        if (block === "mconstr" || block === "mload" || block === "mcustom" || block === "mmc") {
            for (const [type, value] of Object.entries(data)) {
                let html = JSON_stringify(value, 2)
                html = '<pre>' + html.replaceAll('\n', '<br>') + '</pre>'
                this.update(type, html)
            }
        }
    }

    // Update model info string
    update(type, html) {
        let $div = {
            architecture: this.$structureDiv,
            manager: this.$managerDiv,
            modification: this.$modificationDiv,
            epochs: this.$epochsDiv,
        }[type]
        if ($div != null)
            // this.$structureDiv.html('')
            // this.$managerDiv.html('')
            // this.$modificationDiv.html('')
            // this.$epochsDiv.html('')
        // }
        // else
            $div.html(html)
    }

    break() {
        super.break()
        this.inited = false
        this.$body.html('No Model specified')
    }
}