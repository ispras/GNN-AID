class PanelModelView extends PanelView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        this.conf = null // Description of configuration
        this.arch = null // Interactive picture of structure
        this.stat = null // Metrics and plots

        this.init()
        this._collapse(true)
    }

    init() {
        super.init("Model panel")
        this.$div.css("background", '#e1ffcb')

        let $infoDiv = $("<div></div>")
        this.conf = new PanelModelConfigView($infoDiv,
            "mc", ["mconstr", "mload", "mcustom", "mmc"])
        this.$body.append($infoDiv)
        this.conf.init()

        let $archDiv = $("<div></div>")
        this.$body.append($archDiv)
        this.arch = new PanelModelArchView($archDiv,
            "mc", ["mconstr", "mload", "mcustom", "mmc", "mt"])
        this.arch.init()

        let $statDiv = $("<div></div>")
        this.$body.append($statDiv)
        this.stat = new PanelModelStatView($statDiv,
            "mmc", ["mconstr", "mload", "mcustom", "mmc", "mt"])
        this.stat.init()
    }

    // Update model info string
    update(type, config) {
        this._updateArgs = arguments
        if (this.collapsed) {
            return
        }
        let html = JSON.stringify(config, null, 2)
        html = '<pre>' + html.replaceAll('\n', '<br>') + '</pre>'
        this.conf.update(type, html)
    }

    break() {
        this.conf.break()
        this.stat.break()
        this.arch.break()
    }
}