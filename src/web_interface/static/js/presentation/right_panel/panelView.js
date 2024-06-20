// Info panels on the right side
class PanelView extends View {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        this.$div.addClass("info-panel")
        this.$body = null
        this.$label = null
        this.collapsed = false // whether div is collapsed
        this._updateArgs = null // arguments for update(), we keep them until panel is collapsed
    }

    // Create header and body DIVs - called once
    init(title) {
        this.$div.empty()
        let $headDiv = $("<div></div>").addClass("info-panel-header")
        this.$div.append($headDiv)
        this.$label = $("<label></label>").text(this.collapsed ? "+" : "-").css("margin-right", "5px")
        $headDiv.append(this.$label)
        if (title)
            $headDiv.append(`<b>${title}</b>`)
        this.$body = $("<div></div>").addClass("info-panel-body")
        this.$div.append(this.$body)
        $headDiv.click(() => this._collapse(!this.collapsed))

        this._collapse(false)
    }

    // Called each time when need to show panel
    _init() {}

    _collapse(collapsed) {
        if (this.collapsed === collapsed) return
        console.log(this.constructor.name + '.collapse(' + collapsed + ')')
        let f = parseInt(this.$div.css("flex-grow"))
        this.collapsed = collapsed
        if (this.collapsed === (f === 1)) {
            this.$div.animate({"flex-grow": 1 - f}, {duration: 400})
            this.$label.text(this.collapsed ? "+" : "-")
        }
        // else - Last animation is not finished yet, just leave it
        // console.assert(this.collapsed === (f === 1))
        this.collapsed ? this.$body.hide() : this.$body.show()
    }

    onSubmit(block, data) {
        super.onSubmit(block, data)
        if (block === this.requestBlock) {
            this._collapse(false)
        }
    }

    onUnlock(block, args) {
        super.onUnlock(block, args)
        if (block === this.requestBlock)
            this.break()
    }

    // Update body contents
    update() {
    }

    // Drop panel body contents
    break() {
        if (this.$body)
            this.$body.empty()
        this.datasetInfo = null
        this._collapse(true)
    }

    // // Show/hide the panel
    // visible(show) {
    //     show ? this.$div.show() : this.$div.hide()
    // }
}
