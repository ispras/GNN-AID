// View wrapper for several OR views with a header of tabs
class TabView extends MenuView {
    constructor($div, requestBlock, listenBlocks, names) {
        super($div, requestBlock, listenBlocks, true)

        this.divs = [] // list of div containing views
        this._groupClass = this.idPrefix + '-tab'
        this.$tabDiv = $("<div></div>").addClass("tab")
        this.$subviewsDiv = $("<div></div>")
        for (let i = 0; i < names.length; i++) {
            let $button = $("<button></button>")
                .addClass("tablinks").addClass(this._groupClass)
                .text(names[i])
            this.$tabDiv.append($button)
            $button.click((e) => this.switchTab(e, i))
            // this.views.push(views[i])
            let $aDiv = $("<div></div>").hide()
                .attr("id", this.idPrefix + '-' + nameToId(names[i]))
            this.$subviewsDiv.append($aDiv)
            this.divs.push($aDiv)
        }

        // Define sub-views
        this._setupViews()
        console.assert(this.views.length === names.length)

        for (const view of this.views) {
            view.unlock = async (toDefault) => {
                // TODO can we simplify // just a copy
                // console.log(this.constructor.name + "[" + this.requestBlock + "].unlock()")
                if (view.state === MVState.LOCKED) {
                    // NOTE state will become ACTIVE at onUnlock()
                    await Controller.blockRequest(
                        view.requestBlock, 'unlock', {toDefault: toDefault})
                    blockDiv(view.$mainDiv, false)
                }
                // Then call unlock
                await this.unlock(toDefault)
            }
        }

        // Variables
        this._openIx = null // index of open tab
    }

    _setupViews() {
        console.error('Must be implemented in subclass')
    }

    init(args) {
        let firstTime = !this.reuse || this.$mainDiv == null
        super.init(args)
        if (firstTime) { // create
            this.$mainDiv.append(this.$tabDiv)
            // Additional div which is controlled by subviews
            this.$div.append(this.$subviewsDiv)
        }
    }

    break() {
        super.break()
        if (this._openIx !== null) {
            this.divs[this._openIx].hide()
            // this.onhide(this._openIx)
        }
        $('.tablinks.' + this._groupClass).removeClass("active")
        this._openIx = null
    }

    async unlock(toDefault) {
        if (this.state === MVState.LOCKED) {
            // await this.views[this._openIx].unlock(toDefault)
            blockDiv(this.$tabDiv, false) // NOTE: tabs block only
            this.state = MVState.ACTIVE
        }
        else
            console.error('Expected to be locked, but state is', this.state)
    }

    onshow(ix) {
        // console.log('onshow', ix)
    }

    onhide(ix) {
        // console.log('onhide', ix)
    }

    switchTab(event, ix) {
        if (ix === this._openIx) return
        $('.tablinks.' + this._groupClass).removeClass("active")
        event.currentTarget.className += " active"

        if (this._openIx !== null) {
            this.divs[this._openIx].hide()
            this.onhide(this._openIx)
        }
        this._openIx = ix
        this.divs[ix].show()
        this.onshow(ix)
    }
}
