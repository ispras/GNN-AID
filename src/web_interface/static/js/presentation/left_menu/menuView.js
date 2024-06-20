// Left menu
class MenuView extends View {
    constructor($div, requestBlock, listenBlocks, reuse=false) {
        super($div, requestBlock, listenBlocks)
        this.reuse = reuse // if true, hide elements instead of removing them

        // Properties
        this.$mainDiv = null
        this.$acceptDiv = null

        // Variables
        this.state = MVState.OFF
    }

    // Create elements, init block
    init(args) {
        if (!this.reuse || this.$mainDiv == null) { // create
            super.init(args)
            if (this.state !== MVState.OFF)
                return console.error("Can't init " + this.constructor.name + " in state ", this.state)

            this.state = MVState.ACTIVE
            this.$mainDiv = $("<div></div>").attr("id", this.idPrefix + '-main')
            this.$div.append(this.$mainDiv)
            this.$acceptDiv = $("<div></div>")
            this.$div.append(this.$acceptDiv)
        }
        else { // not the first time, reuse
            // Put all elements in initial condition
            this.state = MVState.ACTIVE
            this.$mainDiv.show()
            this.$acceptDiv.show()
            this.$acceptDiv.find('button').text('Accept')
            blockDiv(this.$mainDiv, false)
        }
    }

    // When this menu block is fixed by user
    async accept() {
        if (this._verbose)
            console.log(this.constructor.name + "[" + this.requestBlock + "].accept()")
        let ok = await this._accept()
        if (ok === -1) {
            blockLeftMenu(false)
            return -1
        }
        let errors = await Controller.blockRequest(this.requestBlock, 'submit')
        // If submit was successful, onSubmit() will be called
        // Otherwise do    blockLeftMenu(false)
    }

    // Construct config and request modify() of the block.
    // Returns -1 if configuration cannot be accepted
    async _accept() {
        // To be overridden in subclass
    }

    /**
     * When user wants to edit this menu block which is locked at the moment
     * @param toDefault - false - do not change config (soft unlock),
     * true - set config to default (hard unlock)
     */
    async unlock(toDefault) {
        if (this._verbose)
            console.log(this.constructor.name + "[" + this.requestBlock + "].unlock()")
        if (this.state === MVState.LOCKED) {
            // NOTE state will become ACTIVE at onUnlock()
            await Controller.blockRequest(
                this.requestBlock, 'unlock', {toDefault: toDefault})
            // blockDiv(this.$mainDiv, false)
        }
    }

    break() {
        super.break()
        this.state = MVState.OFF
        if (this.reuse) {
            this.$mainDiv.hide()
            this.$acceptDiv.hide()
        }
        else {
            this.$div.empty()
        }
    }

    // Called from within requestBlock init()
    onInit(block, args) {
        super.onInit(block, args)
        if (block === this.requestBlock) {
            this.init(args)
        }
    }

    // onModify(block, args) {
    //     super.onModify(block, args)
    // }

    // Called from within block.unlock()
    onUnlock(block, args) {
        super.onUnlock(block, args)
        if (block === this.requestBlock) {
            blockLeftMenu(false)
            blockDiv(this.$mainDiv, false)
            this.state = MVState.ACTIVE
        }
    }

    // Called from within block.break()
    onBreak(block) {
        super.onBreak(block)
        if (block === this.requestBlock && this.state !== MVState.OFF) // this block is broken
            this.break()
        // some other block or state - ignore, or define at subclass
    }

    // Called from within block.finalize() in case of success
    onSubmit(block, data) {
        super.onSubmit(block, data)
        blockLeftMenu(false)
        blockDiv(this.$mainDiv, true)
        this.state = MVState.LOCKED
    }

    appendAcceptBreakButtons() {
        let $cb = $("<div></div>").attr("class", "control-block")
        this.$acceptDiv.append($cb)

        let $accept = $("<button></button>").attr("id", this.idPrefix + "-accept").text("Accept")
        $cb.append($accept)

        // Buttons action listeners
        $accept.click(async () => {
            if (this.state === MVState.LOCKED) {
                console.log('pressed Edit')
                // TODO Lock all
                blockLeftMenu(true)
                await this.unlock() // TODO param can be added
                // Unlock all after onUnlock
                $accept.text('Accept')
            }
            else {
                console.log('pressed Accept')
                // Lock all
                blockLeftMenu(true)
                let res = await this.accept()
                // Unblock will be called after submit
                if (res === -1)
                    return
                $accept.text('Edit')
            }
        })
    }
}

// State of MenuView element: 'off' / 'active' / 'locked'
const MVState = Object.freeze({
    OFF: Symbol("off"),
    ACTIVE: Symbol("active"),
    LOCKED: Symbol("locked"),
})

function blockLeftMenu(on) {
    blockDiv($("#menu-left"), on, "wait")
}