/**
 * Manages how the data are drawn at the screen and shown to user.
 * It is controlled by Container or Controller.
 */
class Presenter {
    constructor() {
        this.blockListeners = {}
    }

    // Add all views on the page and their dependencies
    createViews() {
        window.addEventListener('contextmenu', (event) => event.preventDefault())

        // Should go before datasetView, since it uses inputs
        this.visualsView = new VisualsView(
            $("#menu-visuals-view"), null, ["dc", "dvc"])

        this.menuDatasetView = new MenuDatasetView(
            $("#menu-dataset-view"), "dc", [])

        this.menuDatasetVarView = new MenuDatasetVarView(
            $("#menu-dataset-var-view"), "dvc", [])

        this.datasetView = new DatasetView(
            $("#dataset-graph-view"), null,
            ["dc", "dvc", "mmc", "mt", "ei", "er", "el"])

        this.modelView = new MenuModelView(
            $("#menu-model-view"), "mc", [])

        this.modelManagerView = new MenuModelManagerView(
            $("#menu-model-manager-view"), "mmc", [])

        this.modelTrainerView = new MenuModelTrainerView(
            $("#menu-model-trainer-view"), "mt", [])

        this.menuExplainerView = new MenuExplainerView(
            $("#menu-explainer-view"), "e", [])

        this.explainerRunView = new MenuExplainerRunView(
            $("#menu-explainer-run-view"), "er", [])

        this.panelDatasetView = new PanelDatasetView(
            $("#panel-dataset-view"), "dc", ["dvc"])

        this.panelModelView = new PanelModelView(
            $("#panel-model-view"), "mc", ["mt"])

        this.panelExplanationView = new PanelExplanationView(
            $("#panel-explanation-view"), "e", ["ei", "er"])
    }

    // View listen to its requestBlock and listenBlocks
    addListener(view) {
        for (const block of view.listenBlocks) {
            if (!(block in this.blockListeners))
                this.blockListeners[block] = []
            this.blockListeners[block].push(view)
        }
        let block = view.requestBlock
        if (block) {
            if (!(block in this.blockListeners))
                this.blockListeners[block] = []
            this.blockListeners[block].push(view)
        }
    }
}


/**
 * Presentation for one logical block
 */
class View {
    constructor($div, requestBlock, listenBlocks=[]) {
        this.$div = $div
        this.requestBlock = requestBlock
        this.listenBlocks = listenBlocks
        controller.presenter.addListener(this)

        this.idPrefix = this.$div.attr("id")
        this._verbose = true
    }

    // Create this view when the requestBlock is initialized
    init(args) {
        if (this._verbose)
            console.log(this.constructor.name + "[" + this.requestBlock + "].init()")
    }

    // Remove this view since the requestBlock is undefined
    break() {
        if (this._verbose)
            console.log(this.constructor.name + "[" + this.requestBlock + "].break()")
    }

    // Called from within block.init()
    async onInit(block, args) {
        if (this._verbose)
            console.log(this.constructor.name + "[" + this.requestBlock + "].onInit(block=" + block + ")")
    }

    // Called from within block.modify()
    onModify(block, args) {
        if (this._verbose)
            console.log(this.constructor.name + "[" + this.requestBlock + "].onModify(block=" + block + ")")
    }

    // Called from within block.unlock()
    onUnlock(block) {
        if (this._verbose)
            console.log(this.constructor.name + "[" + this.requestBlock + "].onUnlock(block=" + block + ")")
    }

    // Called from within block.break()
    onBreak(block) {
        if (this._verbose)
            console.log(this.constructor.name + "[" + this.requestBlock + "].onBreak(block=" + block + ")")
    }

    // Called from within block.submit() if it was successful
    onSubmit(block, data) {
        if (this._verbose)
            console.log(this.constructor.name + "[" + this.requestBlock + "].onSubmit(block=" + block + ")")
    }

    // Called when received a message not falling in any other category
    onReceive(block, data) {
        if (this._verbose)
            console.log(this.constructor.name + "[" + this.requestBlock + "].onReceive(block=" + block + ")")
    }
}

// Block/unblock html elements.
// If called several times with on=true, it will be blocked until called same times with on=false
function blockDiv($div, on, cursor="notAllowed") {
    // let has = $div.hasClass("disabled")
    let _blockCounter = $div.prop('_blockCounter')
    if (_blockCounter == null)
        _blockCounter = 0
    _blockCounter += on ? 1 : -1
    if (_blockCounter < 0 && !on) {
        // fixme this occurs when e.g. mload accept is pressed (1 block) and then mc & mload call
        //  onSubmit which leads to 2 unblocks
        console.error('too many unblocks!')
        return
    }
    // console.log('block', on, _blockCounter, $div)
    $div.prop('_blockCounter', _blockCounter)

    if (_blockCounter > 0) {
        $div.addClass("disabled")
        $div.addClass(cursor)
        // FIXME all children recursively ?
        $div.find('*').onkeydown = (e) => {
            if (e.keyCode !== 9) {
                e.returnValue = false
                return false
            }
        }
        $div.blur()
    }
    else {
        $div.removeClass("disabled")
        $div.removeClass(cursor)
        $div.find('*').onkeydown = null
        // $div.focus()
    }
}
