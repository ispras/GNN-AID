class MenuExplainerView extends TabView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks, ["Load", "Construct"])
    }

    _setupViews() {
        this.views = [
            new MenuExplainerLoadView(this.divs[0], "el", []),
            new MenuExplainerInitView(this.divs[1], "ei", []),
        ]
    }
}
