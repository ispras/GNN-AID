class MenuModelView extends TabView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks, ["Load", "Construct", "Custom"])
    }

    _setupViews() {
        this.views = [
            new MenuModelLoadView(this.divs[0], "mload", []),
            new MenuModelConstructorView(this.divs[1], "mconstr", []),
            new MenuModelCustomView(this.divs[2], "mcustom", []),
        ]
    }
}
