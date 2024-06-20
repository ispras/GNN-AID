class VisualsView extends View {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        this._isMulti = null

        // single graphs
        this.$singleDiv = $("<div></div>").hide()
            .addClass(this.idPrefix + '-class').addClass('single')
        this.$div.append(this.$singleDiv)
        this.showModeId = 'show-mode'
        this.singleClassAsColorId = 'single-classAsColor'
        // this.layoutStopButtonClass = "layout-stop-button-class"
        this._addShowSingle()

        this.singleGraphLayoutId = "single-graph-layout"
        this.singleGraphEdgesId = "single-graph-edges"
        this.singleGraphLayoutFreezeId = "single-graph-layout-freeze"
        this._addGraph()
        this.singleNeighNodeId = "single-neigh-node"
        this.singleNeighLayoutId = "single-neigh-layout"
        this.singleNeighDepthId = "single-neigh-depth"
        this.singleNeighPartsIds = {}
        this.singleNeighLayoutFreezeId = "single-neigh-layout-freeze"
        this._addNeighborhood()

        // multi graphs
        this.$multiDiv = $("<div></div>").hide()
            .addClass(this.idPrefix + '-class').addClass('multi')
        this.$div.append(this.$multiDiv)
        this.multiNodeTypeAsColorId = 'multi-nodeTypeAsColor'
        this.multiCountId = "multi-count"
        this.multiArrangeId = "multi-arrange"
        this.multiLayoutId = "multi-layout"
        this.multiGraphId = "multi-graph"
        this.multiDepthId = "multi-depth"
        this.multiLayoutFreezeId = "multi-neigh-layout-freeze"
        this._addShowMulti()

        // common
        this.satellitesIds = {}
        this._addSatellites()

        // Listeners for config updates
        this.listeners = {} // key -> {tag -> set of listeners}
    }

    // Show mode
    _addShowSingle() {
        let $cb = $("<div></div>").attr("class", "control-block")
        this.$singleDiv.append($cb)
        let $label = $("<label></label>").text("Show")
        $cb.append($label)
        let $showSelect = $("<select></select>")
            .attr("id", this.idPrefix + '-' + this.showModeId)
        $cb.append($showSelect)
        $showSelect.append($("<option></option>").val("neighborhood").text("neighborhood"))
        $showSelect.append($("<option></option>").val("whole-graph").text("whole graph"))

        $showSelect.change((e) => {
            this.$neighborhoodDiv.hide()
            this.$graphDiv.hide()
            let val = $showSelect.val()
            if (val === "neighborhood") {
                this.$neighborhoodDiv.show()
            }
            else if (val === "whole-graph") {
                this.$graphDiv.show()
            }
            else
                console.error("Not implemented")
            this._update(this.showModeId, val, true, $showSelect)
        })

    }

    _addShowMulti() {
        let $cb = $("<div></div>").attr("class", "control-block")
        this.$multiDiv.append($cb)
        let $label = $("<label></label>").text("Show")
        $cb.append($label)
        let $countSelect = $("<select></select>").attr("id", this.idPrefix + '-' + this.multiCountId)
        $cb.append($countSelect)
        $countSelect.append($("<option></option>").val("several").text("several graphs"))
        $countSelect.append($("<option></option>").val("all").text("all graphs"))
        $countSelect.change((e) => {
            let val = $countSelect.val()
            // this.setEnabled(this.multiArrangeId, val === "all graphs")
            this.setEnabled(this.multiGraphId, val !== "all")
            this.setEnabled(this.multiDepthId, val !== "all")
            this._update(this.multiCountId, val, true, $countSelect)
        })

        $cb = $("<div></div>").attr("class", "control-block")
        this.$multiDiv.append($cb)
        $label = $("<label></label>").text("Graph index")
        $cb.append($label)
        let $graphInput = $("<input>").attr("type", "number")
            .attr("min", "0").attr("step", "1")
            .attr("id", this.idPrefix + '-' + this.multiGraphId)
        $cb.append($graphInput)
        $graphInput.change((e) => this._update(
            this.multiGraphId, $graphInput.val(), true, $graphInput))

        $cb = $("<div></div>").attr("class", "control-block")
        this.$multiDiv.append($cb)
        $label = $("<label></label>").text("Nearby graphs")
        $cb.append($label)
        let $depthInput = $("<input>").attr("type", "number")
            .attr("min", "0").attr("max", MultipleGraphs.MAX_DEPTH)
            .attr("step", "1")
            .attr("id", this.idPrefix + '-' + this.multiDepthId)
        $cb.append($depthInput)
        $depthInput.change((e) => {
            let val = $depthInput.val()
            this.setEnabled(this.multiArrangeId, val > 0)
            this._update(this.multiDepthId, val, true, $depthInput)
        })

        $cb = $("<div></div>").attr("class", "control-block")
        this.$multiDiv.append($cb)
        $label = $("<label></label>").text("Arrange")
        $cb.append($label)
        let $arrangeSelect = $("<select></select>").attr(
            "id", this.idPrefix + '-' + this.multiArrangeId)
        $cb.append($arrangeSelect)
        $arrangeSelect.append($("<option></option>").val("grid").text("grid"))
        $arrangeSelect.append($("<option></option>").val("vertical").text("vertical"))
        $arrangeSelect.append($("<option></option>").val("free").text("free"))
        $arrangeSelect.change((e) => this._update(this.multiArrangeId, $arrangeSelect.val()))

        $cb = $("<div></div>").attr("class", "control-block")
        this.$multiDiv.append($cb)
        $label = $("<label></label>").text("Layout")
        $cb.append($label)
        let $layoutSelect = $("<select></select>").attr(
            "id", this.idPrefix + '-' + this.multiLayoutId)
        $cb.append($layoutSelect)
        $layoutSelect.append($("<option></option>").val("random").text("random"))
        $layoutSelect.append($("<option></option>").val("force").text("force"))
        $layoutSelect.change((e) => this._update(this.multiLayoutId, $layoutSelect.val()))
        $cb.append(this._createLayoutStopButton(this.multiLayoutFreezeId))
    }

    _createLayoutStopButton(id) {
        let $button = $("<button></button>")
            .attr("id", this.idPrefix + '-' + id)
            .css("margin-left", '8px')
        $button._value = false
        $button.append($("<img>")
            .attr("src", "../static/icons/play-svgrepo-com.svg")
            .attr("height", "14px").hide())
        $button.append($("<img>")
            .attr("src", "../static/icons/pause-svgrepo-com.svg")
            .attr("height", "16px"))
        $button.click((e) => {
            $button._value = !$button._value
            this._update(id, $button._value)
            let ix = $button._value ? 0 : 1
            $button.children().eq(ix).show()
            $button.children().eq(1-ix).hide()
        })
        $button.change((e) => this._update(id, $button._value))

        return $button
    }

    _addNeighborhood() {
        this.$neighborhoodDiv = $("<div></div>").attr("id", this.idPrefix + '-single-neigh').hide()
        this.$singleDiv.append(this.$neighborhoodDiv)

        let $cb = $("<div></div>").attr("class", "control-block")
        this.$neighborhoodDiv.append($cb)
        let $label = $("<label></label>").text("Node")
        $cb.append($label)
        let $nodeInput = $("<input>").attr("type", "number")
            .attr("min", "0").attr("step", "1")
            .attr("id", this.idPrefix + '-' + this.singleNeighNodeId)
        $cb.append($nodeInput)
        $nodeInput.change(async (e) => await this._update(
            this.singleNeighNodeId, $nodeInput.val(), true, $nodeInput))

        $cb = $("<div></div>").attr("class", "control-block")
        this.$neighborhoodDiv.append($cb)
        $label = $("<label></label>").text("Neighborhood depth")
        $cb.append($label)
        let $depthSelect = $("<input>").attr("type", "number")
            .attr("min", "0").attr("max", Neighborhood.MAX_DEPTH)
            .attr("step", "1")
            .attr("id", this.idPrefix + '-' + this.singleNeighDepthId)
        $cb.append($depthSelect)
        $depthSelect.change((e) => this._update(
            this.singleNeighDepthId, $depthSelect.val(), true, $depthSelect))

        $cb = $("<div></div>").attr("class", "control-block")
        this.$neighborhoodDiv.append($cb)
        $label = $("<label></label>").text("Layout")
        $cb.append($label)
        let $select = $("<select></select>").attr(
            "id", this.idPrefix + '-' + this.singleNeighLayoutId)
        $cb.append($select)
        $select.append($("<option></option>").val("random").text("random"))
        $select.append($("<option></option>").val("force").text("force"))
        $select.append($("<option></option>").val("radial").text("radial"))
        $select.change((e) => this._update(this.singleNeighLayoutId, $select.val()))
        $cb.append(this._createLayoutStopButton(this.singleNeighLayoutFreezeId))

        // singleNeighPartsIds
        this.singleNeighPartsIds = {}
        for (const name of Neighborhood.PARTS) {
            let $cb = $("<div></div>").attr("class", "control-block")
            this.$neighborhoodDiv.append($cb)
            let $label = $("<label></label>").text("Show " + name)
            $cb.append($label)
            let id = 'single-neigh-part-' + nameToId(name)
            this.singleNeighPartsIds[name] = id
            let $input = $("<input>").attr("type", "checkbox")
                .attr("id", this.idPrefix + '-' + id)
            $label.append($input)
            $input.change((e) => this._update(id, $input.is(':checked')))
            $label.hide() // fixme how to show beautifully ?
        }
    }

    _addGraph() {
        this.$graphDiv = $("<div></div>").attr("id", this.idPrefix + '-single-graph').hide()
        this.$singleDiv.append(this.$graphDiv)

        let $cb = $("<div></div>").attr("class", "control-block")
        this.$graphDiv.append($cb)
        let $label = $("<label></label>").text("Layout")
        $cb.append($label)
        let $select = $("<select></select>").attr(
            "id", this.idPrefix + '-' + this.singleGraphLayoutId)
        $cb.append($select)
        $select.append($("<option></option>").val("random").text("random"))
        $select.append($("<option></option>").val("force").text("force"))
        $select.change((e) => this._update(this.singleGraphLayoutId, $select.val()))
        $cb.append(this._createLayoutStopButton(this.singleGraphLayoutFreezeId))

        // singleGraphEdgesId
        $cb = $("<div></div>").attr("class", "control-block")
        this.$graphDiv.append($cb)
        $label = $("<label></label>").text("Show edges")
        $cb.append($label)
        let $input = $("<input>").attr("type", "checkbox").prop("checked", true)
            .attr("id", this.idPrefix + '-' + this.singleGraphEdgesId)
        $label.append($input)
        $input.change((e) => this._update(this.singleGraphEdgesId, $input.is(':checked')))
    }

    // Satellites checkboxes
    _addSatellites() {
        this.$satellitesDiv = $("<div></div>").attr("id", this.idPrefix + '-single').hide()
            .addClass(this.idPrefix + '-class').addClass('var')
        this.$div.append(this.$satellitesDiv)
        this.$satellitesDiv.append($("<div></div>").attr("class", "menu-separator"))
        // TODO use satellites list

        this.satellitesIds = {}
        for (const satellite of VisibleGraph.SATELLITES) {
            let $cb = $("<div></div>").attr("class", "control-block")
            this.$satellitesDiv.append($cb)
            let $label = $("<label></label>").text("Show " + satellite)
            $cb.append($label)
            let id = 'satellites-' + nameToId(satellite)
            this.satellitesIds[satellite] = id
            let $input = $("<input>").attr("type", "checkbox").prop("checked", true)
                .attr("id", this.idPrefix + '-' + id)
            $label.append($input)
            $input.change((e) => this._update(id, $input.is(':checked')))
        }

        // singleClassAsColorId
        let $cb = $("<div></div>").attr("class", "control-block")
            .addClass(this.idPrefix + '-class').addClass('single').addClass('var').hide()
        this.$div.append($cb)
        let $label = $("<label></label>").text("Colored nodes")
            .attr("title", "Nodes are colored according to their labels")
        $cb.append($label)
        let $singleInput = $("<input>").attr("type", "checkbox").prop("checked", true)
            .attr("id", this.idPrefix + '-' + this.singleClassAsColorId)
        $label.append($singleInput)
        $singleInput.change((e) => this._update(
            this.singleClassAsColorId, $singleInput.is(':checked')))

        // multiNodeTypeAsColorId
        $cb = $("<div></div>").attr("class", "control-block")
            .addClass(this.idPrefix + '-class').addClass('multi').addClass('var').hide()
        this.$div.append($cb)
        $label = $("<label></label>").text("Colored nodes")
            .attr("title", "If nodes' features are 1-hot, then they are colored according to type")
        $cb.append($label)
        let $multiInput = $("<input>").attr("type", "checkbox").prop("checked", true)
            .attr("id", this.idPrefix + '-' + this.multiNodeTypeAsColorId)
        $label.append($multiInput)
        $multiInput.change((e) => this._update(
            this.multiNodeTypeAsColorId, $multiInput.is(':checked')))
    }

    async _update(key, value, block_await=false, $element) {
        // Block element to prevent several events until processing finishes
        // console.log('_update', key, value)
        if (block_await && $element) {
            blockDiv($element, true)
        }

        if (key in this.listeners)
            for (const set of Object.values(this.listeners[key]))
                for (const listener of set)
                    if (block_await) {
                        blockVisualsMenu(true)
                        await listener(key, value)
                        blockVisualsMenu(false)
                    }
                    else
                        listener(key, value)

        if (block_await && $element) {
            blockDiv($element, false)
        }
    }

    addListener(key, listener, tag=0) {
        if (!(key in this.listeners))
            this.listeners[key] = {}
        if (!(tag in this.listeners[key]))
            this.listeners[key][tag] = new Set()
        this.listeners[key][tag].add(listener)
    }

    removeListeners(tag) {
        for (const [key, tagSet] of Object.entries(this.listeners)) {
            if (tag in tagSet) {
                delete tagSet[tag]
                if (Object.keys(tagSet).length === 0)
                    delete this.listeners[key]
            }
        }
    }

    // Get all elements by their id
    _getById(id) {
        return $('#' + this.idPrefix + '-' + id)
    }

    /**
     * Get all elements by their class
     * @param classes - 1 or several classes
     * @param excludeOthers - if true, exclude elements containing non-listed classes
     * @returns {*|Window.jQuery|HTMLElement}
     * @private
     */
    _getByClass(classes, excludeOthers=false) {
        if (classes == null)
            classes = []
        else if ((typeof classes) === "string")
            classes = [classes]
        // Assume classes is a list of classes
        let query = '.' + this.idPrefix + '-class'
        if (excludeOthers) {
            // Get all that have one or more of these classes and none of other classes
            let $all = $(query)
            for (const cls of ['multi', 'single', 'var']) { // NOTE we must list all classes
                if (!classes.includes(cls))
                    $all = $all.not('.' + cls)
            }
            return $all
        }
        else {
            for (const cls of classes)
                query += '.' + cls
            return $(query)
        }
    }

    getValue(key) {
        let $elem = this._getById(key)
        if ($elem.prop('nodeName') === "BUTTON")
            return console.error('Not implemented')
        else
            if ($elem.prop("type") === "checkbox")
                return $elem.is(':checked')
            else
                return $elem.val()
    }

    setValue(key, val, fire=false) {
        let $elem = this._getById(key)
        if ($elem.prop('nodeName') === "BUTTON")
            return console.error('Not implemented')

        if (typeof val == "boolean")
            $elem.prop('checked', val)
        else
            $elem.val(val)

        if (fire)
            $elem.change()
    }

    setEnabled(key, enabled) {
        this._getById(key).prop("disabled", !enabled)
    }

    // Call change() for specified element
    fireEvent(key) {
        this._getById(key).change()
    }

    // Call change() for all elements listened by listeners with specified tag
    fireEventsByTag(tag) {
        for (const [key, tagSet] of Object.entries(this.listeners))
            if (tag in tagSet)
                this.fireEvent(key)
    }

    async onInit(block, args) {
        await super.onInit(block, args)
        if (block === "dvc") {
            this.datasetInfo = args
            this._isMulti = this.datasetInfo.count > 1
            if (this._isMulti) {
                this._getByClass('multi', true).show()
            }
            else {
                this._getByClass('single', true).show()
            }

            // Set initial config - TODO extend
            let initConfig = [
                // [this.showModeId, 'whole-graph'],
                [this.showModeId, 'neighborhood'],
                [this.singleGraphLayoutId, 'random'],
                [this.singleNeighLayoutId, 'force'],
                [this.singleNeighNodeId, 0],
                [this.singleNeighDepthId, 2],
                [this.singleClassAsColorId, true],
                [this.multiNodeTypeAsColorId, true],
                [this.multiLayoutId, 'force'],
                [this.multiGraphId, 0],
                [this.multiDepthId, 0],
                [this.multiCountId, "several"],
                [this.multiArrangeId, "grid"],
            ]
            for (const p of Neighborhood.PARTS)
                initConfig.push([this.singleNeighPartsIds[p], true])
            for (const satellite of VisibleGraph.SATELLITES)
                initConfig.push([this.satellitesIds[satellite], true])

            for (const [id, val] of initConfig)
                this.setValue(id, val)
        }
    }

    onSubmit(block, data) {
        super.onSubmit(block, data)
        if (block === "dc") {
        }
        else if (block === "dvc") {
            this._getByClass(['var', this._isMulti ? 'multi' : 'single'], true).show()
        }
    }

    onUnlock(block) {
        super.onUnlock(block)
        if (block === "dc") {
            this._getByClass().hide()
        }
        else if (block === "dvc") {
            this._getByClass('var').hide()
        }
    }
}

function blockVisualsMenu(on) {
    blockDiv($("#menu-visuals-view"), on, "wait")
}