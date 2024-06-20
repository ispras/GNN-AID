// const WEIGHT_COLORMAP = 'bwr'
const MAX_MATRIX_SIZE = 10000 // max cells in matrix to omit its drawing

function stringifyKwargs(kwargs) {
    let res = ''
    if (kwargs)
        for (let [k, v] of Object.entries(kwargs)) {
            if (res.length > 0) res += ', '
            if (v.constructor === Object)
                v = '{' + stringifyKwargs(v) + '}'
            res += k.toString() + '=' + v.toString()
        }
    return res
}

/**
 * Architecture of GNN Model
 */
class PanelModelArchView extends PanelView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        this.size = 20 // Rectangle primitive size

        this.svgPanel = null
        this.showTipFunc = null // Tip update function to call when necessary
        this.modelStructureConfig = null
        this.modelWeights = null
        this.primitives = null // SVG primitives (rects) for parameters, repeat modelWeights architecture
    }

    init() {
        super.init("Architecture")
    }

    onSubmit(block, data) {
        super.onSubmit(block, data)
        if (block === "mconstr" || block === "mload" || block === "mcustom") {
            // FIXME for mcust weights and architecture can be unknown
            this.modelStructureConfig = data["architecture"]["layers"]
            this.modelWeights = data["weights"]
            this._init()
        }
        else if (block === "mmc") {
            // this.modelWeights = data["weights"]
            // this._init()
        }
    }

    onReceive(block, data) {
        if (block === "mt") {
            if ("weights" in data) {
                this.modelWeights = data["weights"]
                this.update()
            }
        }
    }

    _getParameter(modelWeights, keys) {
        let data = modelWeights
        let name = 'Param'
        for (const key of keys) {
            data = data[key]
            name += '[' + key + ']'
        }
        return `${name}=` + data.toFixed(5)
    }

    // Add mouse listener for all elements which shows a tip with given text
    _addTip(element, func, params) {
        element.onmousemove = (e) => {
            this.svgPanel.$tip.show()
            this.svgPanel.$tip.css("left", e.pageX + 10)
            this.svgPanel.$tip.css("top", e.pageY + 15)
            this.showTipFunc = () => this.svgPanel.$tip.html(func(this.modelWeights, params))
            this.showTipFunc()
        }
        element.onmouseout = (e) => {
            this.svgPanel.$tip.hide()
            this.showTipFunc = null
        }
    }

    // Create SVG elements
    _init() {
        if (this.svgPanel)
            this.$body.empty()

        if (this.modelStructureConfig == null || Object.keys(this.modelStructureConfig).length === 0) {
            this.$body.html("Not available")
            this._collapse(true)
            // FIXME what if we know weights but not structure ?
            return
        }

        // parent SVG element
        this.svgPanel = new SvgPanel(this.$body[0])
        this.svgPanel.$svg.css("border", "1px solid #22ee22")
        this._collapse(false)

        // Init all SVG primitives
        this.primitives = {}

        // Function drawing an parameters data object
        let marginText = 5
        let marginBlocks = 140 // TODO make it = rightmost bound of all texts
        let draw = (kv, primitives, offsets, keysList=[], depth=0) => {
            for (let [key, value] of Object.entries(kv)) {
                let text
                // Additional functions: activation, dropout, etc
                if (key.includes('additional')) {
                    let activation = value["activation"]
                    if (activation) {
                        let kwargs = stringifyKwargs(activation["activation_kwargs"])
                        if (kwargs.length > 0) kwargs = ' (' + kwargs + ')'
                        this.svgPanel.$svg.append(Svg.text(
                            "Activation: " + activation["activation_name"] + kwargs,
                            marginText + 12*depth, offsets[1] + 8,
                            'middle', `${20 - 2*depth}px`,
                            'normal', "#000000"
                        ))
                        offsets[1] += 25 - 2*depth
                    }
                    let dropout = value["dropout"]
                    if (dropout) {
                        let kwargs = stringifyKwargs(dropout["dropout_kwargs"])
                        if (kwargs.length > 0) kwargs = ' (' + kwargs + ')'
                        this.svgPanel.$svg.append(Svg.text(
                            dropout["dropout_name"] + kwargs,
                            marginText + 12*depth, offsets[1] + 8,
                            'middle', `${20 - 2*depth}px`,
                            'normal', "#000000"
                        ))
                        offsets[1] += 25 - 2*depth
                    }
                    let connections = value["connections"]
                    if (connections) {
                        for (const connection of connections) {
                            let kwargs = stringifyKwargs(connection["connection_kwargs"])
                            if (kwargs.length > 0) kwargs = ' (' + kwargs + ')'
                            let text = 'Conn: to ' + connection["into_layer"] + kwargs
                            this.svgPanel.$svg.append(Svg.text(
                                text,
                                marginText + 12*depth, offsets[1] + 8,
                                'middle', `${20 - 2*depth}px`,
                                'normal', "#000000"
                            ))
                            offsets[1] += 25 - 2*depth
                        }
                    }
                    // TODO others
                    continue
                }

                // Text
                offsets[1] += 25 - 2*depth
                text = Svg.text(key, marginText + 12*depth, offsets[1],
                    'middle', `${20 - 2*depth}px`,
                    'bold', "#000000")
                this.svgPanel.$svg.append(text)

                if (value.constructor === Object) {
                    // Go deeper
                    primitives[key] = {}
                    draw(value, primitives[key], offsets, keysList.concat([key]),depth+1)
                }
                else if (typeof(value) === 'number') { // Number
                    let bbox = text.getBBox() // bbox of last text element
                    text = Svg.text(value, bbox.x + bbox.width + 5, offsets[1],
                        'middle', `${20 - 2*depth}px`,
                        'bold', "#000000")
                    this.svgPanel.$svg.append(text)
                    primitives[key] = text
                    offsets[1] += 25
                }
                else if (value.constructor === Array) {
                    let arrayPrimitives = []
                    primitives[key] = arrayPrimitives
                    // Depending on dimensionality
                    if (!Array.isArray(value[0])) { // 1-dim vector
                        let b = value.length
                        offsets[1] -= 10
                        for (let i = 0; i < b; ++i) {
                            let rect = document.createElementNS("http://www.w3.org/2000/svg", "rect")
                            let x = marginBlocks + this.size * i + 12*depth
                            let y = offsets[1]
                            rect.setAttribute('x', x)
                            rect.setAttribute('y', y)
                            rect.setAttribute('width', this.size)
                            rect.setAttribute('height', this.size)
                            let color = valueToColor(value[i], EMBEDDING_COLORMAP, -2, 2, true, 0.2)
                            rect.setAttribute('fill', color)
                            rect.setAttribute('stroke', '#000000')
                            rect.setAttribute('stroke-width', 1)
                            this.svgPanel.$svg.append(rect)
                            this._addTip(rect, this._getParameter, keysList.concat(key, i))
                            arrayPrimitives.push(rect)
                        }
                        offsets[1] += this.size + 20
                    }
                    else { // dims >= 2
                        let omit0 = false
                        if (Array.isArray(value[0][0])) { // dims >= 3
                            if (value.length > 1) {
                                console.error("Can't work with >=2 dims parameters with value[0].length > 1")
                                return
                            }
                            value = value[0]
                            omit0 = true
                        }
                        // Matrix
                        let w = value.length
                        let h = value[0].length
                        if (w * h > MAX_MATRIX_SIZE) {
                            this.svgPanel.$svg.append(Svg.text(
                                `[Matrix ${h}x${w}]`,
                                marginBlocks + 12*depth, offsets[1],
                                'middle', '20px',
                                'normal', "#000000"
                            ))
                            offsets[1] += this.size + 20
                        }
                        else {
                            offsets[1] -= 10
                            for (let i = 0; i < w; ++i) {
                                let rowPrimitives = []
                                for (let j = 0; j < h; ++j) {
                                    let rect = document.createElementNS("http://www.w3.org/2000/svg", "rect")
                                    let x = marginBlocks + this.size * i + 12 * depth
                                    let y = offsets[1] + this.size * j
                                    rect.setAttribute('x', x)
                                    rect.setAttribute('y', y)
                                    rect.setAttribute('width', this.size)
                                    rect.setAttribute('height', this.size)
                                    let color = valueToColor(value[i][j], EMBEDDING_COLORMAP, -2, 2, true, 0.2)
                                    rect.setAttribute('fill', color)
                                    rect.setAttribute('stroke', '#000000')
                                    rect.setAttribute('stroke-width', 1)
                                    this.svgPanel.$svg.append(rect)
                                    rowPrimitives.push(rect)
                                    let keys = keysList.concat(key)
                                    if (omit0) keys = keys.concat(0)
                                    this._addTip(rect, this._getParameter, keys.concat(i, j))
                                }
                                arrayPrimitives.push(rowPrimitives)
                            }
                            offsets[1] += h * this.size + 20
                        }
                    }
                }
                else console.error("Unknown type")
            }
        }

        // Insert additional elements into modelWeights (activation, dropout, etc)
        let modelDataAdv = {}
        let lastIx = 0
        for (let [key, value] of Object.entries(this.modelWeights)) {
            let ix = parseInt(key.slice(-1))
            if (ix > lastIx) {
                modelDataAdv['additional' + lastIx] = {
                    'activation': this.modelStructureConfig[lastIx]["activation"],
                    'dropout': this.modelStructureConfig[lastIx]["dropout"],
                    'connections': this.modelStructureConfig[lastIx]["connections"],
                }
                lastIx = ix
            }
            if (key.includes('gin')) {
                // Advance value["nn"]
                let valueNnAdv = {}
                let lastNnIx = 0
                for (const [k, v] of Object.entries(value["nn"])) {
                    let nnIx = parseInt(k.slice(-1))
                    if (nnIx > lastNnIx) {
                        valueNnAdv['additional' + lastNnIx] = {
                            'activation': this.modelStructureConfig[lastIx]["gin"][lastNnIx]["activation"],
                        }
                        lastNnIx = nnIx
                    }
                    valueNnAdv[k] = v
                }
                valueNnAdv['additional' + lastNnIx] = {
                    'activation': this.modelStructureConfig[lastIx]["gin"][lastNnIx]["activation"],
                }
                value["nn"] = valueNnAdv
            }
            modelDataAdv[key] = value
        }
        // Same for the last layer
        modelDataAdv['additional' + lastIx] = {
            'activation': this.modelStructureConfig[lastIx]["activation"],
            'dropout': this.modelStructureConfig[lastIx]["dropout"],
            'connections': this.modelStructureConfig[lastIx]["connections"],
        }

        // Draw
        let offset = 10
        let offsets = [offset, offset]
        draw(modelDataAdv, this.primitives, offsets)

        let bbox = this.svgPanel.$svg[0].getBBoxWithCopying()
        let w = bbox.width + 2*offset
        let h = offsets[1] + offset
        this.svgPanel.$svg.css("width", w + "px")
        this.svgPanel.$svg.css("height", h + "px")
    }

    // Update SVG elements colors according to a new modelWeights assuming same modelConfig
    update() {
        if (super.collapsed) {
            this._updateArgs = arguments
            return
        }
        if (this.primitives == null)
            return

        console.log('PanelModelArchView.update()')
        let _upd = (kv, primitives, depth=0) => {
            for (let [key, value] of Object.entries(kv)) {
                if (value.constructor === Object) {
                    // Go deeper
                    _upd(value, primitives[key],depth+1)
                }
                else if (value.constructor === Array) {
                    let arrayPrimitives = primitives[key]
                    if (!Array.isArray(value[0])) {
                        // Vector
                        for (let i = 0; i < value.length; ++i) {
                            let color = valueToColor(value[i], EMBEDDING_COLORMAP, -2, 2, true, 0.2)
                            arrayPrimitives[i].setAttribute('fill', color)
                        }
                    }
                    else { // dims >= 2
                        if (Array.isArray(value[0][0])) { // dims >= 3
                            if (value.length > 1) {
                                console.error("Can't work with >=2 dims parameters with value[0].length > 1")
                                return
                            }
                            value = value[0]
                        }
                        // Matrix
                        let w = value.length
                        let h = value[0].length
                        if (w * h <= MAX_MATRIX_SIZE)
                            for (let i = 0; i < w; ++i)
                                for (let j = 0; j < h; ++j) {
                                    let color = valueToColor(value[i][j], EMBEDDING_COLORMAP, -2, 2, true, 0.2)
                                    arrayPrimitives[i][j].setAttribute('fill', color)
                                }
                    }
                }
                else if (typeof(value) === 'number') { // Number
                    primitives[key].textContent = value
                }
                else console.error("Unknown type")
            }
        }

        _upd(this.modelWeights, this.primitives)
        if (this.showTipFunc)
            this.showTipFunc()
    }

    break() {
        super.break()
        this.showTipFunc = null
        this.svgPanel = null
        this.primitives = null
    }
}
