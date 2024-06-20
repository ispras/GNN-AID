// Default output size of linear layer
LINEAR_LAYER_OUTPUT_SIZE = 16

class LayerBlock {
    static leftMargin = 8
    static batchnormColor = "#fbb"
    static dropoutColor = "#bbf"
    static convColor = "#bfb"
    static linearColor = "#bff"
    static connectionColor = "#ffb"

    constructor(type, absIx, ix) {
        console.assert(Array('n', 'g', 'gc').includes(type))
        this.type = type // Type: 'n' or 'g' or 'gc'
        this.absIx = absIx // Absolute index
        this.ix = ix // Order index
        this.nodeCount = null // number of node layers
        this.graphCount = null // number of graph layers
        this.$div = $("<div></div>")
            .css("border", "solid black 1px")
            .css("border-radius", "5px")
            .css("border-color", "#999999")
            .css("margin", "2px 0 0 2px")
        this.$label = $("<label></label>").css("grid-column", 1)

        // Variables

        // Layer type
        this.$typeSelect = null
        this.$typeParamsDiv = null // div for layer type params

        this.protParamsBuilder = null // Params for prot layer

        // One of 3 alternatives for type value
        this.$convSelect = null
        this.convParamsBuilder = null // for conv kwargs
        this.$linearOutputSizeInput = null
        this.sequential = null

        // Bathcnorm value and kwargs
        this.$batchnormSelect = null
        this.batchnormParamsBuilder = null

        // Activation value and kwargs
        this.$activationSelect = null
        this.activationParamsBuilder = null

        // Dropout value and kwargs
        this.$dropoutFlag = null
        this.$dropoutProbInput = null

        // Connections
        this.connections = null

    }

    /// Update this layer as the last one.
    // Return true if it is enough, false if the previous layer needs to be adapted too.
    setAsLast() {
        if (this.type === 'n')
            // Last node layer must have a pooling connection
            this.connections.setAsLast()

        // FIXME add others
        this.$activationSelect.val("LogSoftmax").change()
    }

    /// Modify layer dimensions according to its output dimension given.
    // Returns true if no modifications of previous layers are needed, otherwise returns false.
    setOutputSize(outputSize) {
        // Last node or graph layer must have corresponding output size
        switch (this.$typeSelect.val()) {
            case "conv":
                let convName = this.$convSelect.val()
                if (convName === 'APPNP')
                    return false
                if (convName === 'CGConv') {
                    this.convParamsBuilder.setValue("channels", outputSize)
                    return false
                }
                if (convName === 'GATConv' && this.convParamsBuilder.kwArgs['heads'] > 1) {
                    this.convParamsBuilder.setValue('heads', 1)
                    alert("Number of heads for GAT layer was set to 1 since it is the last layer. " +
                        "If you still want to have more than 1 heads, add more layers after")
                }
                this.convParamsBuilder.setValue("out_channels", outputSize)
                break
            case "lin":
                this.$linearOutputSizeInput.val(outputSize)
                break
            case "prot":
                this.$linearOutputSizeInput.val(outputSize)
                break
            case "gin":
                this.sequential.setAsLast(outputSize)
                break
            default:
                console.error('Not implemented')
        }
        return true
    }

    // copy(absIx, ix) {
    //     // FIXME not working. do we need it?
    //     let newLayerBlock = new LayerBlock(absIx, ix)
    //     newLayerBlock.build()
    //     newLayerBlock.$typeSelect.val(this.$typeSelect.val()).change()
    //     // FIXME why copied 2 times ???
    //     if (this.$convSelect)
    //         newLayerBlock.$convSelect.val(this.$convSelect.val()).change()
    //     newLayerBlock.$batchnormSelect.val(this.$batchnormSelect.val()).change()
    //     newLayerBlock.$activationSelect.val(this.$activationSelect.val()).change()
    //     newLayerBlock.$dropoutFlag.val(this.$dropoutFlag.val()).change()
    //     // TODO add all the rest
    //
    //     return newLayerBlock
    // }

    _name() {
        return {'n': 'Node layer', 'g': 'Graph layer', 'gc': 'Custom graph layer'}[this.type]
    }

    // Update layer label and skip connections
    update(ix, nodeCount, graphCount) {
        if (ix !== null) this.ix = ix
        this.nodeCount = nodeCount
        this.graphCount = graphCount
        this.$label.html(`<h4>${this._name()} ${this.ix}</h4>`)
        if (this.connections)
            this.connections.update(this.ix, this.nodeCount, this.graphCount)
    }

    async build(nodeCount, graphCount) {
        this.nodeCount = nodeCount
        this.graphCount = graphCount
        let $cc = this.$div, options, $cb, id

        // 0. Header with buttons
        let $headDiv = $("<div></div>")
            .css("display", "grid")
            .css("background-color", "#a5a5a5")
            .css("grid-template-columns", "2fr 1fr")
        $cc.append($headDiv)
        $headDiv.append(this.$label)
        this.$label.html(`<h4>${this._name()} ${this.ix}</h4>`)
        let $buttonsDiv = $("<div></div>")
            .css("grid-column", 2).css("display", "flex").css("flex-flow", "row-reverse")
        $headDiv.append($buttonsDiv)

        if (this.type === 'gc')
            this.buildButtons($buttonsDiv, false, false, true)
        else
            this.buildButtons($buttonsDiv)

        // 1. Layer type
        $cb = $("<div></div>").attr("class", "control-block")
        $cc.append($cb)
        id = "menu-model-constructor-type-" + this.absIx
        $cb.append($("<label></label>").text("Layer type").attr("for", id))
        this.$typeSelect = $("<select></select>").attr("id", id)
        $cb.append(this.$typeSelect)
        if (this.type === 'n')
            this.$typeSelect.append($("<option></option>").val("conv").text("Convolution"))
        if (this.type === 'n' || this.type === 'g')
            this.$typeSelect.append($("<option></option>").val("lin").text("Linear"))
        if (this.type === 'gc')
            this.$typeSelect.append($("<option></option>").val("prot").text("Prot"))
        if (this.type === 'n')
            this.$typeSelect.append($("<option></option>").val("gin").text("GIN"))
        // this.$typeSelect.append($("<option></option>").val("None").text("None"))

        let self = this
        this.$typeSelect.change(function () {
            self.buildTypeParameters(null)
            self.buildTypeParameters(this.value)
        })

        // 1.1 Layer type constructor params
        $cb = $("<div></div>").attr("class", "control-block")
        $cc.append($cb)
        this.$typeParamsDiv = $("<div></div>")
            .css("margin-left", LayerBlock.leftMargin + "px")
        $cb.append(this.$typeParamsDiv)

        this.$typeSelect.change()

        // 2. BatchNorm
        if (this.type !== 'gc') {
            $cc.append($("<div></div>").attr("class", "menu-separator"))
            options = [
                ["None", "None"], ["BatchNorm1d", "1d"], ["BatchNorm2d", "2d"], ["BatchNorm3d", "3d"]]
            let $batchnormParamsDiv
            [$cb, this.$batchnormSelect, $batchnormParamsDiv, this.batchnormParamsBuilder]
                = await addOptionsWithParams("menu-model-constructor-batchnorm-" + this.absIx,
                "Batch normalization", options, "F", LayerBlock.batchnormColor)

            $cc.append($cb)
            $cc.append($batchnormParamsDiv)
        }

        // 3. Activation
        $cc.append($("<div></div>").attr("class", "menu-separator"))
        options = [
            ["ReLU", "ReLU"],
            ["ReLU6", "ReLU6"],
            ["LogSoftmax", "Log-Softmax"],
            ["Sigmoid", "Sigmoid"],
            ["Tanh", "Tanh"],
            ["ELU", "ELU"],
            ["LeakyReLU", "LeakyReLU"],
            ["None", "None"]]
        let $activationParamsDiv
        [$cb, this.$activationSelect, $activationParamsDiv, this.activationParamsBuilder]
            = await addOptionsWithParams("menu-model-constructor-activation-" + this.absIx,
            "Activation", options, "F", null)

        $cc.append($cb)
        $cc.append($activationParamsDiv)

        // 4. Dropout
        if (this.type !== 'gc') {
            $cc.append($("<div></div>").attr("class", "menu-separator"))
            $cb = $("<div></div>").attr("class", "control-block")
            $cc.append($cb)
            id = "menu-model-constructor-dropout-" + this.absIx
            $cb.append($("<label></label>").text("Dropout").attr("for", id))
            this.$dropoutFlag = $("<input>").attr("id", id).attr("type", "checkbox")
                .prop("checked", false)
            $cb.append(this.$dropoutFlag)

            // 4.1 Dropout params
            let $dropoutParamsDiv = $("<div></div>")
                .attr("class", "disabledDiv")
                .css("margin-left", "5px")
                .css("display", "flex")
            $cb.append($dropoutParamsDiv)

            id = "menu-model-constructor-dropout-probability-"
            this.$dropoutProbInput = $("<input>").attr("id", id).attr("type", "number")
                .attr("min", 0).attr("max", 1).attr("step", 0.01).val(0.5)
            $dropoutParamsDiv.append($("<label></label>").text("probability").attr("for", id))
            $dropoutParamsDiv.append(this.$dropoutProbInput)

            // 4.2 Dropout flag listener
            this.$dropoutFlag.change(() => {
                let on = this.$dropoutFlag.is(":checked")
                if (on) {
                    $dropoutParamsDiv.removeClass("disabledDiv")
                    $dropoutParamsDiv.css("background-color", LayerBlock.dropoutColor)
                } else {
                    $dropoutParamsDiv.addClass("disabledDiv")
                    $dropoutParamsDiv.css("background-color", "transparent")
                }
            })
        }

        // 5. Connections
        if (this.type !== 'gc') {
            $cc.append($("<div></div>").attr("class", "menu-separator"))
            $cb = $("<div></div>").attr("class", "control-block")
            $cc.append($cb)
            $cb.append($("<label></label>").text("Connections"))
            this.connections = new Connections(
                this.type, this.ix, this.nodeCount, this.graphCount,
                "menu-model-constructor-connections-" + this.type + "-" + this.absIx + "-")
            $cc.append(this.connections.$div)
            this.connections.build()
        }
    }

    buildButtons($div, up=true, down=true, close=true, clone=false) {
        if (close) {
            let $close = $("<button></button>").css("padding", 0)
            $close.append($("<img>").attr("src", "../static/icons/close-sm-svgrepo-com.svg")
                .attr("height", "16px"))
            $div.append($close)
            $close.click(() => this.close())
        }

        if (down) {
            let $down = $("<button></button>").css("padding", 0)
            $down.append($("<img>").attr("src", "../static/icons/arrow-down-sm-svgrepo-com.svg")
                .attr("height", "16px"))
            $div.append($down)
            $down.click(() => this.down())
        }

        if (up) {
            let $up = $("<button></button>").css("padding", 0)
            $up.append($("<img>").attr("src", "../static/icons/arrow-up-sm-svgrepo-com.svg")
                .attr("height", "16px"))
            $div.append($up)
            $up.click(() => this.up())
        }

        if (clone) {
            // TODO implement correctly
            // let $clone = $("<button></button>").css("padding", 0).text("clone")
            // $div.append($clone)
            // $clone.click(() => this.clone())
        }
    }

    async buildTypeParameters(type) {
        let $tpc = this.$typeParamsDiv
        if (type == null) {
            $tpc.empty()
        }
        else if (type === "conv") {
            // Convolution
            let options = [
                ["GCNConv", "GCN"],
                ["SAGEConv", "SAGE"],
                ["GATConv", "GAT"],
                ["SGConv", "SG"],
                ["TAGConv", "TAG"],
                ["ARMAConv", "ARMA"],
                ["SSGConv", "SSG"],
                ["APPNP", "APPNP"],
                ["GMM", "GMM"],
                ["CGConv", "CGConv"],
            ]
            let $cb, $convParamsDiv
            [$cb, this.$convSelect, $convParamsDiv, this.convParamsBuilder]
                = await addOptionsWithParams("menu-model-constructor-conv-Conv-" + this.absIx,
                "Convolution", options, "M", LayerBlock.convColor)

            $tpc.append($cb)
            $cb.append(this.$convSelect)
            $tpc.append($convParamsDiv)
        }
        else if (type === "lin") {
            // Linear layer
            let $cb = $("<div></div>").attr("class", "control-block")
            $tpc.append($cb)
            let id = "menu-model-constructor-linear-outsize-" + this.absIx
            $cb.append($("<label></label>").text("Output size").attr("for", id))
                .css("background-color", LayerBlock.linearColor)
            this.$linearOutputSizeInput = $("<input>").attr("id", id).attr("type", "number")
                .attr("min", 1).attr("step", 1).val(LINEAR_LAYER_OUTPUT_SIZE)
            $cb.append(this.$linearOutputSizeInput)
        }
        else if (type === "gin") {
            // Sequential layers
            this.sequential = new SequentialLayer(
                "menu-model-constructor-sequential-" + this.absIx + "-")
            $tpc.append(this.sequential.$div)
            this.sequential.build()
        }
        else if (type === "prot") {
            // Prot params
            let $paramsDiv = $("<div></div>")
                // .css("margin-left", LayerBlock.leftMargin + "px")
                .css("background-color", LayerBlock.linearColor)
            this.protParamsBuilder = new ParamsBuilder($paramsDiv, 'M', "menu-model-constructor-prot-params-")
            await this.protParamsBuilder.build("Prot")
            $tpc.append($paramsDiv)

            // Linear layer - need it nominally just to set outSize = num_classes
            let $cb = $("<div></div>").attr("class", "control-block")
            $tpc.append($cb)
            let id = "menu-model-constructor-prot-outsize-" + this.absIx
            $cb.append($("<label></label>").text("Output size").attr("for", id))
                .css("background-color", LayerBlock.linearColor)
            this.$linearOutputSizeInput = $("<input>").attr("id", id).attr("type", "number")
                .attr("min", 1).attr("step", 1).val(LINEAR_LAYER_OUTPUT_SIZE)
            $cb.append(this.$linearOutputSizeInput)
            // Not showing - because prot layer is the last and outSize = num_classes
            $cb.hide()
        }
        else {
            console.error("Not implemented")
        }
    }

    constructConfig(inputSize=-1) {
        let config = {
            'label': {'n': 'n', 'g': 'g', 'gc': 'g'}[this.type], // FIXME replace gc with checker?
            'layer': {},
        }
        let layer = config['layer']
        let outputSize = -1
        let type = this.$typeSelect.val()
        // layer['layerName'] = type
        if (type === 'prot') { // Prot
            layer['layer_name'] = 'Prot'
            layer['layer_kwargs'] = {
                'in_features': inputSize,
                'num_classes': parseInt(this.$linearOutputSizeInput.val()),
            }
            Object.assign(layer['layer_kwargs'], this.protParamsBuilder.kwArgs)
            outputSize = parseInt(this.$linearOutputSizeInput.val())
        } else if (type === 'conv') {
            let name = layer['layer_name'] = this.$convSelect.val()
            layer['layer_kwargs'] = {}

            if (!['APPNP', 'CGConv'].includes(name))
                layer['layer_kwargs']['in_channels'] = inputSize

            Object.assign(layer['layer_kwargs'], this.convParamsBuilder.kwArgs)
            outputSize = this.convParamsBuilder.kwArgs['out_channels']

            if (name === 'CGConv') {
                delete layer['layer_kwargs']['out_channels']
                layer['layer_kwargs']['channels'] = inputSize
                outputSize = inputSize
            }
            else if (name === 'GATConv') {
                outputSize = layer['layer_kwargs']['out_channels'] * layer['layer_kwargs']['heads']
            }
        } else if (type === 'lin') {
            layer['layer_name'] = 'Linear'
            layer['layer_kwargs'] = {
                'in_features': inputSize,
                'out_features': parseInt(this.$linearOutputSizeInput.val()),
            }
            outputSize = layer['layer_kwargs']['out_features']
        } else if (type === 'gin') {
            let [a, c] = this.sequential.constructConfig(inputSize)
            layer['gin_seq'] = a
            outputSize = c
            // [layer['gin_seq'], outputSize] = this.sequential.constructConfig(inputSize)
            layer['layer_name'] = 'GINConv'
            layer['layer_kwargs'] = null // TODO
        } else
            console.error('Not implemented')

        // 2. Batchnorm
        if (this.$batchnormSelect && this.$batchnormSelect.val() !== "None") {
            config['batchNorm'] = {
                'batchNorm_name': this.$batchnormSelect.val(),
                'batchNorm_kwargs': Object.assign({}, this.batchnormParamsBuilder.kwArgs)
            }
            config['batchNorm']['batchNorm_kwargs']['num_features'] = outputSize
        }
        // 3. Activation
        if (this.$activationSelect && this.$activationSelect.val() !== "None") {
            config['activation'] = {
                'activation_name': this.$activationSelect.val(),
                'activation_kwargs': Object.assign({}, this.activationParamsBuilder.kwArgs),
            }
        }
        // 4. Dropout
        if (this.$dropoutFlag && this.$dropoutFlag.is(':checked'))
            config['dropout'] = {
                'dropout_name': 'Dropout',
                'dropout_kwargs': {
                    'p': parseFloat(this.$dropoutProbInput.val()),
                }
            }
        // 5. Connections
        let additionalNodeIxes = []
        let additionalGraphIxes = []
        if (this.connections) {
            [config['connections'], additionalNodeIxes, additionalGraphIxes]
                = this.connections.constructConfig()

        }
        return [config, outputSize, additionalNodeIxes, additionalGraphIxes]
    }
}
