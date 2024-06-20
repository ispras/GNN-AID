class SequentialLayer {
    constructor(idPrefix) {
        this.$div = $("<div></div>")
        this.idPrefix = idPrefix ? idPrefix : timeBasedId() + '-'

        this.$buttonDiv = null
        this.linearInputs = null
        this.batchnormInputs = null
        this.batchnormBuilders = null
        this.activationBuilders = null
        this.activationInputs = null
    }

    build() {
        this.linearInputs = []
        this.batchnormInputs = []
        this.batchnormBuilders = []
        this.activationBuilders = []
        this.activationInputs = []

        this.$buttonDiv = $("<div></div>").attr("class", "control-block")
        this.$div.append(this.$buttonDiv)
        let $button = $("<button></button>").text("+ block").css("margin-right", "12px")
        this.$buttonDiv.append($button)

        $button.click(() => this.addBlock())

        this.addBlock()
    }

    setAsLast(outputSize) {
        this.linearInputs[this.linearInputs.length-1].val(outputSize)
    }

    async addBlock() {
        let ix = this.linearInputs.length
        // Linear
        let $cb = $("<div></div>").attr("class", "control-block")
        this.$buttonDiv.before($cb)
        let id = this.idPrefix + "linear" + ix
        $cb.append($("<label></label>").text("Linear output size").attr("for", id))
            .css("background-color", LayerBlock.linearColor)
        let $linearOutputSizeInput = $("<input>").attr("id", id).attr("type", "number")
            .attr("min", 1).attr("step", 1).val(LINEAR_LAYER_OUTPUT_SIZE)
        $cb.append($linearOutputSizeInput)
        this.linearInputs.push($linearOutputSizeInput)

        // Batchnorm
        let options = [["None", "None"], ["BatchNorm1d", "1d"],
            ["BatchNorm2d", "2d"], ["BatchNorm3d", "3d"]]
        let $batchnormSelect, $batchnormParamsDiv, batchnormParamsBuilder
        [$cb, $batchnormSelect, $batchnormParamsDiv, batchnormParamsBuilder]
            = await addOptionsWithParams("menu-model-constructor-seq-batchnorm-" + ix,
            "Batch normalization", options, "F", LayerBlock.batchnormColor)

        this.$buttonDiv.before($cb)
        this.$buttonDiv.before($batchnormParamsDiv)
        this.batchnormInputs.push($batchnormSelect)
        this.batchnormBuilders.push(batchnormParamsBuilder)

        // Activation
        options = [
            ["ReLU", "ReLU"],
            ["ReLU6", "ReLU6"],
            ["LogSoftmax", "Log-Softmax"],
            ["Sigmoid", "Sigmoid"],
            ["Tanh", "Tanh"],
            ["ELU", "ELU"],
            ["LeakyReLU", "LeakyReLU"],
        ]
        let $activationSelect, $activationParamsDiv, activationParamsBuilder
        [$cb, $activationSelect, $activationParamsDiv, activationParamsBuilder]
            = await addOptionsWithParams("menu-model-constructor-seq-activation-" + ix,
            "Activation", options, "F", null)

        this.$buttonDiv.before($cb)
        this.$buttonDiv.before($activationParamsDiv)
        this.activationInputs.push($activationSelect)
        this.activationBuilders.push(activationParamsBuilder)
    }

    constructConfig(inputSize=-1) {
        let config = []
        for (let i = 0; i < this.linearInputs.length; i++) {
            let linInput = parseInt(this.linearInputs[i].val())
            let block = {
                'layer': {
                    'layer_name': 'Linear',
                    'layer_kwargs': {
                        'in_features': inputSize,
                        'out_features': linInput,
                    },
                }
            }
            let batchnormInput = this.batchnormInputs[i].val()
            if (batchnormInput !== "None") {
                block['batchNorm'] = {
                    'batchNorm_name': batchnormInput,
                    'batchNorm_kwargs': Object.assign({}, this.batchnormBuilders[i].kwArgs),
                }
                block['batchNorm']['batchNorm_kwargs']['num_features'] = linInput
            }
            block['activation'] = {
                'activation_name': this.activationInputs[i].val(),
                'activation_kwargs': Object.assign({}, this.activationBuilders[i].kwArgs),
            }
            inputSize = linInput
            config.push(block)
        }
        return [config, inputSize]
    }
}
