class MenuModelConstructorView extends MenuView {
    constructor($div, requestBlock, listenBlocks) {
        super($div, requestBlock, listenBlocks)

        // Variables
        this.num_feats = null
        this.num_classes = null
        this.multi = null
    }

    async init(args) {
        super.init(args)

        this.state = MVState.ACTIVE

        this.num_feats = args[0]
        this.num_classes = args[1]
        this.multi = args[2]

        this.$configConstructorDiv = $("<div></div>")
        this.$mainDiv.append(this.$configConstructorDiv)

        await this.buildStructure()
        this.$mainDiv.append($("<div></div>").attr("class", "menu-separator"))

        this.appendAcceptBreakButtons()
    }

    async _accept() {
        let mc = this.constructModelConfig()
        if (mc == null)
            return -1

        console.log("architecture", mc)
        await Controller.blockRequest(this.requestBlock, 'modify', {layers: mc})
    }

    // Update skip-connections according to number of blocks of each type
    updConnections() {
        let iter = this.nodeLayerBlocks.iterator()
        let result = iter.next()
        while (!result.done) {
            result.value.update(null, this.nodeLayerBlocks.length, this.graphLayerBlocks.length)
            result = iter.next()
        }
        iter = this.graphLayerBlocks.iterator()
        result = iter.next()
        while (!result.done) {
            result.value.update(null, this.nodeLayerBlocks.length, this.graphLayerBlocks.length)
            result = iter.next()
        }
    }

    // Configuration
    async buildStructure() {
        let $cc = this.$configConstructorDiv
        $cc.append($("<label></label>").html("<h3>Structure</h3>"))

        let $nodeLayerBlocksDiv = $("<div></div>")
        $cc.append($nodeLayerBlocksDiv)

        // Reset blocks
        this.absLayersCounter = 0
        this.nodeLayerBlocks = new BiList()
        this.graphLayerBlocks = new BiList()
        this.customGraphLayerBlock = null // TODO can we have a lot of them ?

        // First node layer block is obligate
        let layerBlock = new LayerBlock('n', this.absLayersCounter, 0)
        this.nodeLayerBlocks.append(layerBlock)
        $nodeLayerBlocksDiv.append(layerBlock.$div)
        await layerBlock.build(this.nodeLayerBlocks.length, this.graphLayerBlocks.length)

        // Add node layer button
        let $cb = $("<div></div>").attr("class", "control-block")
        $cc.append($cb)
        let $addNodeLayerButton = $("<button></button>").attr("id", "model-button-constructor-add-layer")
            .text("+ node layer").css("margin-right", "12px")
        $cb.append($addNodeLayerButton)

        $addNodeLayerButton.click(() => {
            this.absLayersCounter++
            let layerBlock = new LayerBlock('n', this.absLayersCounter,
                this.nodeLayerBlocks.length, this.graphLayerBlocks.length)
            $nodeLayerBlocksDiv.append(layerBlock.$div)
            layerBlock.build(this.nodeLayerBlocks.length, this.graphLayerBlocks.length)
            this.nodeLayerBlocks.append(layerBlock)
            this.updConnections()
        })

        // Graph layer block if needed
        let $graphLayerBlocksDiv
        if (this.multi) {
            $graphLayerBlocksDiv = $("<div></div>")
            $cc.append($graphLayerBlocksDiv)

            // First graph layer block is obligate
            let layerBlock = new LayerBlock('g', this.absLayersCounter, 0)
            this.graphLayerBlocks.append(layerBlock)
            $graphLayerBlocksDiv.append(layerBlock.$div)
            await layerBlock.build(this.graphLayerBlocks.length, this.graphLayerBlocks.length)
            this.nodeLayerBlocks.first().update(null, this.nodeLayerBlocks.length, this.graphLayerBlocks.length)

            let $cb = $("<div></div>").attr("class", "control-block")
            $cc.append($cb)
            let $addGraphLayerButton = $("<button></button>").attr("id", "model-button-constructor-add-layer")
                .text("+ graph layer").css("margin-right", "12px")
            $cb.append($addGraphLayerButton)

            $addGraphLayerButton.click(() => {
                this.absLayersCounter++
                let layerBlock = new LayerBlock('g', this.absLayersCounter, this.graphLayerBlocks.length)
                $graphLayerBlocksDiv.append(layerBlock.$div)
                layerBlock.build(this.nodeLayerBlocks.length, this.graphLayerBlocks.length)
                this.graphLayerBlocks.append(layerBlock)
                this.updConnections()
            })
        }

        // Custom layer block - for now Prot, only 1
        let $customGraphLayerBlockDiv
        let $customGraphLayerButtonDiv
        if (this.multi) {
            $customGraphLayerBlockDiv = $("<div></div>")
            $cc.append($customGraphLayerBlockDiv)

            $customGraphLayerButtonDiv = $("<div></div>").attr("class", "control-block")
            $cc.append($customGraphLayerButtonDiv)
            let $addCustomGraphLayerButton = $("<button></button>").attr("id", "model-button-constructor-add-layer")
                .text("+ custom graph layer").css("margin-right", "12px")
            $customGraphLayerButtonDiv.append($addCustomGraphLayerButton)

            $addCustomGraphLayerButton.click(async () => {
                this.absLayersCounter++
                let layerBlock = new LayerBlock('gc', this.absLayersCounter, 0)
                $customGraphLayerBlockDiv.append(layerBlock.$div)
                await layerBlock.build(this.nodeLayerBlocks.length, this.graphLayerBlocks.length)
                this.customGraphLayerBlock = layerBlock
                this.updConnections() // fixme not sure we need
                $customGraphLayerButtonDiv.hide()
            })
        }

        // LayerBlock control buttons listeners
        let self = this
        LayerBlock.prototype.close = function () {
            if (this.type === 'gc') {
                self.customGraphLayerBlock = null
                this.$div.remove()
                $customGraphLayerButtonDiv.show()
            }
            else {
                let layerBlocks = this.type === 'n' ? self.nodeLayerBlocks : self.graphLayerBlocks
                if (layerBlocks.length === 1) return
                let ix = this.ix
                this.$div.remove()
                layerBlocks.remove(ix)
                let iter = layerBlocks.iterator(ix)
                let result = iter.next()
                while (!result.done) {
                    result.value.ix -= 1
                    result = iter.next()
                }
                self.updConnections()
            }
        }
        LayerBlock.prototype.up = function () {
            let layerBlocks = this.type === 'n' ? self.nodeLayerBlocks : self.graphLayerBlocks
            let $layerBlocksDiv = this.type === 'n' ? $nodeLayerBlocksDiv : $graphLayerBlocksDiv
            let ix = this.ix
            if (ix > 0) {
                $layerBlocksDiv.children().eq(ix).after($layerBlocksDiv.children().eq(ix - 1))
                this.update(ix - 1, self.nodeLayerBlocks.length, self.graphLayerBlocks.length)
            }
            layerBlocks.toFirst(ix)
            layerBlocks.get(ix).update(ix, self.nodeLayerBlocks.length, self.graphLayerBlocks.length)
            self.updConnections()
        }
        LayerBlock.prototype.down = function () {
            let layerBlocks = this.type === 'n' ? self.nodeLayerBlocks : self.graphLayerBlocks
            let $layerBlocksDiv = this.type === 'n' ? $nodeLayerBlocksDiv : $graphLayerBlocksDiv
            let ix = this.ix
            if (ix < layerBlocks.length - 1) {
                $layerBlocksDiv.children().eq(ix).before($layerBlocksDiv.children().eq(ix + 1))
                this.update(ix + 1, self.nodeLayerBlocks.length, self.graphLayerBlocks.length)
            }
            layerBlocks.toLast(ix)
            layerBlocks.get(ix).update(ix, self.nodeLayerBlocks.length, self.graphLayerBlocks.length)
            self.updConnections()
        }
        // LayerBlock.prototype.clone = function () {
        //     let layerBlocks = this.type === 'n' ? self.nodeLayerBlocks : self.graphLayerBlocks
        //     let ix = this.ix
        //     self.absLayersCounter++
        //     let layerBlock = this.copy(self.absLayersCounter, ix+1)
        //     $layerBlocksDiv.children().eq(ix).after(layerBlock.$div)
        //     layerBlocks.insert(layerBlock, ix+1)
        //     let iter = layerBlocks.iterator(ix+2)
        //     let result = iter.next()
        //     while (!result.done) {
        //         result.value.updateIx(result.value.ix+1)
        //         result = iter.next()
        //     }
        // }
    }

    // Form model config from selectors values
    constructModelConfig() {
        // Change layers output size to num_classes, and all its precedents if needed
        let setOutputSize = (layerBlocks) => {
            let iter = layerBlocks.reverseIterator()
            let result = iter.next()
            let done
            while (!result.done) {
                done = result.value.setOutputSize(this.num_classes)
                if (done)
                    break
                result = iter.next()
            }
            if (!done && this.num_feats !== this.num_classes) {
                // Prevent creating only layers which do not define sizes enough
                alert('Invalid GNN structure: at least one layer should have modifiable output dimension!')
                return null
            }
        }

        // Last layer must have Activation=LogSoftmax and outsize=number of classes
        if (this.multi) {
            if (this.customGraphLayerBlock) {
                this.customGraphLayerBlock.setAsLast()
                let done = this.customGraphLayerBlock.setOutputSize(this.num_classes)
                if (!done)
                    // TODO setOutputSize for previous layers
                    console.error('Not implemented')
            }
            else {
                this.graphLayerBlocks.last().setAsLast()
                let done = this.graphLayerBlocks.last().setOutputSize(this.num_classes)
                if (!done)
                    // TODO setOutputSize for previous layers
                    console.error('Not implemented')
            }
            this.nodeLayerBlocks.last().setAsLast()
        }
        else {
            setOutputSize(this.nodeLayerBlocks)
            this.nodeLayerBlocks.last().setAsLast()
        }

        let architecture = []
        let additionalNodeIxes = []
        let additionalGraphIxes = []
        // node layer ix -> additional input size
        let additionalNodeSize = new Array(this.nodeLayerBlocks.length).fill(0)
        // graph layer ix -> additional input size
        let additionalGraphSize = new Array(this.graphLayerBlocks.length).fill(0)

        // Add node layers
        let ix = 0 // layer index
        let iter = this.nodeLayerBlocks.iterator()
        let result = iter.next()
        let cfg
        let inputSize = this.num_feats
        while (!result.done) {
            let outputSize
            [cfg, outputSize, additionalNodeIxes, additionalGraphIxes]
                = result.value.constructConfig(inputSize + additionalNodeSize[ix])
            if (outputSize != null) // If out_channels is defined, use it as next in_channels
                inputSize = outputSize
            for (const i of additionalNodeIxes)
                additionalNodeSize[i] += inputSize
            for (const i of additionalGraphIxes)
                additionalGraphSize[i] += inputSize
            architecture.push(cfg)
            result = iter.next()
            ix++
        }

        // Add graph layers
        if (this.multi) {
            inputSize = 0
            ix = 0 // layer index
            iter = this.graphLayerBlocks.iterator()
            result = iter.next()
            while (!result.done) {
                [cfg, inputSize, additionalNodeIxes, additionalGraphIxes]
                    = result.value.constructConfig(inputSize + additionalGraphSize[ix])
                for (const i of additionalGraphIxes)
                    additionalGraphSize[i] += inputSize
                architecture.push(cfg)
                result = iter.next()
                ix++
            }

            // Add last prot layer if present
            if (this.customGraphLayerBlock) {
                [cfg, inputSize, additionalNodeIxes, additionalGraphIxes]
                    = this.customGraphLayerBlock.constructConfig(inputSize)
                // for (const i of additionalGraphIxes)
                //     additionalGraphSize[i] += inputSize
                architecture.push(cfg)
            }
        }
        return architecture
    }
}

