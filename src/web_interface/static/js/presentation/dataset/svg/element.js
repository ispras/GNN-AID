const MAX_FEATURES_SHOWN = 10
const MIN_NODE_RADIUS = 5

// Group of SVG primitives representing some element (node, edge, graph frame, etc)
class SvgElement {
    // Get scaled node radius
    static scaledRadius(radius, scale) {
        return Math.max(MIN_NODE_RADIUS, Math.ceil(radius * (scale/100) ** 0.5))
    }

    constructor(x, y, r, color, show, $tip) {
        // X position
        this.x = x
        // Y position
        this.y = y
        // Size
        this.r = r
        // scale
        this.s = 1
        // Flag whether all is shown or not
        this.show = show

        // Tip info
        this.$tip = $tip
        this.tipText = {} // Tip text depending of object type {<type> -> text}
        this.tipShown = null // Which type of tip is currently shown

        this.lightMode = false

        // Satellites
        this.satellites = {}
    }

    // Add mouse listener for all elements which shows a tip with given text
    _addTip(elements, type) {
        let onmousemove = (e) => {
            this.$tip.show()
            this.$tip.css("left", e.clientX + 10)
            this.$tip.css("top", e.clientY + 15)
            this.tipShown = type
            this.$tip.html(this.tipText[type])
        }
        let onmouseout = (e) => {
            this.tipShown = null
            this.$tip.hide()
        }
        for (const element of elements) {
            element.onmousemove = onmousemove
            element.onmouseout = onmouseout
        }
    }

    // Several
    _featureTipText(values, valuesPerRow=10) {
        let text = ""
        for (let i = 0; i < values.length; i++) {
            text += values[i].toFixed(5)
            text += (i+1) % valuesPerRow === 0 ? "<br>" : " "
        }
        return text
    }

    setSatellite(satellite) {
        let args = Array.prototype.slice.call(arguments, 1)
        switch (satellite) {
            case 'labels':
                return this.setLabels.apply(this, args)
            case 'predictions':
                return this.setPredictions.apply(this, args)
            case 'embeddings':
                return this.setEmbeddings.apply(this, args)
            case 'train-test-mask':
                return this.setTrainMask.apply(this, args)
            case 'features':
                return this.setFeatures.apply(this, args)
            case 'scores':
                return this.setScores.apply(this, args)
            default:
                console.error('Unknown satellite: ' + satellite)
        }
    }

    setFeatures() {}

    setScores() {}

    // Add node labels values
    setLabels(classIndex, numClasses) {
        let r = SvgElement.scaledRadius(this.r, this.s)
        let size = 0.8*r
        let labels = this.satellites['labels']
        labels.blocks = []
        for (let i=0; i<numClasses; i++) {
            let includes = Array.isArray(classIndex) ? classIndex.includes(i) : i === classIndex
            labels.blocks.push(Svg.circle(
                labels.placeX(i, r, numClasses),
                labels.placeY(i, r, numClasses), size/2,
                includes ? 'rgb(0,0,0)' : 'rgb(255,255,255)', '#fff',
                !this.lightMode && this.show))
        }
        this._addTip(labels.blocks, "label")
        this.tipText["label"] = "class: " + classIndex
        return true
    }

    // Add node predictions
    setPredictions(preds) {
        let predictions = this.satellites['predictions']
        let createNew = predictions.blocks == null
        if (createNew) {
            let r = SvgElement.scaledRadius(this.r, this.s)
            let size = 0.8*r
            predictions.blocks = []
            for (let i=0; i<preds.length; i++)
                predictions.blocks.push(Svg.circle(
                    predictions.placeX(i, r, preds.length),
                    predictions.placeY(i, r, preds.length), size/2,
                    null, '#000000',
                    !this.lightMode && this.show))

            this._addTip(predictions.blocks, "prediction")
        }
        // Set colors
        // TODO do not update if not visible. But then how to update when it gets visible??
        for (let i=0; i<preds.length; i++)
            predictions.blocks[i].setAttribute('fill',
                valueToColor(preds[i], PREDICTION_COLORMAP))

        let tipText = "prediction:" + preds.reduce((a, c) => a + '<br>' + c.toFixed(5), '')
        this.tipText["prediction"] = tipText
        // Update tip if shown
        if (this.tipShown === "prediction")
            this.$tip.html(tipText)
        return createNew
    }

    // Add node embeddings
    setEmbeddings(embeds) {
        let embeddings = this.satellites['embeddings']
        let createNew = embeddings.blocks == null
        if (createNew) {
            let r = SvgElement.scaledRadius(this.r, this.s)
            let size = 0.8 * r
            embeddings.blocks = []
            for (let i = 0; i < embeds.length; i++) {
                let rect = document.createElementNS("http://www.w3.org/2000/svg", "rect")
                rect.setAttribute('x', embeddings.placeX(i, r, embeds.length))
                rect.setAttribute('y', embeddings.placeY(i, r, embeds.length))
                rect.setAttribute('width', size)
                rect.setAttribute('height', size)
                rect.setAttribute('stroke', '#000')
                rect.setAttribute('stroke-width', 1)
                rect.setAttribute('display', !this.lightMode && this.show ? "inline" : "none")
                embeddings.blocks.push(rect)
            }
            this._addTip(embeddings.blocks, "embedding")
        }
        // Set colors
        // TODO do not update if not visible. But then how to update when it gets visible??
        for (let i=0; i<embeds.length; i++) {
            let rect = embeddings.blocks[i]
                let color = valueToColor(embeds[i], EMBEDDING_COLORMAP, -2, 2, true, 0.2)
            rect.setAttribute('fill', color)
        }
        let tipText = "embedding:" + embeds.reduce((a, c) => a + '<br>' + c.toFixed(5), '')
        this.tipText["embedding"] = tipText
        // Update tip if shown
        if (this.tipShown === "embedding")
            this.$tip.html(tipText)
        return createNew
    }

    // Add node trainMask
    setTrainMask(trainMask) {
        let trainmask = this.satellites['train-test-mask']
        let createNew = trainmask.blocks == null
        if (createNew) {
            let r = SvgElement.scaledRadius(this.r, this.s)
            let size = 0.8*r
            let text = document.createElementNS("http://www.w3.org/2000/svg", "text")
            text.setAttribute('x', trainmask.placeX(0, r, text.textContent.length))
            text.setAttribute('y', trainmask.placeY(0, r, text.textContent.length))
            text.setAttribute('dominant-baseline', 'middle')
            text.setAttribute('text-anchor', 'middle')
            text.setAttribute('font-size', `${2 / 3 * size}pt`)
            text.setAttribute('pointer-events', 'none')
            text.setAttribute('display', !this.lightMode && this.show ? "inline" : "none")
            trainmask.blocks = [text]
        }
        let text = trainmask.blocks[0]
        text.textContent = {0: '', 1: "train", 2: "test", 3: "val"}[trainMask]
        text.setAttribute('fill', {0: '#000', 1: "#120", 2: "#f00", 3: "#00f"}[trainMask])
        return createNew
    }

    moveTo(x, y) {
        this.x = x
        this.y = y
    }

    scale(s) {
        this.s = s
        // this.lightMode = s < LIGHT_MODE_SCALE_THRESHOLD_SINGLE
        for (const satellite of Object.values(this.satellites))
            satellite.scale(s)
    }

    // Set visibility for all elements
    visible(show) {
        this.show = show
    }
}
