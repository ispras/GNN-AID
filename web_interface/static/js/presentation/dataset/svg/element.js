const MAX_FEATURES_SHOWN = 10
const MIN_NODE_RADIUS = 5

// Group of SVG primitives representing some element (node, edge, graph frame, etc)
class SvgElement {
    // Get scaled node radius
    static scaledRadius(radius, scale) {
        return Math.max(MIN_NODE_RADIUS, Math.ceil(radius * (scale/100) ** 0.5))
    }
    static MAX_FEATURES_SHOWN = MAX_FEATURES_SHOWN

    constructor(x, y, r, color, show, $tip) {
        // X position
        this.x = x
        // Y position
        this.y = y
        // Size
        this.r = r
        // Color
        this.color = color
        // scale
        this.s = 1
        // Flag whether all is shown or not
        this.show = show

        // Graphics container element
        this.g = document.createElementNS('http://www.w3.org/2000/svg', 'g')
        this.g.setAttribute('display', show ? "inline" : "none")
        this.gSat = {}
        for (const satellite of VisibleGraph.SATELLITES) {
            let g = document.createElementNS('http://www.w3.org/2000/svg', 'g')
            g.setAttribute("id", satellite)
            this.gSat[satellite] = g
            this.g.appendChild(g)
        }

        this.maxFeaturesShown = MAX_FEATURES_SHOWN

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
            case 'logits':
                return this.setLogits.apply(this, args)
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

    setFeatures(feats) {
        if (feats == null)
            console.error('null')
        let r = SvgElement.scaledRadius(this.r, this.s)
        let size = 0.8 * r
        let features = this.satellites['features']
        features.blocks = []
        let tipText = "Feature:"
        if (feats.length > this.maxFeaturesShown) {
            let rect = document.createElementNS("http://www.w3.org/2000/svg", "rect")
            let x = features.placeX(0, r, feats.length)
            let y = features.placeY(0, r, feats.length)
            rect.setAttribute('x', x)
            rect.setAttribute('y', y)
            rect.setAttribute('width', size)
            rect.setAttribute('height', size)
            let color = 'rgb(255,255,255)'
            rect.setAttribute('fill', color)
            rect.setAttribute('stroke', '#ffffff')
            rect.setAttribute('stroke-width', 1)
            rect.setAttribute('display', !this.lightMode && this.show ? "inline" : "none")
            features.blocks.push(rect)
            tipText += '<br>' + this._featureTipText(feats)
        }
        else {
            for (let i = 0; i < feats.length; i++) {
                let rect = document.createElementNS("http://www.w3.org/2000/svg", "rect")
                let x = features.placeX(i, r, feats.length)
                let y = features.placeY(i, r, feats.length)
                rect.setAttribute('x', x)
                rect.setAttribute('y', y)
                rect.setAttribute('width', size)
                rect.setAttribute('height', size)
                let color = valueToColor(feats[i], EMBEDDING_COLORMAP, -2, 2, true, 0.2)
                rect.setAttribute('fill', color)
                rect.setAttribute('stroke', '#000')
                rect.setAttribute('stroke-width', 1)
                rect.setAttribute('display', !this.lightMode && this.show ? "inline" : "none")
                features.blocks.push(rect)
            }
            tipText += feats.reduce((a, c) => a + '<br>' + c.toFixed(5), '')
        }
        this._addTip(features.blocks, "feature")
        this.tipText["feature"] = tipText

        this.gSat['features'].innerHTML = ''
        for (const e of features.blocks)
            this.gSat['features'].appendChild(e)

        return true
    }

    setScores(values) {
        let r = SvgElement.scaledRadius(this.r, this.s)
        let size = 0.8 * r
        let scores = this.satellites['scores']
        scores.blocks = []
        let tipText = "Scores:"
        if (values.length > MAX_FEATURES_SHOWN) {
            let rect = document.createElementNS("http://www.w3.org/2000/svg", "rect")
            let x = scores.placeX(0, r, values.length)
            let y = scores.placeY(0, r, values.length)
            rect.setAttribute('x', x)
            rect.setAttribute('y', y)
            rect.setAttribute('width', size)
            rect.setAttribute('height', size)
            let color = 'rgb(255,255,255)'
            rect.setAttribute('fill', color)
            rect.setAttribute('stroke', '#ffffff')
            rect.setAttribute('stroke-width', 1)
            rect.setAttribute('display', !this.lightMode && this.show ? "inline" : "none")
            scores.blocks.push(rect)
            tipText += '<br>' + this._featureTipText(values)
        }
        else {
            for (let i = 0; i < values.length; i++) {
                let rect = document.createElementNS("http://www.w3.org/2000/svg", "rect")
                let x = scores.placeX(i, r, values.length)
                let y = scores.placeY(i, r, values.length)
                rect.setAttribute('x', x)
                rect.setAttribute('y', y)
                rect.setAttribute('width', size)
                rect.setAttribute('height', size)
                let color = valueToColor(values[i], EMBEDDING_COLORMAP, -2, 2, true, 0.2)
                rect.setAttribute('fill', color)
                rect.setAttribute('stroke', '#000')
                rect.setAttribute('stroke-width', 1)
                rect.setAttribute('display', !this.lightMode && this.show ? "inline" : "none")
                scores.blocks.push(rect)
            }
            tipText += values.reduce((a, c) => a + '<br>' + c.toFixed(5), '')
        }
        this._addTip(scores.blocks, "scores")
        this.tipText["scores"] = tipText
        return true
    }

    // Add node labels values
    setLabels(classIndex, numClasses) {
        if (!numClasses) {
            console.log('numClasses is null. Labels will not be created')
            return
        }
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

        this.gSat['labels'].innerHTML = ''
        for (const e of labels.blocks)
            this.gSat['labels'].appendChild(e)

        return true
    }

    // Add node predictions
    setPredictions(preds) {
        if (!Array.isArray(preds))
            preds = [preds]
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

        this.gSat['predictions'].innerHTML = ''
        for (const e of predictions.blocks)
            this.gSat['predictions'].appendChild(e)
        predictions.blocks = null

        return createNew
    }

    // Add node logits
    setLogits(embeds) {
        if (!Array.isArray(embeds))
            embeds = [embeds]
        let logits = this.satellites['logits']
        let createNew = logits.blocks == null
        if (createNew) {
            let r = SvgElement.scaledRadius(this.r, this.s)
            let size = 0.8 * r
            logits.blocks = []
            for (let i = 0; i < embeds.length; i++) {
                let rect = document.createElementNS("http://www.w3.org/2000/svg", "rect")
                rect.setAttribute('x', logits.placeX(i, r, embeds.length))
                rect.setAttribute('y', logits.placeY(i, r, embeds.length))
                rect.setAttribute('width', size)
                rect.setAttribute('height', size)
                rect.setAttribute('stroke', '#000')
                rect.setAttribute('stroke-width', 1)
                rect.setAttribute('display', !this.lightMode && this.show ? "inline" : "none")
                logits.blocks.push(rect)
            }
            this._addTip(logits.blocks, "logit")
        }
        // Set colors
        // TODO do not update if not visible. But then how to update when it gets visible??
        for (let i=0; i<embeds.length; i++) {
            let rect = logits.blocks[i]
                let color = valueToColor(embeds[i], EMBEDDING_COLORMAP, -2, 2, true, 0.2)
            rect.setAttribute('fill', color)
        }
        let tipText = "logit:" + embeds.reduce((a, c) => a + '<br>' + c.toFixed(5), '')
        this.tipText["logit"] = tipText
        // Update tip if shown
        if (this.tipShown === "logit")
            this.$tip.html(tipText)

        this.gSat['logits'].innerHTML = ''
        for (const e of logits.blocks)
            this.gSat['logits'].appendChild(e)
        logits.blocks = null

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

        this.gSat['train-test-mask'].innerHTML = ''
        for (const e of trainmask.blocks)
            this.gSat['train-test-mask'].appendChild(e)
        trainmask.blocks = null

        return createNew
    }

    // Apply translation to (x, y) transform to graphics container
    translate(x, y) {
        this.g.setAttribute('transform', 'translate(' + x + ',' + y + ')')
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
        if (show === this.show)
            return
        this.show = show
        this.g.setAttribute('display', show ? "inline" : "none")
    }

    removeSatellites() {
        for (const g of Object.values(this.gSat))
            g.innerHTML = ''
    }
}