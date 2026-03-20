// A node with an SVG primitive to draw it
class SvgNode extends SvgElement {
    constructor(x, y, r, form, width, color, text, show, $tip) {
        super(x, y, r, color, show, $tip)
        this.form = form // form: circle, rect, polygon
        this.width = width // stroke-width
        this.color = color // default color
        this.numEdges = null // number of edges in case of a polygon form
        // this.showFeatures = true
        // this.showLabels = true
        this.showLogits = true
        this.showPredictions = true
        this.showTrainMask = true

        if (form === "circle") {
            this.body = Svg.circle(x, y, r, color, null, show)
        }
        else if (form.startsWith("poly")) {
            this.numEdges = form.slice(-1)
            this.body = Svg.polygon(x, y, r, this.numEdges, color, null, show)
        }
        else
            console.error("Unknown form")
        this.g.appendChild(this.body)

        this.body.setAttribute('stroke-width', this.width)
        this.body.setAttribute('class', 'graph-node')
        this.body.setAttribute('draggable', "true")

        this.text = Svg.text(text, x, y, 'middle', `${2 / 3 * r}pt`)
        this.text.setAttribute('text-anchor', 'middle')
        this.text.setAttribute('class', 'graph-node-text')
        this.text.setAttribute('display', show ? "inline" : "none")
        this.g.appendChild(this.text)

        // this.attrBlocks = null
        // this.attrTextBlocks = null

        let labels = this.satellites['labels'] = new Satellite("circle", this.r)
        labels.placeX = (ix, r, count) => this.x + 0.8 * r*(-count/2 + 1/2 + ix)
        labels.placeY = (ix, r, count) => this.y - 1.6 * r

        let predictions = this.satellites['predictions'] = new Satellite("circle", this.r)
        predictions.placeX = (ix, r, count) => this.x + 0.8 * r*(-count/2 + 1/2 + ix)
        predictions.placeY = (ix, r, count) => this.y + 1.6 * r
        // this.g.appendChild(predictions)

        let features = this.satellites['features'] = new Satellite("rect", this.r)
        features.placeX = (ix, r, count) => this.x - 2*r
        features.placeY = (ix, r, count) => this.y - r + ix * 0.8 * r
        // for (const b of features.blocks)
        //     this.g.appendChild(b)

        let logits = this.satellites['logits'] = new Satellite("rect", this.r)
        logits.placeX = (ix, r, count) => this.x + 1.2*r
        logits.placeY = (ix, r, count) => this.y - r + ix * 0.8 * r
        // this.g.appendChild(logits)

        let trainmask = this.satellites['train-test-mask'] = new Satellite("text", this.r)
        trainmask.placeX = (ix, r, count) => this.x
        trainmask.placeY = (ix, r, count) => this.y + 0.61 * r
        // this.g.appendChild(trainmask)
    }

    // Add tip to this node
    addTip(text) {
        this.tipText["node"] = text
        this._addTip([this.body], "node")
    }

    // Add node features values
    setFeatures(feats) {
        if (feats == null)
            console.error('null')
        let r = SvgElement.scaledRadius(this.r, this.s)
        let size = 0.8 * r
        let features = this.satellites['features']
        features.blocks = []
        let tipText = "Feature:"
        if (feats.length > MAX_FEATURES_SHOWN) {
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

    // // Add node attributes values
    // setAttributes(attributesTypes, values) {
    //     let r = this._scaledRadius()
    //     let size = 0.8*r // TODO 0.8 magic constant
    //     for (let i=0; i<values.length; i++) {
    //         // TODO create upper limit
    //         if (i >= 10) break
    //         let rect = document.createElementNS("http://www.w3.org/2000/svg", "rect")
    //         let x = this._leftX(i, r)
    //         let y = this._leftY(i, r)
    //         rect.setAttribute('x', x)
    //         rect.setAttribute('y', y)
    //         rect.setAttribute('width', size)
    //         rect.setAttribute('height', size)
    //         let color = 'rgb(255,255,255)'
    //         switch (attributesTypes[i]) {
    //             case "continuous":
    //                 color = valueToColor(values[i], IMPORTANCE_COLORMAP)
    //                 break
    //
    //             case "categorical":
    //                 let text = document.createElementNS("http://www.w3.org/2000/svg", "text")
    //                 text.textContent = values[i]
    //                 text.setAttribute('ix', i)
    //                 text.setAttribute('x', x+size)
    //                 text.setAttribute('y', y+size/2)
    //                 text.setAttribute('dominant-baseline', 'middle')
    //                 text.setAttribute('text-anchor', 'end')
    //                 // text.setAttribute('class', 'graph-node-text')
    //                 text.setAttribute('font-size', `${2 / 3 * size}pt`)
    //                 text.setAttribute('pointer-events', 'none')
    //                 text.setAttribute('display', this.show && this.showFeatures ? "inline" : "none")
    //                 this.attrTextBlocks.push(text)
    //                 break
    //
    //             case "other":
    //                 // TODO what can we show ?
    //                 break
    //         }
    //         rect.setAttribute('fill', color)
    //         rect.setAttribute('stroke', '#ffffff')
    //         rect.setAttribute('stroke-width', 1)
    //         rect.setAttribute('display', this.show && this.showFeatures ? "inline" : "none")
    //         this.attrBlocks.push(rect)
    //         rect.onmousemove = (e) => onmousemove(e)
    //         rect.onmouseout = (e) => onmouseout(e)
    //     }
    //
    //     // let tipText = "Attributes:" + values.reduce((a, c) => a + '<br>' + c, '')
    //     let tipText = "Features:" + this._attributesTipText(values)
    //     this._addTip(this.attrBlocks, tipText)
    // }


    move(shift) {
        this.moveTo(this.x + shift.x, this.y + shift.y)
    }

    moveTo(x, y) {
        this.x = x
        this.y = y
        if (this.form === "circle") {
            this.body.cx.baseVal.value = x
            this.body.cy.baseVal.value = y
            // this.body.setAttribute('cx', this.x)
            // this.body.setAttribute('cy', this.y)
        }
        else {
            this.body.setAttribute('points', svgPolygon(x, y, this.r*this.s, this.numEdges))
        }
        this.text.setAttribute('x', this.x)
        this.text.setAttribute('y', this.y)

        console.log('moveTo  = etAttribute')

        // for (const satellite of Object.values(this.satellites))
        //     satellite.moveTo(x, y)
    }

    scale(s) {
        super.scale(s)
        let r = SvgElement.scaledRadius(this.r, s)
        let size = 0.8*r // size of element
        if (this.form === "circle")
            this.body.setAttribute('r', r)
        else
            this.body.setAttribute('points', svgPolygon(this.x, this.y, r, this.numEdges))
        this.text.setAttribute('font-size', `${2 / 3 * r}pt`)
    }

    // // Set visibility for all elements
    // visible(show) {
    //     this.show = show
    //     // this.body.setAttribute('visibility', show ? "visible" : "hidden")
    //     // this.text.setAttribute('visibility', show ? "visible" : "hidden")
    //     this.body.setAttribute('display', show ? "inline" : "none")
    //     this.text.setAttribute('display', show ? "inline" : "none")
    //
    //     for (const satellite of Object.values(this.satellites))
    //         satellite.visible(!this.lightMode && show)
    // }

    // Set stroke color
    setColor(color, width) {
        this.body.setAttribute('stroke', color)
        if (width)
            this.body.setAttribute('stroke-width', width)
    }

    // Set fill color
    setFillColorIdx(colorIdx) {
        this.body.setAttribute('fill',
            colorIdx >= 0 ? `url(#RadialGradient${colorIdx})`: this.color)
    }

    // Change stroke color back to default
    dropColor() {
        this.body.setAttribute('stroke', this.color)
        this.body.setAttribute('stroke-width', this.width)
    }

    // Change fill color back to default
    dropFillColor() {
        this.body.setAttribute('fill', this.color)
    }

    // setText(text) {
    //     if (text == null)
    //         this.text.remove()
    //     else {
    //         if (this.text && this.text.textContent === text)
    //             return
    //         this.text = Svg.text(text, this.x, this.y, 'middle', `${2 / 3 * this.r}pt`)
    //         this.text.setAttribute('text-anchor', 'middle')
    //         this.text.setAttribute('class', 'graph-node-text')
    //         this.text.setAttribute('display', this.show ? "inline" : "none")
    //         this.g.appendChild(this.text)
    //     }
    // }
}

/// Create an SVG path 'points' which draws a symmetric n-polygon around point pos with radius r
function svgPolygon(x, y, r, n) {
    const angleStep = (2 * Math.PI) / n;
    let points = ""
    for (let i = 0; i < n; i++) {
        const angle = i * angleStep - Math.PI / 2; // Start from top (rotate 90 degrees)
        let xi = x + r * Math.cos(angle)
        let yi = y + r * Math.sin(angle)
        points += ` ${xi},${yi}`
    }
    return points
}

