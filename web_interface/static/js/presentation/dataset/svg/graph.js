// SVG primitives for a graph frame + labels
class SvgGraph extends SvgElement {
    constructor(x, y, width, height, color, text, show, $tip) {
        super(x, y, 32, color, show, $tip)
        this.width = width
        this.height = height
        this.color = color // default color

        this.frame = document.createElementNS("http://www.w3.org/2000/svg", "path")
        this.frame.setAttribute("stroke", color)
        // важно: нет заливки => нет "площади" для событий
        this.frame.setAttribute("fill", "none")
        this.frame.setAttribute("stroke-width", 2)
        this.frame.setAttribute("display", this.show)
        // чтобы события проходили сквозь внутреннюю область,
        // а обводка при этом могла оставаться интерактивной:
        this.frame.setAttribute("pointer-events", "stroke")
        this.g.appendChild(this.frame)

        this.text = document.createElementNS("http://www.w3.org/2000/svg", "text")
        this.text.textContent = text
        this.text.setAttribute('x', x)
        this.text.setAttribute('y', y)
        this.text.setAttribute('dominant-baseline', 'auto')
        this.text.setAttribute('text-anchor', 'start')
        this.text.setAttribute('fill', '#000')
        this.text.setAttribute('font-size', '10pt')
        this.text.setAttribute('pointer-events', 'none')
        this.text.setAttribute('display', this.show)
        this.g.appendChild(this.text)

        let labels = this.satellites['labels'] = new Satellite("circle", this.r)
        labels.placeX = (ix, r, count) => this.x + this.width + 0.8 * r*(-count + 1/2 + ix)
        labels.placeY = (ix, r, count) => this.y - r/2

        let features = this.satellites['features'] = new Satellite("rect", this.r)
        features.placeX = (ix, r, count) => this.x - r
        features.placeY = (ix, r, count) => this.y + this.height/2 + 0.8 * r * (-count/2 - 1/2 + ix)

        let predictions = this.satellites['predictions'] = new Satellite("circle", this.r)
        predictions.placeX = (ix, r, count) => this.x + this.width + 0.8 * r*(-count + 1/2 + ix)
        predictions.placeY = (ix, r, count) => this.y + r/2

        let logits = this.satellites['logits'] = new Satellite("rect", this.r)
        logits.placeX = (ix, r, count) => this.x + this.width + 0.1 * r
        logits.placeY = (ix, r, count) => this.y + this.height/2 + 0.8 * r * (-count/2 - 1/2 + ix)

        let trainmask = this.satellites['train-test-mask'] = new Satellite("text", this.r)
        trainmask.placeX = (ix, r, count) => this.x + r
        trainmask.placeY = (ix, r, count) => this.y + 0.61 * r

        let scores = this.satellites['scores'] = new Satellite("rect", this.r)
        scores.placeX = (ix, r, count) => this.x + 0.8 * r * ix
        scores.placeY = (ix, r, count) => this.y + this.height + 0.1 * r
    }

    move(shift) {
        this.moveTo(this.x + shift.x, this.y + shift.y)
    }

    moveTo(x, y, width, height) {
        const r = SvgElement.scaledRadius(this.r, this.s)

        this.x = x - r
        this.y = y - 1.5*r
        this.width = width + 2 * r
        this.height = height + 2.5 * r

        // прямоугольник path: M x y h w v h h -w Z
        const d = `M ${this.x} ${this.y} h ${this.width} v ${this.height} h ${-this.width} Z`
        this.frame.setAttribute("d", d)

        this.text.setAttribute("x", this.x)
        this.text.setAttribute("y", this.y)
    }

    scale(s) {
        super.scale(s)
        this.text.setAttribute(
            'font-size', `${0.8 * SvgElement.scaledRadius(this.r, s)}pt`)
    }

    // Set visibility for all elements
    visible(show) {
        this.show = show
        this.frame.setAttribute('display', show ? "inline" : "none")

        for (const satellite of Object.values(this.satellites))
            satellite.visible(!this.lightMode && show)
    }

    // Add graph scores values
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

    // // Set stroke color
    // setColor(color) {
    //     // this.circle.setAttribute('stroke', color)
    // }
    //
    // // Set fill color
    // setFillColorIdx(colorIdx) {
    //     // this.circle.setAttribute('fill',
    //     //     colorIdx >= 0 ? `url(#RadialGradient${colorIdx})`: this.color)
    // }
    //
    // // Change stroke color back to default
    // dropColor() {
    //     // this.circle.setAttribute('stroke', this.color)
    // }
    //
    // // Change fill color back to default
    // dropFillColor() {
    //     // this.circle.setAttribute('fill', this.color)
    // }
}

