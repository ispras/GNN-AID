// SVG primitives for a graph frame + labels
class SvgGraph extends SvgElement {
    constructor(x, y, width, height, color, text, show, $tip) {
        super(x, y, 30, color, show, $tip)
        this.width = width
        this.height = height
        this.color = color // default color

        this.frame = document.createElementNS("http://www.w3.org/2000/svg", "rect")
        this.frame.setAttribute('x', x)
        this.frame.setAttribute('y', y)
        this.frame.setAttribute('width', width)
        this.frame.setAttribute('height', height)
        this.frame.setAttribute('stroke', color)
        this.frame.setAttribute('fill', 'rgba(255,255,255,0)')
        this.frame.setAttribute('stroke-width', 1)
        this.frame.setAttribute('display', this.show)

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

        let labels = this.satellites['labels'] = new Satellite("circle", this.r)
        labels.placeX = (ix, r, count) => this.x + this.width + 0.8 * r*(-count + 1/2 + ix)
        labels.placeY = (ix, r, count) => this.y - r/2

        let predictions = this.satellites['predictions'] = new Satellite("circle", this.r)
        predictions.placeX = (ix, r, count) => this.x + this.width + 0.8 * r*(-count + 1/2 + ix)
        predictions.placeY = (ix, r, count) => this.y + r/2

        let embeddings = this.satellites['embeddings'] = new Satellite("rect", this.r)
        embeddings.placeX = (ix, r, count) => this.x + this.width + 0.1 * r
        embeddings.placeY = (ix, r, count) => this.y + this.height/2 + 0.8 * r * (-count + 1/2 + ix)

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
        this.x = x
        this.y = y
        this.width = width
        this.height = height
        this.frame.setAttribute('x', this.x)
        this.frame.setAttribute('y', this.y)
        this.frame.setAttribute('width', this.width)
        this.frame.setAttribute('height', this.height)
        this.text.setAttribute('x', this.x)
        this.text.setAttribute('y', this.y)

        // for (const satellite of Object.values(this.satellites))
        //     satellite.moveTo(x, y)
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

