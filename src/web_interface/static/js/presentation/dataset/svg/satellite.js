class Satellite {
    constructor(type, r) {
        this.type = type
        this.r = r
        this.count = null
        this.blocks = null
        this.show = null
    }

    // X coordinate to place
    placeX(ix, r, count) {}

    // Y coordinate to place
    placeY(ix, r, count) {}

    moveTo() {
        if (this.show && this.blocks) {
            let count = this.blocks.length
            if (this.type === 'circle')
                for (let i = 0; i < this.blocks.length; i++) {
                    this.blocks[i].setAttribute('cx', this.placeX(i, this.r, count))
                    this.blocks[i].setAttribute('cy', this.placeY(i, this.r, count))
                }
            else if (this.type === 'rect' || this.type === 'text')
                for (let i = 0; i < this.blocks.length; i++) {
                    this.blocks[i].setAttribute('x', this.placeX(i, this.r, count))
                    this.blocks[i].setAttribute('y', this.placeY(i, this.r, count))
                }
        }
    }

    scale(s) {
        let r = SvgElement.scaledRadius(this.r, s)
        let size = 0.8*r // size of element
        // this.lightMode = s < LIGHT_MODE_SCALE_THRESHOLD
        if (this.show && this.blocks) {
            let count = this.blocks.length
            if (this.type === 'circle')
                for (let i = 0; i < this.blocks.length; i++) {
                    this.blocks[i].setAttribute('cx', this.placeX(i, r, count))
                    this.blocks[i].setAttribute('cy', this.placeY(i, r, count))
                    this.blocks[i].setAttribute('r', size / 2)
                }
            else if (this.type === 'rect')
                for (let i = 0; i < this.blocks.length; i++) {
                    this.blocks[i].setAttribute('x', this.placeX(i, r, count))
                    this.blocks[i].setAttribute('y', this.placeY(i, r, count))
                    this.blocks[i].setAttribute('width', size)
                    this.blocks[i].setAttribute('height', size)
                }
            else if (this.type === 'text')
                for (let i = 0; i < this.blocks.length; i++) {
                    this.blocks[i].setAttribute('x', this.placeX(i, r, count))
                    this.blocks[i].setAttribute('y', this.placeY(i, r, count))
                    this.blocks[i].setAttribute('font-size', `${2 / 3 * size}pt`)
                }
        }
    }

    visible(show) {
        // this.show = show
        // let vis = !this.lightMode && this.show
        if (this.blocks)
            for (let i=0; i<this.blocks.length; i++)
                this.blocks[i].setAttribute('display', this.show && show ? "inline" : "none")
    }

}
