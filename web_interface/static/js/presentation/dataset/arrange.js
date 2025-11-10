class Arrange {
    constructor() {
        this.graphs = null // {ix -> Graph}
        this.margin = 1
    }

    // Set a list of graphs to arrange
    setGraphs(graphs) {
        this.graphs = graphs
    }

    _center(bbox) {
        return new Vec(bbox.x + bbox.width/2, bbox.y + bbox.height/2)
    }

    // Apply arranging algorithm
    apply() {
        // Do nothing - graphs just overlap
    }
}

class VerticalArrange extends Arrange {
    constructor() {
        super()
        this.order = null // {graph number -> graphIx}
    }

    // Set a list of graphs to arrange
    setGraphs(graphs) {
        super.setGraphs(graphs)
        this.order = {}
        let num = 0
        for (const [ix, aGraph] of Object.entries(this.graphs)) {
            this.order[num] = parseInt(ix)
            num += 1
        }
    }

    // Apply arranging algorithm, target is index of graph that should not be moved
    apply(targetGraphIx) {
        let graphBbox = {} // graphIx -> BBox
        let targetNum = 0 // number of target Graph
        let num = 0
        for (const [ix, aGraph] of Object.entries(this.graphs)) {
            graphBbox[ix] = aGraph.approxBBox()
            // graphBbox[ix] = aGraph.edgePrimitivesBatches[0].path.getBBox()
            if (this.order[num] === targetGraphIx)
                targetNum = num
            num++
        }

        let center = this._center(graphBbox[this.order[targetNum]])

        // Before targetNum
        let cumShift = new Vec(0, 0) // cumulative shift from the center
        for (let num = targetNum-1; num >= 0; num--) {
            let ixA = this.order[num]
            let ixB = this.order[num+1]
            cumShift.y -= (graphBbox[ixA].height/2 + graphBbox[ixB].height/2 + this.margin * this.graphs[ixA].scale)
            // Shift = wanted pos - real pos
            this.graphs[ixA].move(Vec.sub(Vec.add(center, cumShift), this._center(graphBbox[ixA])))
        }

        // After targetNum
        cumShift = new Vec(0, 0) // cumulative shift from the center
        for (let num = targetNum+1; num < Object.keys(this.order).length; num++) {
            let ixA = this.order[num]
            let ixB = this.order[num-1]
            cumShift.y += graphBbox[ixA].height/2 + graphBbox[ixB].height/2 + this.margin * this.graphs[ixA].scale
            // Shift = wanted pos - real pos
            this.graphs[ixA].move(Vec.sub(Vec.add(center, cumShift), this._center(graphBbox[ixA])))
        }
    }

}

class GridArrange extends Arrange {
    constructor(rows, columns) {
        super()
        this.grid = null // Array[x][y] = graphIx
        this.rows = rows
        this.columns = columns
    }

    // Set a list of graphs to arrange
    setGraphs(graphs) {
        super.setGraphs(graphs)
        let n = Object.keys(this.graphs).length
        this.columns = this.columns ? this.columns : Math.ceil(n**0.5)
        this.rows = this.rows ? this.rows : Math.ceil(n / this.columns)
        this.grid = Array(this.columns)
        let x = 0
        let y = 0
        this.grid[0] = new Array(this.rows)
        for (const [ix, aGraph] of Object.entries(this.graphs)) {
            this.grid[x][y] = parseInt(ix)
            y++
            if (y % this.rows === 0) {
                y = 0
                x++
                if (x < this.columns) this.grid[x] = new Array(this.rows)
            }
        }
    }

    // Apply arranging algorithm, target is index of graph that should not be moved
    apply(targetGraphIx) {
        let graphBbox = {} // graphIx -> BBox
        let targetX = 0 // x-grid of target Graph
        let targetY = 0 // y-grid of target Graph
        let x = 0
        let y = 0
        for (const [ix, aGraph] of Object.entries(this.graphs)) {
            graphBbox[ix] = aGraph.approxBBox()
            // graphBbox[ix] = aGraph.edgePrimitivesBatches[0].path.getBBox()
            if (this.grid[x][y] === targetGraphIx) {
                targetX = x
                targetY = y
            }
            y++
            if (y % this.rows === 0) {
                y = 0
                x++
            }
        }

        let center = this._center(graphBbox[this.grid[targetX][targetY]])

        // Define Max widths and heights
        let wx = [] // width[x]
        let hy = [] // height[y]
        for (let i=0;i<this.columns;i++) {
            let widths = []
            for (let j=0;j<this.rows;j++) {
                let ix = this.grid[i][j]
                if (ix == null) continue
                widths.push(graphBbox[ix].width)
            }
            wx.push(Math.max.apply(null, widths))
            // wx.push(Math.max(this.grid[i].map(ix => graphBbox[ix].width)))
        }
        for (let j=0;j<this.rows;j++) {
            let heights = []
            for (let i=0;i<this.columns;i++) {
                let ix = this.grid[i][j]
                if (ix == null) continue
                heights.push(graphBbox[ix].height)
            }
            hy.push(Math.max.apply(null, heights))
            // hy.push(Math.max(this.grid.map(gridI => graphBbox[gridI[j]].height)))
        }

        // Define shift(x) and shift(y)
        let shiftX = Array(this.columns)
        let shiftY = Array(this.rows)
        let margin = 2 * this.margin * this.graphs[this.grid[targetX][targetY]].scale
        let cum = 0
        shiftY[targetY] = 0
        for (let y = targetY-1; y >= 0; y--) {
            cum -= hy[y] + hy[y+1] + margin
            shiftY[y] = cum/2
        }
        cum = 0
        for (let y = targetY+1; y < this.rows; y++) {
            cum += hy[y] + hy[y-1] + margin
            shiftY[y] = cum/2
        }

        cum = 0
        shiftX[targetX] = 0
        for (let x = targetX-1; x >= 0; x--) {
            cum -= wx[x] + wx[x+1] + margin
            shiftX[x] = cum/2
        }
        cum = 0
        for (let x = targetX+1; x < this.columns; x++) {
            cum += wx[x] + wx[x-1] + margin
            shiftX[x] = cum/2
        }

        // Move all graphs
        for (let x=0; x<this.columns; x++) {
            for (let y=0; y<this.rows; y++) {
                let ix = this.grid[x][y]
                if (ix == null) continue
                this.graphs[ix].move(Vec.sub(Vec.add(center, new Vec(shiftX[x], shiftY[y])),
                    this._center(graphBbox[ix])))
            }
        }
    }

}