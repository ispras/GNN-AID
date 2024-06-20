const EDGE_ARROW_SIZE = 10

// An edge with an SVG primitive to draw it
class SvgEdge extends SvgElement{
    constructor(x1, x2, y1, y2, color, width, directed, show) {
        super(x1, y1, 1, color, show, null)
        this.width = width
        this.directed = directed

        this.path = document.createElementNS("http://www.w3.org/2000/svg", "path")
        this.path.setAttribute('d', `M${x1},${y1} L${x2},${y2}`)
        this.path.setAttribute('fill', "rgba(255,255,255,0)")
        this.path.setAttribute('stroke', color)
        this.path.setAttribute('display', show ? "inline" : "none")
        this.path.setAttribute('stroke-width', width)
    }

    moveTo(x1, y1, x2, y2) {
        let d
        if (this.directed)
            d = svgEdge(new Vec(x1, y1), new Vec(x2, y2), this.directed)
        else
            d = `M${x1},${y1} L${x2},${y2}`
        this.path.setAttribute('d', d)
    }

    visible(show) {
        this.show = show
        this.path.setAttribute('display', show ? "inline" : "none")
    }

    // Set stroke color
    setColor(color) {
        this.path.setAttribute('stroke', color)
    }

    // Change color back to default
    dropColor() {
        this.path.setAttribute('stroke', this.color)
    }
}

// Curvature of an arc for a directed edge ( =2*sin(alpha/2), 0 < alpha < PI)
CURVATURE = 0.5

/// Create an SVG path 'd' which draws an edge from pos1 to pos2
function svgEdge(pos1, pos2, directed, arrow=true) {
    if (!directed)
        return ` M${pos1.x},${pos1.y} L${pos2.x},${pos2.y}`

    // Arc
    let dist = pos1.dist(pos2)
    let rad = dist / CURVATURE
    let d = ` M${pos1.x},${pos1.y} A ${rad} ${rad} 0 0 1 ${pos2.x} ${pos2.y}`
    if (arrow) {
        // Arrow
        // TODO use approx?
        let p = Vec.sub(pos2, pos1).mul(0.5)
        let p_len = p.abs()
        if (p_len === 0) // Points coincide
            return ''
        let len = (rad - (rad * rad - dist * dist / 4) ** 0.5)
        let l = new Vec(p.y / p_len, -p.x / p_len)
        let h = Vec.add(pos1, Vec.add(p, Vec.mul(l, len)))
        let theta = 3 / 4
        let al = Vec.rotate(l, Math.PI + theta).mul(EDGE_ARROW_SIZE)
        let ar = Vec.rotate(l, -theta).mul(EDGE_ARROW_SIZE)
        d += ` M${h.x + al.x},${h.y + al.y} L${h.x},${h.y} L${h.x + ar.x},${h.y + ar.y}`
    }
    return d
}

/// Create an SVG path 'd' which draws a self-loop edge
function svgSelfLoop(pos, directed, scale=1, arrow=true) {
    let r = Math.max(MIN_NODE_RADIUS, Math.ceil(2 * (scale ** 0.5))) // double of node radius
    let x = pos.x
    let y = pos.y - r
    let d = ` M${x},${y} m-${r},0 a${r},${r} 0 1,0 ${2*r},0 a${r},${r} 0 1,0 -${2*r},0`
    if (directed && arrow) {
        // Add arrow
        y -= r
        let x1 = x - EDGE_ARROW_SIZE * 0.7
        let y1 = y - EDGE_ARROW_SIZE * 0.7
        let y2 = y + EDGE_ARROW_SIZE * 0.7
        d += ` M${x1},${y1} L${x},${y} L${x1},${y2}`
    }
    return d

}

// An edge with an SVG primitive to draw it
class SvgEdgeBatch {
    constructor(batch, color, width, directed, show) {
        this.color = color
        this.width = width
        this.directed = directed
        this.show = show

        this.pos = null // node -> Vec
        this.batch = batch
        this.scale = 1 // scale
        this.lightMode = false // when true, not drawing arrows and self-loops
        let d = this._d()

        this.path = document.createElementNS("http://www.w3.org/2000/svg", "path")
        this.path.setAttribute('d', d)
        // this.path.setAttribute('d', `M${x1},${y1} L${x2},${y2}`)
        this.path.setAttribute('stroke', color)
        this.path.setAttribute('fill', "rgba(255,255,255,0)")
        this.path.setAttribute('display', show ? "inline" : "none")
        this.path.setAttribute('stroke-width', width)
    }

    _d(pos) {
        let d = ''
        for (let [i, j] of this.batch) {
            if (!this.directed && i > j)
                [i, j] = [j, i]
            if (pos) {
                if (i === j && !this.lightMode)
                    d += svgSelfLoop(pos[i], this.directed, this.scale)
                else
                    d += svgEdge(pos[i], pos[j], this.directed, !this.lightMode)
            }
            else
                d += ' M0,0 L0,0'
        }
        return d
    }

    move(shift) {
        if (this.pos == null) return
        for (const pos of Object.values(this.pos)) {
            pos.add(shift)
        }
        this.moveTo(this.pos)
    }

    moveTo(pos) {
        this.pos = pos
        this.path.setAttribute('d', this._d(pos))
    }

    setScale(scale) {
        this.scale = scale
        let width = this.width * Math.max(0.1, Math.min(1, scale / 150))
        // this.lightMode = scale < LIGHT_MODE_SCALE_THRESHOLD_SINGLE
        this.path.setAttribute('stroke-width', width)
    }

    visible(show) {
        this.show = show
        this.path.setAttribute('display', show ? "inline" : "none")
    }

    // Set stroke color
    setColor(color) {
        this.path.setAttribute('stroke', color)
    }

    // Change color back to default
    dropColor() {
        this.path.setAttribute('stroke', this.color)
    }
}
