class ForceNeighborhoodLayout extends Layout {
    constructor(alpha=30, beta=20, minV=0.01, visc=0.85, temp=15, rad=1.5) {
        super()

        this.alpha = alpha
        this.beta = beta
        this.visc = visc
        this.minV = minV
        this.temp = temp
        this.rad = rad
        this.stateFlag = 2
        this.constT = temp / 100

        this.thrA = 150
        this.decayFactor = 0.98
    }

    getLayoutType() {
        return 'forceNeighborhood'
    }

    serializeParams() {
        return {
            alpha: this.alpha,
            beta: this.beta,
            visc: this.visc,
            minV: this.minV,
            temp: this.temp,
            rad: this.rad,
            stateFlag: this.stateFlag,
            constT: this.constT,
            thrA: this.thrA,
            decayFactor: this.decayFactor
        }
    }

    serializeGraphData() {
        const layers = {}
        for (const [depth, nodes] of Object.entries(this.visibleGraph.nodes)) {
            layers[depth] = nodes.map(n => this.nodeToIndex.get(n))
        }

        const edgesByDepth = {}
        for (const [depth, edges] of Object.entries(this.visibleGraph.edges)) {
            const arr = new Uint32Array(edges.length * 2)
            for (let e = 0; e < edges.length; e++) {
                arr[2 * e] = this.nodeToIndex.get(edges[e][0])
                arr[2 * e + 1] = this.nodeToIndex.get(edges[e][1])
            }
            edgesByDepth[depth] = arr
        }

        return {
            depth: this.visibleGraph.depth,
            n0: this.visibleGraph.n0.map(n => this.nodeToIndex.get(n)),
            layers,
            edgesByDepth
        }
    }
}