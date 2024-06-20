///
// Full dataset data -- graph, its part (in case of large graphs), or a set of graphs.
// Keeps:
// * nodes and edges with their attributes
// * Metadata for dataset: name, type, is directed, is weighted, number and names of attributes, etc
class Dataset {
    constructor(dataset_data) {
        // Metadata
        // single graph or multiple graphs
        this.domain = null
        // dataset name as string
        this.name = null
        // whether edges are directed
        this.directed = null
        // whether edges have weights
        this.weighted = null
        // list of names of attributes
        this.attributesNames = null
        // list of attributes types
        this.attributesTypes = null
        // list of values of attributes (list of values for categorical, [low, high] for continuous)
        this.attributesValues = null
        // dict of {labeling name -> number of classes}
        this.labelingClasses = null

        // Data contents. In case of multiple graphs, each is a list of values
        this.count = null // number of graphs
        this.numNodes = null // (list of) full number of nodes
        this.numEdges = null // (list of) full number of edges
        this.adj = null // (list of) dict {i -> Set(j1, j2, ...)}
        this.adjIn = null // (list of) dict {i -> Set(j1, j2, ...)} for input edges
        this.nodeAttrs = null // (list of) dict {attr -> {node -> value}}
        this.nodeInfo = null // dict of additional nodes info
        // this.labelings = null // (list of) dict {labeling -> {node -> class}}

        if (dataset_data != null)
            this.setData(dataset_data)
    }

    isMultipleGraphs() {
        return this.count > 1
    }

    setData(datasetData) {
        // console.log(dataset_data)
        // TODO extend to set of graphs dataset
        let info = datasetData['info']
        this.domain = info['domain']
        this.name = info['name']
        this.numNodes = info['nodes']
        this.directed = info['directed'] // TODO parse bool
        // this.weighted = info['weighted']
        if ('node_attributes' in info) {
            this.attributesNames = info['node_attributes']['names']
            this.attributesTypes = info['node_attributes']['types']
            // if (this.attributesTypes == null)
            //     this.attributesTypes = Array(this.attributesNames.length).fill(false)
            this.attributesValues = info['node_attributes']['values']
            // if (this.attributesValues == null)
            //     this.attributesValues = Array(this.attributesNames.length).fill('?')
        }
        else {
            this.attributesNames = []
            this.attributesTypes = []
            this.attributesValues = []
        }
        // TODO add edge labels

        this.labelingClasses = info['labelings']

        this.count = info['count']
        if (this.isMultipleGraphs()) {
            this.adj = []
            if (this.directed)
                this.adjIn = []
            this.numEdges = []
            // Create adjacency dicts (bidirectional)
            for (const edges of datasetData['edges']) {
                let adj = {}
                let adjIn = {}
                let numEdges = 0
                for (const [i, j] of edges) {
                    this._addEdge(adj, i, j)
                    if (this.directed)
                        this._addEdge(adjIn, j, i)
                    numEdges += 1
                }
                this.adj.push(adj)
                if (this.directed)
                    this.adjIn.push(adjIn)
                this.numEdges.push(numEdges)
            }
        }
        else {
            this.adj = {}
            if (this.directed)
                this.adjIn = {}
            this.numEdges = 0
            // Create adjacency dict (bidirectional)
            for (const [i, j] of datasetData['edges'][0]) {
                this._addEdge(this.adj, i, j)
                if (this.directed)
                    this._addEdge(this.adjIn, j, i)
                this.numEdges += 1
            }
        }
        // Handle attributes
        this.nodeAttrs = datasetData["node_attributes"]
        this.nodeInfo = info["node_info"]
    }

    _addEdge = function (adj, i, j) {
        if (!(i in adj))
            adj[i] = new Set()
        adj[i].add(j)
        if (!this.directed) { // Add reciprocal edge
            if (!(j in adj))
                adj[j] = new Set()
            adj[j].add(i)
        }
    }

    // Get a list of attributes values for a node (of a graph)
    getNodeAttrs(node, graph=0) {
        if (this.nodeAttrs == null)
            return null
        let res = []
        for (const a of this.attributesNames) {
            let attrs = this.nodeAttrs[a][graph]
            res.push(attrs[node])
        }
        // let attrs = this.nodeAttrs[graph]
        // for (const a of this.attributesNames)
        //     res.push(attrs[a][node])
        return res
    }

    getDegree(node, graph) {
        // TODO
    }

    // // Get a statistics of possible attribute values
    // getNodeAttrValues() {
    //     let attrVals = {}  // attr -> Set(v1, v2,...)
    //     for (const a of Object.keys(this.nodeAttrs))
    //         attrVals[a] = new Set()
    //
    //     for (const [a, nodeVal] of Object.entries(this.nodeAttrs))
    //         for (const [node, val] of Object.entries(nodeVal))
    //             attrVals[a].add(val)
    //
    //     return attrVals
    // }

    // Get information HTML
    getInfo() {
        // TODO extend
        // let N = this.numNodes
        // let E = this.numEdges
        let html = ''
        html += `Name: ${this.name}`
        // html += `<br>Size: ${N} nodes, ${E} edges`
        html += `<br>Directed: ${this.directed}`
        // html += `<br>Weighted: ${this.weighted}`
        html += `<br>Attributes:`

        // List attributes or their number
        if (this.attributesNames.length > 30)
            html += ` ${this.attributesNames.length} (not shown)`
        else {
            let $attrList = $("<ul></ul>")
            for (let i = 0; i < Math.min(10, this.attributesNames.length); i++) {
                let item = '"' + this.attributesNames[i] + '"'
                item += ' - ' + this.attributesTypes[i]
                item += ', values: [' + this.attributesValues[i] + ']'
                let $item = $("<li></li>").text(item)
                $attrList.append($item)
            }
            html += $attrList.prop('outerHTML')
        }
        return html
    }

    // Get information about the node
    getNodeInfo(node, graph) {
        if (this.nodeInfo == null) return ''
        let res = {}
        if (this.isMultipleGraphs()) {
            for (const [attr, info] of Object.entries(this.nodeInfo))
                if (graph in info)
                    res[attr] = info[graph][node]
        }
        else {
            for (const [attr, info] of Object.entries(this.nodeInfo))
                if (node in info)
                    res[attr] = info[node]
        }
        return JSON.stringify(res)
    }

    getNeighborhood(node, graph) {
        // TODO extend to Multiple
        if (this.isMultipleGraphs()) {
            console.error("Can't get Neighborhood of Multiple Graphs")
            return
        }
        let n1 = new Set()
        if (this.directed) {
            if (node in this.adj)
                this.adj[node].forEach(n1.add, n1)
            if (node in this.adjIn)
                this.adjIn[node].forEach(n1.add, n1)
        }
        else
            n1 = new Set(this.adj[node])

        let n2 = new Set()
        let e01 = {}
        let e10 = {}
        e01[node] = new Set(this.adj[node])
        let e11 = {}
        let e12 = {}
        let e21 = {}
        let e22 = {}
        for (const n of n1) {
            if (n in this.adj)
                for (const nn of this.adj[n]) {
                    if (n1.has(nn)) {
                        this._addEdge(e11, n, nn)
                    }
                    else if (nn === node) {
                        if (this.directed)
                            e10[n] = new Set([node])
                    }
                    else {
                        n2.add(nn)
                        this._addEdge(e12, n, nn)
                    }
                }
            if (this.directed && n in this.adjIn)
                for (const nn of this.adjIn[n]) { // ingoing edges
                    if (n1.has(nn)) {
                        this._addEdge(e11, nn, n)
                    } else if (nn !== node) {
                        n2.add(nn)
                        this._addEdge(e21, nn, n)
                    }
                }
        }
        for (const n of n2) {
            if (n in this.adj)
                for (const nn of this.adj[n]) {
                    if (n2.has(nn))
                        this._addEdge(e22, n, nn)
                }
            if (this.directed && n in this.adjIn)
                for (const nn of this.adjIn[n]) { // ingoing edges
                    if (n2.has(nn))
                        this._addEdge(e22, nn, n)
                }
        }
        return [node, n1, n2, e01, e10, e11, e12, e21, e22]
    }

    getGraph(graph) {
        if (this.isMultipleGraphs())
            return [this.numNodes[graph], this.adj[graph], this.adjIn ? this.adjIn[graph] : null]
        else
            return [this.numNodes[0], this.adj, this.adjIn]
    }

    // Get a subgraph induced on a given set of nodes
    inducedSubgraph(nodes, graph, asEdgeList=false) {
        let adj = this.adj
        let adjIn = this.adjIn
        if (this.isMultipleGraphs()) {
            adj = adj[graph]
            if (this.directed)
                adjIn = adjIn[graph]
        }
        let newAdj = {}
        let newAdjIn = {}

        for (const n of nodes) {
            newAdj[n] = new Set()
            for (const n1 of adj[n]) {
                if (nodes.has(n1))
                    newAdj[n].add(n1)
            }
            if (this.directed) {
                newAdjIn[n] = new Set()
                for (const n1 of adjIn[n]) {
                    if (nodes.has(n1))
                        newAdjIn[n].add(n1)
                }
            }
        }
        if (asEdgeList) {
            // Return result as edge list
            let edgeList = []
            for (const [n, ns] of Object.entries(newAdj))
                for (const n1 of ns)
                    edgeList.push([n, n1])
            if (this.directed)
                for (const [n, ns] of Object.entries(newAdjIn))
                    for (const n1 of ns)
                        edgeList.push([n, n1])
            return edgeList
        }
        else
            return [nodes.size, newAdj, newAdjIn]
    }
}
