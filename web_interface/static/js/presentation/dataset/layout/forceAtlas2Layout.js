// ─── Quadtree ────────────────────────────────────────────────────────────────

const QUAD_MAX_DEPTH = 64

class QuadNode {
    constructor(x, y, size, depth = 0) {
        this.x = x
        this.y = y
        this.size = size
        this.depth = depth

        this.totalMass = 0
        this.count = 0       // количество узлов графа в ячейке — нужно для репульсии
        this.cx = 0
        this.cy = 0

        this.nodeId = -1
        this.bucket = null   // Array<id> при depth >= QUAD_MAX_DEPTH
        this.children = null
    }

    insert(id, pos, m) {
        if (this.totalMass === 0) {
            this.nodeId = id
            this.totalMass = m[id]
            this.count = 1
            this.cx = pos[id].x
            this.cy = pos[id].y
            return
        }

        // Обновляем центр масс и счётчик
        const newMass = this.totalMass + m[id]
        this.cx = (this.cx * this.totalMass + pos[id].x * m[id]) / newMass
        this.cy = (this.cy * this.totalMass + pos[id].y * m[id]) / newMass
        this.totalMass = newMass
        this.count++

        // ── Корзина на максимальной глубине ──────────────────────────────────
        if (this.depth >= QUAD_MAX_DEPTH) {
            if (this.bucket === null) {
                this.bucket = (this.nodeId >= 0) ? [this.nodeId] : []
                this.nodeId = -1
            }
            this.bucket.push(id)
            return
        }

        if (this.nodeId >= 0) {
            this._subdivide()
            this._insertIntoChild(this.nodeId, pos, m)
            this.nodeId = -1
        }

        this._insertIntoChild(id, pos, m)
    }

    _subdivide() {
        const h = this.size / 2
        const d = this.depth + 1
        this.children = [
            new QuadNode(this.x,     this.y + h, h, d), // NW
            new QuadNode(this.x + h, this.y + h, h, d), // NE
            new QuadNode(this.x,     this.y,     h, d), // SW
            new QuadNode(this.x + h, this.y,     h, d), // SE
        ]
    }

    _insertIntoChild(id, pos, m) {
        const h = this.size / 2
        const qx = pos[id].x >= this.x + h ? 1 : 0
        const qy = pos[id].y >= this.y + h ? 1 : 0
        this.children[qy * 2 + qx].insert(id, pos, m)
    }

    // Barnes-Hut обход.
    // Аппроксимация суперузлом умножает силу на count (количество узлов в ячейке),
    // а НЕ на totalMass. totalMass = сумма степеней узлов — она нужна только для
    // правильного центра масс. Для репульсии важно число источников, не их «вес».
    applyRepulsion(id, pos, force_r, theta, cutoff, v) {
        if (this.totalMass === 0) return

        // ── Корзина (практически совпадающие координаты) ─────────────────────
        if (this.bucket !== null) {
            for (const bid of this.bucket) {
                if (bid === id) continue
                const dx = pos[id].x - pos[bid].x
                const dy = pos[id].y - pos[bid].y
                const dist2 = dx * dx + dy * dy
                if (dist2 < 1e-10) continue
                const dist = Math.sqrt(dist2)
                if (dist > cutoff) continue
                const f = force_r(dist) / dist
                v.x += dx * f
                v.y += dy * f
            }
            return
        }

        if (this.nodeId === id) return

        const dx = pos[id].x - this.cx
        const dy = pos[id].y - this.cy
        const dist2 = dx * dx + dy * dy
        if (dist2 === 0) return

        const dist = Math.sqrt(dist2)

        // Ранний выход: ближайшая точка ячейки дальше cutoff.
        // Ячейка квадратная, поэтому самая близкая её точка к pos[id]
        // не ближе чем dist - size*√2/2. Если даже это дальше cutoff — пропускаем.
        if (dist - this.size * 0.707 > cutoff) return

        // Критерий Barnes-Hut
        if (this.children === null || this.size / dist < theta) {
            const f = force_r(dist) * this.count / dist
            v.x += dx * f
            v.y += dy * f
            return
        }

        for (const child of this.children) {
            child.applyRepulsion(id, pos, force_r, theta, cutoff, v)
        }
    }
}

// Построить квадродерево по текущим позициям
function buildQuadtree(nodes, pos, m) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    for (const n of nodes) {
        if (pos[n].x < minX) minX = pos[n].x
        if (pos[n].y < minY) minY = pos[n].y
        if (pos[n].x > maxX) maxX = pos[n].x
        if (pos[n].y > maxY) maxY = pos[n].y
    }

    const pad = 1
    const size = Math.max(maxX - minX, maxY - minY) + pad * 2
    const root = new QuadNode(minX - pad, minY - pad, size)

    for (const n of nodes) {
        root.insert(n, pos, m)
    }
    return root
}

// ─── ForceAtlas2 + Barnes-Hut ────────────────────────────────────────────────

class ForceAtlas2Layout extends Layout {
    /**
     * @param {number} alpha   - коэффициент репульсии
     * @param {number} beta    - резерв
     * @param {number} minV    - минимальная скорость (порог остановки)
     * @param {number} visc    - вязкость (сохранён для совместимости)
     * @param {number} temp    - начальная температура (макс. смещение за шаг)
     * @param {number} rad     - масштаб сил: меньше rad → короче рёбра
     * @param {number} theta   - параметр Barnes-Hut: 0.5 точно, 1.2 быстро
     * @param {boolean} linLog - LinLog mode: f_a = log(1+d), лучше для неоднородных графов
     */
    constructor(alpha=30, beta=20, minV=0.01, visc=0.85, temp=15, rad=1.5, theta=0.8, linLog=false) {
        super()

        this.alpha = alpha
        this.beta = beta
        this.visc = visc
        this.minV = minV
        this.temp = temp
        this.rad = rad
        this.theta = theta
        this.linLog = linLog
        this.repCutoff = Infinity  // будет установлен в respawn после масштабирования rad

        this.stateFlag = 2
        this.constT = temp / 100

        this.numNodes = null
        this.edges = null
        this.v = {}
        this.a = {}
        this.m = {}

        this.thrA = 150
        this.decayFactor = 0.98
    }

    setVisibleGraph(visibleGraph) {
        console.assert(visibleGraph instanceof Graph)
        this.edges = visibleGraph.getEdges()
        this.numNodes = visibleGraph.getNumNodes()
        super.setVisibleGraph(visibleGraph)
    }

    respawn() {
        super.respawn()
        this.v = {}
        this.a = {}
        this.m = {}
        for (let n = 0; n < this.numNodes; n++)
            this.m[n] = 1
        for (const [i, j] of this.visibleGraph.getEdges()) {
            this.m[i]++
            this.m[j]++
        }
        for (let n = 0; n < this.numNodes; n++) {
            this.v[n] = new Vec(0, 0)
            this.a[n] = new Vec(0, 0)
        }
    }

    cool() {
        this.temp *= this.decayFactor
    }

    // Репульсия: FA2 базовая формула k²/d с порогом расстояния.
    // За пределами cutoff сила равна нулю — изолированные компоненты
    // не улетают бесконечно далеко, оставаясь в пределах cutoff от остальных.
    // cutoff масштабируется через rad, чтобы не нужно было подбирать вручную.
    force_r(d) {
        if (d > this.repCutoff) return 0
        return this.alpha * this.rad ** 2 / d
    }

    // Аттракция: LinLog или кубическая (оригинал)
    force_a(d) {
        if (this.linLog) {
            return Math.log(1 + d) / this.rad
        }
        return d ** 3 / this.rad
    }

    step() {
        let nodes = this.visibleGraph.getNodes()
        let edges = this.visibleGraph.getEdges()

        // ── Управление состоянием ─────────────────────────────────────────────
        if (this.stateFlag === 2) {
            this.stateFlag = 0
            this.temp *= nodes.length / 100
            this.constT = this.temp / (10 * Math.sqrt(nodes.length))
            this.rad *= Math.sqrt(Math.sqrt(nodes.length))
            // Порог репульсии: ~10 средних длин рёбер.
            // При таком значении соседние компоненты всё ещё отталкиваются,
            // но бесконечно далёкие — нет.
            this.repCutoff = this.rad * 10
        }
        if (this.lockedNode != null) {
            this.stateFlag = 1
            this.temp = this.constT
        } else {
            if (this.stateFlag === 1) {
                this.stateFlag = 0
                this.temp *= 10 / Math.sqrt(Math.sqrt(nodes.length))
            }
        }

        // ── Сброс скоростей ───────────────────────────────────────────────────
        for (const i of nodes) {
            this.v[i].x = 0
            this.v[i].y = 0
        }

        // ── Репульсия через Barnes-Hut  O(n log n) ───────────────────────────
        const tree = buildQuadtree(nodes, this.pos, this.m)

        for (const i of nodes) {
            const acc = { x: 0, y: 0 }
            tree.applyRepulsion(i, this.pos, this.force_r.bind(this), this.theta, this.repCutoff, acc)
            this.v[i].x += acc.x
            this.v[i].y += acc.y
        }

        // ── Аттракция по рёбрам  O(edges) ────────────────────────────────────
        // Делим на m[i]/m[j] — FA2 поправка на степень узла
        for (const [i, j] of edges) {
            const dx = this.pos[i].x - this.pos[j].x
            const dy = this.pos[i].y - this.pos[j].y
            const d  = Math.sqrt(dx * dx + dy * dy)
            if (d === 0) continue

            const f = this.force_a(d) / d

            this.v[i].x -= dx * f / this.m[i]
            this.v[i].y -= dy * f / this.m[i]
            this.v[j].x += dx * f / this.m[j]
            this.v[j].y += dy * f / this.m[j]
        }

        // ── Интеграция позиций ────────────────────────────────────────────────
        this.moving = false
        for (const i of nodes) {
            if (i === this.lockedNode) continue

            const speed = this.v[i].abs()
            if (speed === 0) continue

            const clampedSpeed = Math.min(speed, this.temp)
            const scale = clampedSpeed / speed

            this.pos[i].x += this.v[i].x * scale
            this.pos[i].y += this.v[i].y * scale
        }

        if (this.temp > 0.005) {
            this.moving = true
        }
        if (!this.lockedNode) {
            this.cool()
        }
        this.iteration++
    }
}