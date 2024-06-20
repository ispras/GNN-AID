Array.prototype.sum = function () {
    return this.reduce((a, b) => a + b, 0)
}

// // HSV values in [0..1] returns [r, g, b] values from 0 to 255
// function hsv_to_rgb(h, s, v) {
//     let f, p, q, r, g, b
//     let h_i = Math.floor(h * 6)
//     f = h * 6 - h_i
//     p = v * (1 - s)
//     q = v * (1 - f * s)
//     let t = v * (1 - (1 - f) * s)
//     if (h_i === 0)
//         [r, g, b] = [v, t, p]
//     if (h_i === 1)
//         [r, g, b] = [q, v, p]
//     if (h_i === 2)
//         [r, g, b] = [p, v, t]
//     if (h_i === 3)
//         [r, g, b] = [p, q, v]
//     if (h_i === 4)
//         [r, g, b] = [t, p, v]
//     if (h_i === 5)
//         [r, g, b] = [v, p, q]
//     return [Math.floor(r * 256), Math.floor(g * 256), Math.floor(b * 256)]
// }

// Random HSL color
function randomColor(s=0.5, l=0.95) {
    const golden_ratio_conjugate = 0.618033988749895
    if (!this.h)
        this.h = 0 // Math.random()
    this.h += golden_ratio_conjugate
    this.h %= 1
    return `hsl(${Math.floor(this.h*360)}, ${Math.floor(s*100)}%, ${Math.floor(l*100)}%)`
}

/**
 * Translate numeric value to color of the colormap
 * @param value
 * @param colormap
 * @param min - minimal possible value
 * @param max - maximal possible value
 * @param soft - if true, use exponential smoothing, i.e. min and max value used as
 * @param p - pad for soft=true, i.e. f(min)=p, f(max)=1-p
 *
 * @return RGB color string
 */
function valueToColor(value, colormap, min=0, max=1, soft=false, p=0.2) {
    value = parseFloat(value) // Just for safety
    if (soft) {
        // Use sigmoid function s.t. f(min)=p, f(max)=1-p, f(+inf)=1, f(-inf)=0
        let a = 2*Math.log(p/(1-p))/(max-min)
        let b = -Math.log(p/(1-p)) - a*min
        value = 1/(1+Math.exp(a*value+b))
    }
    else {
        value = (value - min) / (max-min)
    }
    let [r, g, b] = evaluate_cmap(value, colormap, false)
    return `rgb(${r}, ${g}, ${b})`
}

// 2-D vector supporting + - *
class Vec {
    constructor(x, y) {
        this.x = x
        this.y = y
    }
    str(precision) {
        return `${this.x.toPrecision(precision)}, ${this.y.toPrecision(precision)}`
    }
    set(vec) {
        this.x = vec.x
        this.y = vec.y
    }
    add(a) {
        if ((typeof a) == "number") {
            this.x += a
            this.y += a
        }
        else { // Vec
            this.x += a.x
            this.y += a.y
        }
        return this
    }
    sub(a) {
        if ((typeof a) == "number") {
            this.x -= a
            this.y -= a
        }
        else { // Vec
            this.x -= a.x
            this.y -= a.y
        }
        return this
    }
    mul(a) {
        this.x *= a
        this.y *= a
        return this
    }
    dist(vec) {
        return ((this.x-vec.x)**2 + (this.y-vec.y)**2)**0.5
    }
    abs() {
        return ((this.x)**2 + (this.y)**2)**0.5
    }
    static add(vec, a) {
        if ((typeof a) == "number")
            return new Vec(vec.x+a, vec.y+a)
        else // Vec
            return new Vec(vec.x+a.x, vec.y+a.y)
    }
    static sub(vec, a) {
        if ((typeof a) == "number")
            return new Vec(vec.x - a, vec.y - a)
        else // Vec
            return new Vec(vec.x - a.x, vec.y - a.y)
    }
    static mul(vec, a) {
        return new Vec(vec.x * a, vec.y * a)
    }
    static rotate(vec, angle) {
        let cos = Math.cos(angle)
        let sin = Math.sin(angle)
        return new Vec(vec.x * cos - vec.y * sin, vec.x * sin + vec.y * cos)
    }
}

class BiList {
    constructor() {
        this.Node = class {
            constructor(value) {
                this.value = value
                this.next = null
                this.previous = null
            }
        }

        this._first = null // First Node
        this._last = null // Last Node
        this.length = 0 // Number of elements
    }

    // Get the first element
    first() {
        return this._first.value
    }

    // Get the last element
    last() {
        return this._last.value
    }

    // Add element at the end
    append(value) {
        let newNode = new this.Node(value)

        if (this.length === 0)
            this._first = this._last = newNode

        else {
            newNode.previous = this._last
            this._last.next = newNode
            this._last = newNode
            if (this.length === 1)
                this._first = newNode.previous
        }
        this.length++
        return this
    }

    // Add element at the beginning
    prepend(value) {
        let newNode = new this.Node(value)

        if (this.length === 0)
            this._first = this._last = newNode

        else {
            newNode.next = this._first
            this._first.previous = newNode
            this._first = newNode
            if (this.length === 1)
                this._last = newNode.next
        }
        this.length++
        return this
    }

    // Insert element at index
    insert(value, index) {
        if (index == null)
            index = this.length

        if (!Number.isInteger(index) || index < 0 || index > this.length)
            console.log(`Invalid index. Current length is ${this.length}.`)

        if (index === 0)
            return this.prepend(value)

        if (index >= this.length)
            return this.append(value)

        // Reach the node at that index
        let newNode = new this.Node(value)
        let previousNode = this._first;
        for (let k = 0; k < index - 1; k++)
            previousNode = previousNode.next

        let nextNode = previousNode.next;
        newNode.next = nextNode;
        previousNode.next = newNode;
        newNode.previous = previousNode;
        nextNode.previous = newNode;

        this.length++
        return this
    }

    // Remove element at index
    remove (index) {
        if (!Number.isInteger(index) || index < 0 || index >= this.length)
            console.log(`Invalid index. Current length is ${this.length}.`)

        // Remove first
        if (index === 0) {
            this._first = this._first.next
            this._first.previous = null
        }

        // Remove last
        else if (index === this.length - 1) {
            this._last = this._last.previous
            this._last.next = null
        }

        // Remove node at an index
        else {
            let previousNode = this._first
            for (let k = 0; k < index - 1; k++)
                previousNode = previousNode.next
            let deleteNode = previousNode.next
            let nextNode = deleteNode.next
            previousNode.next = nextNode
            nextNode.previous = previousNode
        }
        this.length--
    }

    // Move element at the index to index-1
    toFirst(index) {
        if (!Number.isInteger(index) || index < 0 || index >= this.length)
            console.log(`Invalid index. Current length is ${this.length}.`)

        if (index <= 0) return false

        let prev = this._first
        for (let k = 0; k < index-1; k++)
            prev = prev.next

        let node = prev.next
        let prev2 = prev.previous
        let next = node.next

        if (prev2) prev2.next = node
        node.previous = prev2
        node.next = prev
        prev.previous = node
        prev.next = next
        if (next) next.previous = prev

        if (index === 1) this._first = node
        if (index === this.length-1) this._last = prev

        return true
    }

    // Move element at the index to index+1
    toLast(index) {
        if (!Number.isInteger(index) || index < 0 || index >= this.length)
            console.log(`Invalid index. Current length is ${this.length}.`)

        if (index >= this.length-1) return false

        let node = this._first
        for (let k = 0; k < index; k++)
            node = node.next

        let prev = node.previous
        let next = node.next
        let next2 = next.next

        if (prev) prev.next = next
        next.previous = prev
        next.next = node
        node.previous = next
        node.next = next2
        if (next2) next2.previous = node

        if (index === 0) this._first = next
        if (index === this.length-2) this._last = node

        return true
    }

    // Iterate over elements from start to end index
    iterator(start=0, end=Infinity) {
        let nextIndex = start
        let length = this.length

        let node = this._first
        for (let k = 0; k < start && k < this.length; k++)
            node = node.next

        return {
            next() {
                let result
                if (nextIndex < end && nextIndex < length) {
                    result = {value: node.value, done: false}
                    nextIndex++
                    node = node.next
                    return result
                }
                return {value: null, done: true}
            }
        }
    }

    // Iterate over elements from start to end index in reverse order
    reverseIterator(start=Infinity, end=0) {
        let length = this.length
        let prevIndex = Math.min(start, length-1)

        let node = this._first
        for (let k = 0; k < prevIndex && k < this.length; k++)
            node = node.next

        return {
            next() {
                let result
                if (prevIndex >= end && prevIndex >= 0) {
                    result = {value: node.value, done: false}
                    prevIndex--
                    node = node.previous
                    return result
                }
                return {value: null, done: true}
            }
        }
    }

    // Get element by index
    get(index) {
        if (!Number.isInteger(index) || index < 0 || index >= this.length)
            console.log(`Invalid index. Current length is ${this.length}.`)

        let node = this._first
        for (let k = 0; k < index; k++)
            node = node.next

        return node.value
    }
}

class Svg {
    static text(text, x, y,
                dominantBaseline='middle',
                fontSize='18px',
                fontWeight='normal',
                fill=null,) {
        let item = document.createElementNS("http://www.w3.org/2000/svg", "text")
        item.textContent = text
        item.setAttribute('x', x)
        item.setAttribute('y', y)
        item.setAttribute('dominant-baseline', dominantBaseline)
        item.setAttribute('font-weight', fontWeight)
        if (fill) item.setAttribute('fill', fill)
        item.setAttribute('font-size', fontSize)
        return item
    }

    static circle(x, y, r, fill=null, stroke=null, show=true) {
        let item = document.createElementNS("http://www.w3.org/2000/svg", "circle")
        item.setAttribute('cx', x)
        item.setAttribute('cy', y)
        item.setAttribute('r', r)
        if (fill) item.setAttribute('fill', fill)
        if (stroke) item.setAttribute('stroke', stroke)
        item.setAttribute('display', show ? "inline" : "none")
        return item
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function JSON_parse(string) {
    return JSON.parse(string, function (key, value) {
        if (value === 'NaN')
            return NaN
        if (value === 'Infinity')
            return Infinity
        if (value === '-Infinity')
            return -Infinity
        return value
    })
}

function JSON_stringify(object, space) {
    return JSON.stringify(object, function (key, value) {
        if (value === Infinity)
            return 'Infinity'
        if (value === -Infinity)
            return '-Infinity'
        return value
    }, space)
}

// Create a valid HTML id from a given name
function nameToId(name) {
    let string = "" + name
    // TODO what about other symbols?
    for (const x of " ,;:'\"()[]{}/") {
        string = string.replaceAll(x, '-')
    }
    return string
}

// Create unique HTML id based on current time moment
function timeBasedId() {
    return ('' + performance.timeOrigin + performance.now())
        .replaceAll('.', '-')
}

// Version of SVG.getBBox() which works even if SVG is not rendered
// Adapted from https://stackoverflow.com/a/45465286/8900030
SVGGraphicsElement.prototype.getBBoxWithCopying = function() {
    let bbox, tempDiv, tempSvg
    tempDiv = document.createElement("div")
    tempDiv.setAttribute("style", "position:absolute; visibility:hidden; width:0; height:0")
    if (this.tagName === "svg") {
        tempSvg = this.cloneNode(true)
    } else {
        tempSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg")
        tempSvg.appendChild(this.cloneNode(true))
    }
    tempDiv.appendChild(tempSvg)
    document.body.appendChild(tempDiv)
    bbox = tempSvg.getBBox()
    document.body.removeChild(tempDiv)
    return bbox
}

// Convolution, Activation, Batchnorm
async function addOptionsWithParams(id, label, options, paramsType, paramsColor) {
    let $cb = $("<div></div>").attr("class", "control-block")
    $cb.append($("<label></label>").text(label).attr("for", id))
    let $optionSelect = $("<select></select>").attr("id", id)
    $cb.append($optionSelect)
    if (options == null) {
        options = []
        for (const key of Object.keys(await ParamsBuilder.getParams(paramsType)))
            options.push([key, key])
    }
    for (const [val, text] of options) {
        $optionSelect.append($("<option></option>").val(val).text(text))
    }

    let $paramsDiv = $("<div></div>")
        .css("margin-left", LayerBlock.leftMargin + "px")
    let paramsBuilder = new ParamsBuilder($paramsDiv, paramsType, id + "-")

    $optionSelect.change(async function () {
        paramsBuilder.drop()
        $paramsDiv.css("background-color", "transparent")
        if (this.value !== "None") {
            if (paramsColor)
                $paramsDiv.css("background-color", paramsColor)
            await paramsBuilder.build(this.value)
        }
    })
    await $optionSelect.change()
    return [$cb, $optionSelect, $paramsDiv, paramsBuilder]
}

