// An SVG element with own tip and ids
class SvgPanel {
    constructor(parentElement, width, height) {
        this.parentElement = parentElement
        this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        this.$svg = $(this.svg)
            .css("background-color", "#e7e7e7")
            .css("flex-shrink", "0")
        if (width)
            this.$svg.css("width", width + "px")
        if (height)
            this.$svg.css("height", height + "px")

        this.parentElement.appendChild(this.svg)
        this.idPrefix = "svg-" + timeBasedId() + "-"

        this.$tip = $("<span></span>").attr("id", this.idPrefix + "tooltip")
            .addClass("tooltip-text").css("position", "fixed")
        this.parentElement.appendChild(this.$tip[0])
    }

    // Add <g> element with a specified id postfix
    add(id) {
        let g = document.createElementNS("http://www.w3.org/2000/svg","g")
        g.setAttribute("id", this.idPrefix + id)
        this.svg.appendChild(g)
        return g
    }

    // Get <g> element with a specified id postfix
    get(id) {
        return $("#" + this.idPrefix + id)
    }
}