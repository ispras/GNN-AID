class ProgressBar {
    constructor(idPrefix=null, show=false) {
        this.idPrefix = idPrefix ? idPrefix : timeBasedId() + '-'
        this.show = show

        this.$div = $("<div></div>").addClass("control-block").addClass("progress-bar")
            .attr("id", this.idPrefix + 'progressBar')
        this.$innerDiv = $("<div></div>")
        this.$div.append(this.$innerDiv)
        this.$label = $("<label></label>").text("Not running")
        this.$div.append(this.$label)

        this.visible(this.show)
    }

    // Set progress level
    setLoad(load=0) {
        if (!(load >= 0 && load <= 1))
            console.error("Progress bar load must be within [0;1], not ", load)
        this.$innerDiv.css("width", 100 * load + '%')
    }

    // Set text
    setText(text="") {
        this.$label.text(text)
    }

    // Set progress to zero
    start() {
        this.setLoad(0)
        this.setText("Starting...")
    }

    // Set progress to zero
    drop() {
        this.setLoad(0)
        this.setText("Not running")
    }

    // Show on/off
    visible(show) {
        this.show = show
        show ? this.$div.show() : this.$div.hide()
    }
}
