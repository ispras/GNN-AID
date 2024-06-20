/**
 * Create selectors for parameters according to their types, default values and constraints.
 */
class ParamsBuilder {
    static allowed_types = Array('F', 'FW', 'M', 'EI', 'ELR', 'EGR', 'O')
    static cachedParams = {} // type -> parameters

    // Get parameters information of the given type
    static async getParams(type) {
        console.assert(ParamsBuilder.allowed_types.includes(type))
        if (type in ParamsBuilder.cachedParams)
            return ParamsBuilder.cachedParams[type]

        let params = null
        await $.ajax({
            type: 'POST',
            url: '/ask',
            data: {
                ask: "parameters",
                type: type,
            },
            success: (parameters) => {
                params = ParamsBuilder.cachedParams[type] = JSON_parse(parameters)
            }
        })
        return params
    }

    // Manually add parameters with custom type
    static addParams(type, parameters) {
        if (ParamsBuilder.allowed_types.includes(type) && type in ParamsBuilder.cachedParams) {
            // Update
            Object.assign(ParamsBuilder.cachedParams[type], parameters)
        }
        else {
            // Add new
            ParamsBuilder.allowed_types.push(type)
            ParamsBuilder.cachedParams[type] = parameters
        }
    }

    constructor($element, type, idPrefix) {
        this.$element = $element
        this.types = [].concat(type)
        this.idPrefix = idPrefix ? idPrefix : timeBasedId() + '-'

        this.parameters = null // json of selectors parameters
        this.selectors = {} // dict of {argument -> selector jquery object}
        this.kwArgs = {} // dict of {argument -> value}
    }

    // HTML element id generated from its name
    id(name) {
        return nameToId(this.idPrefix + name)
    }

    // Remove all selectors
    drop() {
        this.$element.empty()
        this.kwArgs = {}
    }

    // Manually set the value of a selector
    setValue(key, value) {
        if (!(key in this.selectors)) return
        let $selector = this.selectors[key]
        switch ($selector[0].type) {
            case "checkbox":
                $selector.prop('checked', value).change()
                break
            case "number":
                $selector.val(value).change()
                break
            default:
                console.error('Not implemented for selector of type', $selector[0].type)
        }
    }

    // Manually enable/disable selector
    setEnabled(key, value) {
        this.selectors[key].attr("disabled", !value)
    }

    // Manually remove selector
    removeParam(key) {
        if (key in this.selectors) {
            this.selectors[key].parent().remove()
            delete this.kwArgs[key]
        }
    }

    // Manually rename selector
    renameParam(key, name) {
        console.assert(key in this.selectors)
        let $input = this.selectors[key]
        let $label = $(`label[for=${$input.attr("id")}]`)
        $label.text(name)
    }

    // Build selectors according to the case of key=value
    async build(key, postFunction) {
        this.parameters = {}
        for (const type of this.types) {
            for (const [k, v] of Object.entries(await ParamsBuilder.getParams(type))) {
                if (!(k in this.parameters))
                    this.parameters[k] = {}
                Object.assign(this.parameters[k], v)
            }
        }

        let $div = this.$element
        // $div.append($("<label></label>").html("<b>Parameters</b>"))
        let parameters = this.parameters[key]
        if (!parameters) return

        for (const [name, params] of Object.entries(parameters)) {
            if (name === "_technical_parameter") continue
            let [label, type, def, possible, tip] = params
            let $cb = $("<div></div>").attr("class", "control-block")
            $div.append($cb)
            let id = this.id(name)
            $cb.append($("<label></label>").text(label).attr("for", id).attr("title", tip))

            let $input = $("<input>")
            $input.attr("id", id)
            this.selectors[name] = $input
            if (type === "bool") {
                $input.attr("type", "checkbox")
                $input.prop('checked', def)
                $input.change(() => this.kwArgs[name] = $input.is(":checked"))
            }

            else if (type === "int" || type === "float") {
                $input.attr("type", "number")
                $input.val(Number.isFinite(def) ? def : possible["min"])
                let checkClass = id + "-radio"
                if ("special" in possible) {
                    // Add special values as separate checkboxes
                    let checkId = id + "-check"
                    for (const variant of possible.special) {
                        let $checkBox = $("<input>").attr("id", checkId)
                            .attr("type", "checkbox").prop('checked', variant === def)
                        $checkBox.addClass(checkClass)
                        $cb.append($checkBox)
                        let $label = $("<label></label>").text(variant == null ? "None" : variant)
                            .attr("for", checkId)
                        $cb.append($label)
                        $checkBox.change((e) => { // Uncheck all but this
                            let wasChecked = $checkBox.is(":checked")
                            $("." + checkClass).prop("checked", false)
                            $checkBox.prop("checked", true)
                            this.kwArgs[name] = variant
                        })
                    }
                    $input.focus(() => {
                        $("." + checkClass).prop("checked", false)
                        $input.trigger("change")
                    })
                    $input.css("min-width", "60px")
                    delete possible.special
                }
                if (type === "int") {
                    $input.attr("step", 1)
                    $input.attr("pattern", "\d+")
                    $input.change(() => this.kwArgs[name] = parseInt($input.val()))
                }
                else {
                    $input.attr("pattern", "\d+.?\d*")
                    $input.change(() => {
                        this.kwArgs[name] = parseFloat($input.val())
                        $("." + checkClass).prop("checked", false)
                    })
                }
                for (const [key, value] of Object.entries(possible))
                    $input.attr(key, value)
            }

            else if (type === "string") {
                $input = $("<select></select>")
                $input.attr("type", "range")
                for (const option of possible) {
                    $input.append($("<option></option>").text(option)
                        .attr("selected", option === def))
                }
                $input.change(() => this.kwArgs[name] = $input.val())
            }
            else
                console.error("Type not supported", type)

            // TODO use tip
            // $input.attr("id", id)
            this.kwArgs[name] = def
            // $input.change(() => this.kwArgs[name] = $input.val()) // TODO parse int ot bool?
            $cb.append($input)
        }

        if (postFunction) postFunction(key)

    }
}