/**
 * Create selectors for parameters according to their types, default values and constraints.
 */
class ParamsBuilder {
    static allowed_types = Array('F', 'FW', 'M', 'EI', 'ELR', 'EGR', 'O',
        'AD-pa', 'AD-pd', 'AD-ea', 'AD-ed', 'AD-ma', 'AD-md')
    static cachedParams = {} // type -> parameters
    static masks = ["train", "val", "test", "all"] // masks

    // Get parameters information of the given type
    static async getParams(type) {
        console.assert(ParamsBuilder.allowed_types.includes(type))
        if (type in ParamsBuilder.cachedParams)
            return ParamsBuilder.cachedParams[type]

        let params = await controller.ajaxRequest('/ask', {
                ask: "parameters",
                type: type,
            })
        ParamsBuilder.cachedParams[type] = params
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

        this.nameParameters = null // json of selectors parameters, dict {name -> params}
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
        // Clear dict without creating a new object
        Object.keys(this.kwArgs).forEach(key => {delete this.kwArgs[key]})
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
        if (!(key in this.selectors)) {
            console.error(`Cannot rename parameter '${key}' since it is not in this.selectors.
             Available names are: [${Object.keys(this.selectors)}]`)
            return
        }
        let $input = this.selectors[key]
        let $label = $(`label[for=${$input.attr("id")}]`)
        $label.text(name)
    }

    // Build selectors according to the given value of the key
    async build(key, postFunction, isEdgeLevel=false) {
        this.nameParameters = {}
        for (const type of this.types) {
            for (const [k, v] of Object.entries(await ParamsBuilder.getParams(type))) {
                if (!(k in this.nameParameters))
                    this.nameParameters[k] = {}
                Object.assign(this.nameParameters[k], v)
            }
        }

        let $div = this.$element
        // $div.append($("<label></label>").html("<b>Parameters</b>"))
        let parameters = this.nameParameters[key]
        if (!parameters) return

        let addInt = (name, params, $cb) => {
            let [label, type, def, possible, tip] = params

            let id = this.id(name)
            let $input = $("<input>").attr("id", id)
            $cb.append($input)
            this.selectors[name] = $input

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

            $input.attr("step", 1)
            $input.attr("pattern", "\d+")
            // $input.change(() => this.kwArgs[name] = parseInt($input.val()))

            for (const [key, value] of Object.entries(possible))
                $input.attr(key, value)

            // Check input value when user unfocus it or change it
            addValueChecker($input, "int", def, possible["min"], possible["max"], "change")
            return $input
        }

        for (const [name, params] of Object.entries(parameters)) {
            if (name === "_technical_parameter") continue
            let [label, type, def, possible, tip] = params
            let $cb = $("<div></div>").attr("class", "control-block")
            $div.append($cb)
            let id = this.id(name)
            $cb.append($("<label></label>").text(label).attr("for", id).attr("title", tip))

            let $input
            if (type === "bool") {
                $input = $("<input>").attr("id", id)
                $cb.append($input)
                this.selectors[name] = $input
                $input.attr("type", "checkbox")
                $input.prop('checked', def)
                $input.change(() => this.kwArgs[name] = $input.is(":checked"))
            }

            else if (type === "int_or_tuple" || type === "int_or_tuple_or_mask") {
                if (isEdgeLevel) { // tuple
                    let $inputsDiv = $cb
                    let $pairOrMaskDiv
                    if (type === "int_or_tuple_or_mask") {
                        $pairOrMaskDiv = $("<div></div>")
                        $cb.append($pairOrMaskDiv)
                        $inputsDiv = $("<div></div>")
                    }

                    // Add 2 inputs
                    let $input1 = addInt(name, params, $inputsDiv)
                    $input1.css("max-width", "80px")
                    $input1.change(
                        () => this.kwArgs[name] = [parseInt($input1.val()), this.kwArgs[name][1]])

                    let $input2 = addInt(name + '2', params, $inputsDiv)
                    $input2.css("max-width", "80px")
                    $input2.change(
                        () => this.kwArgs[name] = [this.kwArgs[name][0], parseInt($input2.val())])

                    // Add option to choose a mask
                    if (type === "int_or_tuple_or_mask") {
                        let $mask = $("<select></select>").attr("id", id + '-mask')

                        // Add radio for pair and 2 int inputs
                        let $cbPair = $("<div></div>").attr("class", "control-block")
                        $pairOrMaskDiv.append($cbPair)
                        let radioId = id + '-pair'
                        let $radioPair = $("<input>").attr("type", "radio")
                            .attr("name", this.idPrefix + "tuple_or_mask")
                            .attr("id", radioId).attr("value", 'pair')
                        $cbPair.append($radioPair)
                        $cbPair.append($("<label></label>").text('Pair')
                            .attr("for", radioId))
                        $radioPair.change(() => {
                            $mask.prop("disabled", true)
                            $input1.prop("disabled", false)
                            $input2.prop("disabled", false)
                            this.kwArgs[name] = [parseInt($input1.val()), parseInt($input2.val())]
                        })
                        $cbPair.append($inputsDiv)
                        // Check the first radio option
                        $radioPair.prop('checked', true)
                        $radioPair.change()

                        // Add radio for mask and mask
                        let $cbMask = $("<div></div>").attr("class", "control-block")
                        $pairOrMaskDiv.append($cbMask)
                        radioId = id + '-mask'
                        let $radioMask = $("<input>").attr("type", "radio")
                            .attr("name", this.idPrefix + "tuple_or_mask")
                            .attr("id", radioId).attr("value", "mask")
                        $cbMask.append($radioMask)
                        $cbMask.append($("<label></label>").text("Mask")
                            .attr("for", radioId))
                        $radioMask.change(() => {
                            $mask.prop("disabled", false)
                            $input1.prop("disabled", true)
                            $input2.prop("disabled", true)
                            this.kwArgs[name] = $mask.val()
                        })

                        $cbMask.append($mask)
                        // this.selectors[name] = $mask
                        $mask.attr("type", "range")
                        for (const option of ParamsBuilder.masks) {
                            $mask.append($("<option></option>").text(option)
                                .attr("selected", option === "test"))
                        }
                        $mask.change(() => this.kwArgs[name] = $mask.val())

                    }

                    // Set the default to edge 0,0 - fixme?
                    def = [0, 0]
                }
                else { // int
                    let $input1 = addInt(name, params, $cb)
                    $input1.change(() => this.kwArgs[name] = parseInt($input1.val()))
                }
            }

            else if (type === "int" || type === "float") {
                $input = $("<input>").attr("id", id)
                $cb.append($input)
                this.selectors[name] = $input
                $input.attr("type", "number")
                $input.val(Number.isFinite(def) ? def : possible["min"])

                // Handle 'possible' and remove it
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

                // Check input value when user unfocus it or change it
                addValueChecker($input, type, def, possible["min"], possible["max"], "change")
            }

            else if (type === "string") {
                $input = $("<select></select>").attr("id", id)
                $cb.append($input)
                this.selectors[name] = $input
                $input.attr("type", "range")
                for (const option of possible) {
                    $input.append($("<option></option>").text(option)
                        .attr("selected", option === def))
                }
                $input.change(() => this.kwArgs[name] = $input.val())
            }

            else if (type === "dynamic") {
                $input = $("<select></select>").attr("id", id)
                $cb.append($input)
                this.selectors[name] = $input
                $input.attr("type", "range")
                let params_type = possible["params_type"]

                // Insert inner ParamsBuilder
                let $innerParamsDiv = $("<div></div>")
                $div.append($("<div></div>").attr("class", "menu-separator"))
                $div.append($innerParamsDiv)
                let paramsBuilder = new ParamsBuilder($innerParamsDiv,
                    params_type, this.idPrefix + "-inner-" + params_type + '-')

                // Result of inner ParamsBuilder is assigned to a specified field
                let resultName = possible["result_to"]
                this.kwArgs[name] = null // will be induced at Backend
                this.kwArgs[resultName] = {
                    _class_name: null,  // to be set on $input.change()
                    _config_kwargs: paramsBuilder.kwArgs,
                    params_type: params_type, // helpful for Backend to put it back
                }

                let self = this
                $input.change(async function () {
                    paramsBuilder.drop()
                    // self.$acceptDiv.hide() // todo need $acceptDiv of parent
                    await paramsBuilder.build(this.value)
                    // self.$acceptDiv.show()
                    self.kwArgs[resultName]["_class_name"] = this.value
                })

                // Add possible values
                let possibleValues = []
                if ("possible" in possible)
                    possibleValues = possible["possible"]
                else // All possible values
                    possibleValues = Object.keys(await ParamsBuilder.getParams(params_type))

                for (const option of possibleValues) {
                    $input.append($("<option></option>").text(option)
                        .attr("selected", option === def))
                }
                $input.change()
            }
            else
                console.error("Type not supported", type)

            if (type !== "dynamic")
                this.kwArgs[name] = def
        }

        if (postFunction) postFunction(key)

    }
}