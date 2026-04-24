LINK_TO_DOCS = "https://gnn-aid.readthedocs.io/en/latest/user_guide/datasets.html#create-new-dataset"
FORMATS = ["ij", "GML"]

class DatasetUploadDialog {
    constructor(onSuccess) {
        this._files = []
        this._metaConfig = null
        this._errorCount = 0
        this._fileChanged = false
        this.onSuccess = onSuccess

        this.uploadId = crypto.randomUUID() // ID of the current upload dialog
        this._buildDom()
        this._bindEvents()
    }

    // ── Public ────────────────────────────────────────────────────────────────

    show() {
        this.$backdrop.show()
    }

    async close() {
        let data = {
            set: "cancel",
            uploadId: this.uploadId
        }
        await controller.ajaxRequest("/dataset", data)
        this.$backdrop.remove()
    }

    // ── DOM construction ──────────────────────────────────────────────────────

    _buildDom() {
        this.$backdrop = $("<div></div>").css({
            position: "fixed", top: 0, left: 0, width: "100%", height: "100%",
            background: "rgba(0,0,0,0.5)", zIndex: 1000,
            display: "flex", justifyContent: "center", alignItems: "center"
        })

        this.$dialog = $("<div></div>").css({
            background: "#fff", borderRadius: "8px", padding: "20px",
            width: "750px", maxWidth: "95vw", height: "75vh", maxHeight: "90vh",
            overflow: "hidden", boxShadow: "0 4px 20px rgba(0,0,0,0.3)",
            boxSizing: "border-box", display: "flex", flexDirection: "column", gap: "12px"
        })
        this.$backdrop.append(this.$dialog)

        let $header = $("<div></div>").css({
            display: "flex", justifyContent: "space-between", alignItems: "center",
            flex: "0 0 auto", gap: "12px"
        })

        let $headerLeft = $("<div></div>").css({
            display: "flex", alignItems: "center", gap: "16px"
        })

        $headerLeft.append($("<h3></h3>").text("Upload dataset").css({ margin: 0 }))

        $headerLeft.append(
            $("<a></a>")
                .text("Help about dataset format")
                .attr("href", LINK_TO_DOCS)
                .attr("target", "_blank")
                .css({ fontSize: "14px" })
        )

        let $closeBtn = $("<button></button>").text("×")
            .css({
                border: "none", background: "none", fontSize: "34px",
                cursor: "pointer", lineHeight: 1, padding: "0 8px",
                minWidth: "42px", minHeight: "42px"
            })

        $closeBtn.click(async () => await this.close())

        $header.append($headerLeft).append($closeBtn)
        this.$dialog.append($header)

        let $content = $("<div></div>").css({
            display: "grid", gridTemplateColumns: "minmax(320px, 0.9fr) minmax(380px, 1.1fr)",
            gap: "16px", flex: "1 1 auto", minHeight: 0
        })

        let $left = $("<div></div>").css({
            display: "flex", flexDirection: "column", gap: "10px", minHeight: 0
        })

        let $right = $("<div></div>").css({
            display: "flex", flexDirection: "column", gap: "10px", minHeight: 0
        })

        // Left: drop-zone
        this.$dropZone = $("<div></div>").css({
            border: "2px dashed #aaa", borderRadius: "4px",
            padding: "14px", textAlign: "center", cursor: "pointer",
            minHeight: "100px", display: "flex", flexDirection: "column",
            justifyContent: "center", alignItems: "center", gap: "8px",
            boxSizing: "border-box", flex: "0 0 auto"
        })
        this.$dropZone.append($("<span></span>").text("Drop files here or click to browse").css({ color: "#666" }))
        this.$fileInput = $("<input></input>")
            .attr("type", "file").attr("multiple", true)
            .css({ display: "none" })
        this.$dropZone.append(this.$fileInput)
        $left.append(this.$dropZone)

        // Left: file tree (small fixed height, only 1 file supported)
        $left.append($("<div></div>").html("<b>Files:</b>").css({ flex: "0 0 auto" }))
        this.$fileTree = $("<div></div>").css({
            border: "1px solid #ccc", borderRadius: "4px",
            padding: "8px", flex: "0 0 52px", overflowY: "auto",
            fontFamily: "monospace", fontSize: "12px", whiteSpace: "pre",
            color: "#444", boxSizing: "border-box"
        }).text("No files selected")
        $left.append(this.$fileTree)

        // Left: Check button centered
        this.$checkBtn = $("<button></button>").text("Check").prop("disabled", true).css({
            alignSelf: "center", padding: "6px 20px", flex: "0 0 auto"
        })
        $left.append(this.$checkBtn)

        // Left: errors area
        let $errorsSection = $("<div></div>").css({
            display: "flex", flexDirection: "column", gap: "4px", flex: "1 1 auto", minHeight: 0
        })
        $errorsSection.append($("<div></div>").html("<b>Errors:</b>"))
        this.$errorsArea = $("<textarea></textarea>")
            .css({
                width: "100%", height: "100%", boxSizing: "border-box",
                fontFamily: "monospace", fontSize: "12px", resize: "none",
                overflowY: "auto", flex: "1 1 auto"
            })
            .prop("readonly", true)
        $errorsSection.append(this.$errorsArea)
        $left.append($errorsSection)

        // Right main: meta-info form
        this.$metaSection = $("<div></div>").css({
            display: "flex", flexDirection: "column", gap: "6px", flex: "1 1 auto", minHeight: 0
        })
        this.$metaSection.append($("<div></div>").html("<b>Dataset metainfo:</b>").css({ flex: "0 0 auto" }))
        this.$metaForm = $("<div></div>").css({
            border: "1px solid #ccc", borderRadius: "4px", padding: "10px",
            flex: "1 1 auto", minHeight: "360px", overflowY: "auto", boxSizing: "border-box"
        })
        this.$metaForm.append(
            $("<div></div>")
                .text("Press Check after selecting files to generate the initial metainfo config")
                .css({ color: "#777", fontSize: "13px" })
        )
        this.$metaSection.append(this.$metaForm)
        $right.append(this.$metaSection)

        $content.append($left).append($right)
        this.$dialog.append($content)

        // Bottom: Submit button only
        let $footer = $("<div></div>").css({
            display: "flex", justifyContent: "flex-end", flex: "0 0 auto"
        })
        this.$submitBtn = $("<button></button>").text("Submit").prop("disabled", true)
        $footer.append(this.$submitBtn)
        this.$dialog.append($footer)

        $("body").append(this.$backdrop)
    }

    _bindEvents() {
        this.$dropZone.on("click", (e) => {
            if (e.target === this.$fileInput[0])
                return

            this.$fileInput[0].click()
        })

        this.$fileInput.on("click", (e) => {
            e.stopPropagation()
        })

        this.$fileInput.on("change", (e) => {
            this._files = Array.from(e.target.files)
            this._resetAfterFilesChanged()
        })

        this.$dropZone.on("dragover", (e) => {
            e.preventDefault()
            this.$dropZone.css("borderColor", "#4a90d9")
        }).on("dragleave", () => {
            this.$dropZone.css("borderColor", "#aaa")
        }).on("drop", (e) => {
            e.preventDefault()
            this.$dropZone.css("borderColor", "#aaa")
            this._files = Array.from(e.originalEvent.dataTransfer.files)
            this._resetAfterFilesChanged()
        })

        this.$checkBtn.click(() => this._onCheck())
        this.$submitBtn.click(() => this._onSubmit())

        this.$metaForm.on("input change", "[data-field], [data-attr-key]", () => {
            this.$submitBtn.prop("disabled", true)
        })
    }

    _resetAfterFilesChanged() {
        this._metaConfig = null
        this._errorCount = 0
        this._fileChanged = true
        this._updateFileTree()
        this.$checkBtn.prop("disabled", this._files.length === 0)
        this.$submitBtn.prop("disabled", true)
        this.$errorsArea.val("")
        this.$metaForm.empty()
        this.$metaForm.append(
            $("<div></div>")
                .text("Press Check after selecting files to generate the initial metainfo config")
                .css({ color: "#777", fontSize: "13px" })
        )
    }

    // ── File tree display ─────────────────────────────────────────────────────

    _updateFileTree() {
        if (this._files.length === 0) {
            this.$fileTree.text("No files selected")
            return
        }

        let root = {}
        for (const file of this._files) {
            let path = file.webkitRelativePath || file.name
            let parts = path.split("/")
            let node = root
            for (let i = 0; i < parts.length - 1; i++) {
                if (!node[parts[i]]) node[parts[i]] = {}
                node = node[parts[i]]
            }
            node[parts[parts.length - 1]] = null
        }

        let lines = []
        const render = (node, indent) => {
            for (const [key, val] of Object.entries(node).sort()) {
                if (val === null)
                    lines.push(indent + "  " + key)
                else {
                    lines.push(indent + "+ " + key)
                    render(val, indent + "  ")
                }
            }
        }
        render(root, "")
        this.$fileTree.text(lines.join("\n"))
    }

    // ── Backend communication ─────────────────────────────────────────────────

    async _onCheck() {
        this.$checkBtn.prop("disabled", true).text("Checking…")
        this.$submitBtn.prop("disabled", true)

        try {
            const formData = new FormData()
            formData.append("set", "check")
            formData.append("uploadId", this.uploadId)

            if (this._fileChanged)
                for (const file of this._files)
                    formData.append("files", file, file.webkitRelativePath || file.name)

            if (this._metaConfig) {
                const [metainfo, validationErrors] = this._collectMetaConfig()
                if (validationErrors.length > 0) {
                    this._showErrors(validationErrors.join("\n"))
                    return
                }
                console.log('metainfo', metainfo)
                formData.append("metainfo", JSON.stringify(metainfo))
            }

            const result = await controller.ajaxRequest("/dataset", formData)
            console.log('result', result)
            let [errors, config] = result

            if (config)
                this._metaConfig = config
            this._fileChanged = false
            this._showErrors(errors)
            if (config)
                this._buildMetaForm(config)

            const hasErrors = errors && errors.length > 0 && errors !== "SUCCESS"
            this.$submitBtn.prop("disabled", hasErrors)
        } finally {
            this.$checkBtn.prop("disabled", this._files.length === 0).text("Check")
        }
    }

    async _onSubmit() {
        this.$submitBtn.prop("disabled", true).text("Submitting…")

        try {
            let data = {
                set: "submit",
                uploadId: this.uploadId
            }
            const result = await controller.ajaxRequest("/dataset", data)
            console.log('result', result)

            this.$backdrop.remove()
            this.onSuccess()
        } finally {
            this.$submitBtn.prop("disabled", false).text("Submit")
        }
    }

    // ── Meta-info form ────────────────────────────────────────────────────────

    _appendTextField($form, key, value) {
        $form.append(this._fieldLabel(key))

        const $input = $("<input></input>")
            .attr("type", "text")
            .attr("data-field", key)
            .val(value)
            .css(this._inputCss())

        $form.append($input)
    }

    _appendFormatField($form, value) {
        $form.append(this._fieldLabel("format"))

        const $select = $("<select></select>")
            .attr("data-field", "format")
            .css(this._inputCss())

        for (const fmt of FORMATS) {
            $select.append($("<option></option>").attr("value", fmt).text(fmt))
        }

        $select.val(value)

        $form.append($select)
    }

    _appendNumberField($form, key, value) {
        $form.append(this._fieldLabel(key))

        const $input = $("<input></input>")
            .attr("type", "number")
            .attr("data-field", key)
            .val(value)
            .css(this._inputCss())

        $form.append($input)
    }

    _appendBoolField($form, key, value) {
        $form.append(this._fieldLabel(key))

        const $select = $("<select></select>")
            .attr("data-field", key)
            .css(this._inputCss())

        $select.append($("<option></option>").attr("value", "false").text("false"))
        $select.append($("<option></option>").attr("value", "true").text("true"))
        $select.val(value ? "true" : "false")

        $form.append($select)
    }

    _appendArrayField($form, key, value) {
        $form.append(this._fieldLabel(key))

        const $input = $("<input></input>")
            .attr("type", "text")
            .attr("data-field", key)
            .attr("data-kind", "json")
            .val(JSON.stringify(value ?? []))
            .css(this._inputCss())

        $form.append($input)
    }

    _fieldLabel(key) {
        return $("<label></label>")
            .text(key + ":")
            .css({
                fontSize: "13px",
                fontWeight: "bold"
            })
    }

    _inputCss() {
        return {
            width: "100%",
            boxSizing: "border-box",
            fontSize: "13px",
            border: "1px solid #ccc",
            padding: "4px 6px",
            borderRadius: "3px",
            minWidth: 0
        }
    }

    _sectionTitle(title) {
        return $("<div></div>")
            .text(title)
            .css({
                marginTop: "16px",
                marginBottom: "6px",
                fontWeight: "bold",
                fontSize: "14px"
            })
    }

    _buildAttributesTable(fieldName, attributes) {
        const $wrapper = $("<div></div>")
            .attr("data-attr-field", fieldName)
            .css({
                border: "1px solid #ddd",
                borderRadius: "4px",
                padding: "8px",
                overflowX: "auto"
            })

        if (!attributes) {
            attributes = {
                names: [],
                types: [],
                values: []
            }
        }

        if (this._isSimpleAttributesInfo(attributes)) {
            const $table = this._buildSingleAttributesTable(fieldName, "", attributes)
            $wrapper.append($table)
        } else {
            for (const [groupName, groupAttributes] of Object.entries(attributes)) {
                $wrapper.append(
                    $("<div></div>")
                        .text(groupName)
                        .css({
                            fontWeight: "bold",
                            margin: "6px 0"
                        })
                )

                const $table = this._buildSingleAttributesTable(fieldName, groupName, groupAttributes)
                $wrapper.append($table)
            }
        }

        return $wrapper
    }

    _isSimpleAttributesInfo(attributes) {
        return (
            attributes &&
            typeof attributes === "object" &&
            Array.isArray(attributes.names) &&
            Array.isArray(attributes.types) &&
            Array.isArray(attributes.values)
        )
    }

    _buildSingleAttributesTable(fieldName, groupName, attributes) {
        const names = attributes?.names ?? []
        const types = attributes?.types ?? []
        const values = attributes?.values ?? []

        const rowCount = Math.max(names.length, types.length, values.length)

        const $table = $("<table></table>")
            .attr("data-attr-table", fieldName)
            .attr("data-attr-group", groupName)
            .css({
                width: "100%",
                borderCollapse: "collapse",
                marginBottom: "8px",
                fontSize: "13px"
            })

        const $thead = $("<thead></thead>")
        const $headRow = $("<tr></tr>")

        for (const title of ["name", "type", "values"]) {
            $headRow.append(
                $("<th></th>")
                    .text(title)
                    .css({
                        textAlign: "left",
                        borderBottom: "1px solid #ccc",
                        padding: "4px"
                    })
            )
        }

        $thead.append($headRow)
        $table.append($thead)

        const $tbody = $("<tbody></tbody>")

        for (let i = 0; i < rowCount; i++) {
            const $row = $("<tr></tr>").attr("data-attr-row", "1")

            const $nameInput = $("<input></input>")
                .attr("type", "text")
                .attr("data-attr-key", "name")
                .val(names[i] ?? "")
                .css(this._tableInputCss())

            const $typeSelect = this._buildAttributeTypeSelect(types[i] ?? "other")

            const $valuesInput = $("<input></input>")
                .attr("type", "text")
                .attr("data-attr-key", "values")
                .val(JSON.stringify(values[i] ?? []))
                .css(this._tableInputCss())

            $row.append(this._tableCell($nameInput))
            $row.append(this._tableCell($typeSelect))
            $row.append(this._tableCell($valuesInput))

            $tbody.append($row)
        }

        $table.append($tbody)

        return $table
    }

    _buildAttributeTypeSelect(value) {
        const $select = $("<select></select>")
            .attr("data-attr-key", "type")
            .css(this._tableInputCss())

        for (const type of ["continuous", "categorical", "vector", "other"]) {
            $select.append(
                $("<option></option>")
                    .attr("value", type)
                    .text(type)
            )
        }

        $select.val(value)

        return $select
    }

    _tableCell($content) {
        return $("<td></td>")
            .css({
                padding: "4px",
                borderBottom: "1px solid #eee",
                verticalAlign: "top"
            })
            .append($content)
    }

    _tableInputCss() {
        return {
            width: "100%",
            boxSizing: "border-box",
            fontSize: "13px",
            border: "1px solid #ccc",
            padding: "3px 5px",
            borderRadius: "3px",
            minWidth: 0
        }
    }

    _buildMetaForm(config) {
        this.$metaForm.empty()

        if (!config) {
            this.$metaForm.append(
                $("<div></div>")
                    .text("No metainfo config received")
                    .css({ color: "#777", fontSize: "13px" })
            )
            return
        }

        this._metaConfig = config

        const $form = $("<div></div>").css({
            display: "grid",
            gridTemplateColumns: "180px minmax(0, 1fr)",
            gap: "8px 10px",
            alignItems: "center"
        })

        this._appendTextField($form, "name", config.name ?? "")
        this._appendFormatField($form, config.format ?? "ij")
        this._appendNumberField($form, "count", config.count ?? 1)
        this._appendBoolField($form, "directed", config.directed ?? false)
        this._appendBoolField($form, "hetero", config.hetero ?? false)
        this._appendArrayField($form, "nodes", config.nodes ?? [])
        this._appendBoolField($form, "remap", config.remap ?? false)

        this.$metaForm.append($form)

        this.$metaForm.append(this._sectionTitle("Node attributes"))
        this.$metaForm.append(this._buildAttributesTable("node_attributes", config.node_attributes))

        this.$metaForm.append(this._sectionTitle("Edge attributes"))
        this.$metaForm.append(this._buildAttributesTable("edge_attributes", config.edge_attributes))

        this.$metaForm.append(this._sectionTitle("Labelings"))
        this.$metaForm.append(
            $("<div></div>")
                .text("Labelings editor will be added later")
                .css({
                    color: "#777",
                    fontSize: "13px",
                    border: "1px solid #ddd",
                    borderRadius: "4px",
                    padding: "8px"
                })
        )
    }

    _collectMetaConfig() {
        const config = {}
        const validationErrors = []

        this.$metaForm.find("[data-field]").each((i, el) => {
            const $el = $(el)
            const key = $el.attr("data-field")
            const raw = $el.val()

            if ($el.attr("data-kind") === "json") {
                config[key] = this._parseJsonField(raw, [])
                return
            }

            if ($el.attr("type") === "number") {
                const numberValue = Number(raw)
                config[key] = Number.isNaN(numberValue) ? 0 : numberValue
                return
            }

            if (el.tagName.toLowerCase() === "select" && (raw === "true" || raw === "false")) {
                config[key] = raw === "true"
                return
            }

            config[key] = raw
        })

        // Validate nodes field
        const $nodesField = this.$metaForm.find("[data-field='nodes']")
        if ($nodesField.length) {
            const nodes = config.nodes
            const isValid = (
                Array.isArray(nodes) &&
                nodes.length > 0 &&
                nodes.every(v => Number.isInteger(v) && v > 0)
            )
            if (!isValid) {
                validationErrors.push("nodes: must be a non-empty list of positive integers, e.g. [10, 20]")
                $nodesField.css("border", "1.5px solid #e03030")
            } else {
                $nodesField.css("border", "")
            }
        }

        config.node_attributes = this._collectAttributesInfo("node_attributes")
        config.edge_attributes = this._collectAttributesInfo("edge_attributes")

        // Пока оставляем как было в исходном config, потому что редактор labelings добавим позже.
        if (this._metaConfig && "labelings" in this._metaConfig)
            config.labelings = this._metaConfig.labelings

        return [config, validationErrors]
    }

    _collectAttributesInfo(fieldName) {
        const $tables = this.$metaForm.find(`table[data-attr-table="${fieldName}"]`)

        if ($tables.length === 0)
            return undefined

        if ($tables.length === 1 && !$($tables[0]).attr("data-attr-group"))
            return this._collectSingleAttributesTable($($tables[0]))

        const result = {}

        $tables.each((i, table) => {
            const $table = $(table)
            const groupName = $table.attr("data-attr-group")

            if (!groupName)
                return

            result[groupName] = this._collectSingleAttributesTable($table)
        })

        return result
    }

    _collectSingleAttributesTable($table) {
        const names = []
        const types = []
        const values = []

        $table.find("tr[data-attr-row]").each((i, row) => {
            const $row = $(row)

            const name = $row.find('[data-attr-key="name"]').val()
            const type = $row.find('[data-attr-key="type"]').val()
            const rawValues = $row.find('[data-attr-key="values"]').val()

            // Пустую строку имени считаем пустой строкой, но строку не выкидываем:
            // иначе можно случайно разъехать массивами names/types/values.
            names.push(name)
            types.push(type)
            values.push(this._parseAttributeValues(rawValues, type))
        })

        return {
            names,
            types,
            values
        }
    }

    _parseAttributeValues(raw, type) {
        const value = this._parseJsonField(raw, [])

        if (type === "continuous") {
            if (Array.isArray(value))
                return value
            return []
        }

        if (type === "categorical") {
            if (Array.isArray(value))
                return value
            return [value]
        }

        if (type === "vector") {
            if (Array.isArray(value))
                return value
            return []
        }

        if (type === "other") {
            if (Array.isArray(value))
                return value
            if (value === "")
                return []
            return [value]
        }

        return value
    }

    _parseJsonField(raw, fallback) {
        if (raw === undefined || raw === null)
            return fallback

        raw = String(raw).trim()

        if (raw === "")
            return fallback

        try {
            return JSON.parse(raw)
        } catch (e) {
            return raw
        }
    }

    // ── Errors ────────────────────────────────────────────────────────────────

    _showErrors(errors) {
        // this._errorCount = errors.length
        this.$errorsArea.val(errors)
    }
}