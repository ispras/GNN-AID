///
// Index of data elements present on backend.
// Each data element is identified by a list of values for a predefined list of keys.
// PrefixStorage keeps all possible value vectors according to hierarchical folder structure.
class PrefixStorage {
    constructor(keys) {
        this.keys = keys
        this.depth = this.keys.length
        this.content = {} // {K1 -> {K2 -> ... -> {K(d-1) -> [values]}...}

        this.selects = null // key -> select
    }

    size() {
        let count = function (obj) {
            return obj.constructor === Object
                ? Object.values(obj).reduce((s, x) => s + count(x), 0)
                : obj.length
        }
        return count(this.content)
    }

    // Find all items satisfying specified key values. Returns a new PrefixStorage.
    filter(key_values) {
        let self = this

        function filter(obj, depth) {
            let key = self.keys[depth]
            if (obj.constructor === Object) {
                if (key in key_values) { // take the value for 1 key
                    let value = key_values[key]
                    if (value.constructor === Object) // value could be a dict
                        value = JSON.stringify(value)
                    if (value in obj)
                        return filter(obj[value], depth + 1)
                    else // all the rest is empty
                        return []
                }
                else // filter all
                    return Object.fromEntries(
                        Object.entries(obj).map(kv => [kv[0], filter(kv[1], depth+1)]))
            }
            else { // set
                if (key in key_values) {
                    console.error("filtering by last key is not tested!")
                    // delete key_values[key]
                    let value = key_values[key]
                    return obj.includes(value) ? [value] : []
                }
                // copy of full set
                return new Array(...obj)
            }
        }

        // Remove keys
        let keys = new Array(...this.keys)
        for (const key of Object.keys(key_values)) {
            const index = keys.indexOf(key)
            if (index > -1)
                keys.splice(index, 1)
        }

        let ps = new PrefixStorage(keys)
        ps.content = filter(this.content, 0)
        return ps
    }

    // Build PrefixStorage from its json
    static fromJSON(string) {
        let obj = JSON_parse(string)
        let ps = new PrefixStorage(obj["keys"])
        ps.content = obj["content"]
        return ps
    }

    /**
     * Create menu selectors according to the contents with hierarchical dependency, i.e.
     * number of selectors == this.depth,
     * elements of selector[i] == this.contents at depth i.
     * @param $div - HTML element where to add selectors
     * @param dropFunc - called when user change any selector
     * @param setFunc - called when user choose option in the last selector
     * @param descriptionInfo - dict with descriptions of all options
     */
    buildCascadeMenu($div, dropFunc, setFunc, descriptionInfo) {
        let self = this
        let keys = this.keys
        let depth = this.depth

        this.selects = {}
        // Create selectors according to the keys
        for (let d = 0; d<keys.length; ++d) {
            const key = keys[d]
            let controlBlock = $("<div></div>").attr("class", "control-block")
            let inner = $("<label></label>").text(key)
            controlBlock.html(inner)
            let $select = $("<select></select>")
            this.selects[d] = $select
            inner.after($select)
            $div.append(controlBlock)
        }

        // Fill options of the main selector
        // TODO can we add it to cycle below?
        let g = this.selects[0]
        g.empty()
        g.append($("<option></option>").attr("selected", "true").attr("value", "")
            .attr("disabled", "").text(`<select ${keys[0]}>`))
        let options = depth > 1 ? Object.keys(self.content) : self.content
        options = options.toSorted((a, b) => {
            return a.toLowerCase().localeCompare(b.toLowerCase())})
        for (const opt of options) {
            let $option = $("<option></option>").attr("value", opt).text(opt)
            if (descriptionInfo && keys[0] in descriptionInfo)
                $option.attr("title", descriptionInfo[keys[0]][opt])
            g.append($option)
        }

        // Add change listeners
        let obj = self.content
        for (let d=1; d<depth; ++d) {
            let $subSel = this.selects[d]

            this.selects[d-1].change(() => {
                // Call drop function
                if (dropFunc)
                    dropFunc()

                // Drop selected values from descendant selectors
                for (let j=d; j<depth; ++j)
                    this.selects[j].empty()

                // Get a list of variants from the corresponding depth
                let object = obj
                for (let j=0; j<d; ++j) {
                    object = object[this.selects[j].get(0).value]
                }
                object = d < depth-1 ? Object.keys(object) : (object)
                object = object.toSorted((a, b) => {
                    return a.toLowerCase().localeCompare(b.toLowerCase())})

                // Fill options in a successor select
                $subSel.append($("<option></option>").attr("selected", "true").attr("value", "")
                    .attr("disabled", "").text(`<select ${keys[d]}>`))
                for (const val of object) {
                    let $option = $("<option></option>").attr("value", val).text(val)
                    if (descriptionInfo && keys[d] in descriptionInfo)
                        $option.attr("title", descriptionInfo[keys[d]][val])
                    $subSel.append($option)
                }

                // If the successor select has only 1 option, choose it
                if (object.length === 1)
                    $subSel.val(object[0]).trigger('change')
            })
        }
        // Last selector calls drop and set setFunc
        this.selects[depth-1].change(() => {
            if (dropFunc)
                dropFunc()
            if (setFunc)
                setFunc()
        })
    }

    // dropCascadeMenu(menuPrefix) { // FIXME do we use it?
    //     let keysNoSpaces = []
    //
    //     // Create selectors according to the keys
    //     for (let d = 0; d<this.keys.length; ++d)
    //         keysNoSpaces.push(nameToId(this.keys[d]))
    //
    //     let g = $(`#${menuPrefix(0)}-${keysNoSpaces[0]}`)
    //     g.empty()
    //     g.append($("<option></option>").attr("selected", "true")
    //         .attr("value", "").attr("disabled", "")
    //         .text(`<select ${keysNoSpaces[0]}>`))
    //     let options = this.depth > 1 ? Object.keys(this.content) : this.content
    //     for (const key of options) {
    //         g.append($("<option></option>").attr("value", key).text(key))
    //     }
    //
    //     // Drop selected values from descendant selectors
    //     for (let j=1; j<this.depth; ++j) {
    //         $(`#${menuPrefix(j)}-${keysNoSpaces[j]}`).empty()
    //     }
    // }

    getConfig() {
        let res = {}
        for (let d = 0; d<this.keys.length; ++d)
            res[this.keys[d]] = this.selects[d].get(0).value
        return res
    }
}

///
// Index of data elements present on backend.
// Each data element is identified by a list of keys and object.
class TuplePrefixStorage {
    constructor() {
        this.content = {} // {K1 -> {K2 -> ... -> {K(d-1) -> [obj]}...}

        this.selects = null // selected object as a list
    }

    size() {
        let count = function (obj) {
            return obj.constructor === Object
                ? Object.values(obj).reduce((s, x) => s + count(x), 0)
                : obj.length
        }
        return count(this.content)
    }

    // // Find all items satisfying specified key values. Returns a new PrefixStorage.
    // filter(key_values) {
    //     let self = this
    //
    //     function filter(obj, depth) {
    //         let key = self.keys[depth]
    //         if (obj.constructor === Object) {
    //             if (key in key_values) { // take the value for 1 key
    //                 let value = key_values[key]
    //                 if (value.constructor === Object) // value could be a dict
    //                     value = JSON.stringify(value)
    //                 if (value in obj)
    //                     return filter(obj[value], depth + 1)
    //                 else // all the rest is empty
    //                     return []
    //             }
    //             else // filter all
    //                 return Object.fromEntries(
    //                     Object.entries(obj).map(kv => [kv[0], filter(kv[1], depth+1)]))
    //         }
    //         else { // set
    //             if (key in key_values) {
    //                 console.error("filtering by last key is not tested!")
    //                 // delete key_values[key]
    //                 let value = key_values[key]
    //                 return obj.includes(value) ? [value] : []
    //             }
    //             // copy of full set
    //             return new Array(...obj)
    //         }
    //     }
    //
    //     // Remove keys
    //     let keys = new Array(...this.keys)
    //     for (const key of Object.keys(key_values)) {
    //         const index = keys.indexOf(key)
    //         if (index > -1)
    //             keys.splice(index, 1)
    //     }
    //
    //     let ps = new PrefixStorage(keys)
    //     ps.content = filter(this.content, 0)
    //     return ps
    // }

    // Build PrefixStorage from its json
    static fromJSON(string) {
        let ps = new TuplePrefixStorage()
        ps.content = JSON_parse(string)
        return ps
    }

    /**
     * Create menu selectors according to the contents with hierarchical dependency, i.e.
     * number of selectors == this.depth,
     * elements of selector[i] == this.contents at depth i.
     * @param $div - HTML element where to add selectors
     * @param dropFunc - called when user change any selector
     * @param setFunc - called when user choose option in the last selector
     * @param descriptionInfo - dict with descriptions of all options
     */
    buildCascadeMenu($div, dropFunc, setFunc, descriptionInfo) {
        let levelMap = new Map(); // уровень -> список подменю на этом уровне

        let add = ($aDiv, content, level = 0, path=[]) => {
            if (!levelMap.has(level)) levelMap.set(level, []);
            for (const [k, v] of Object.entries(content)) {
                let $item = $("<div></div>")//.attr("style", "width: 150px");
                $aDiv.append($item);
                $item.addClass("dropdownmenuitem");

                let $subDiv
                let popperInstance
                let newPath = path.concat([k])

                let isObject = Array.isArray(v)
                if (isObject) {
                    $item.append($("<b></b>").text(k));
                    $item.attr("title", v[0]);
                    $item.click((e) => {
                        if (dropFunc) dropFunc();
                        this.selects = newPath
                        if (setFunc) setFunc(newPath);
                        // e.stopPropagation();
                    });
                } else {
                    $item.append(k);
                    $item.addClass("has-submenu");

                    $subDiv = $("<div></div>").attr("class", "submenu").css("position", "absolute").hide();
                    $("body").append($subDiv);
                    add($subDiv, v, level + 1, newPath);

                    levelMap.get(level + 1).push($subDiv);

                    popperInstance = Popper.createPopper($item[0], $subDiv[0], {
                        placement: 'right-start',
                        modifiers: [
                            // { name: 'preventOverflow', options: { boundary: document.body } },
                            // { name: 'offset', options: { offset: [0, 0] } },
                        ],
                    });

                    // If click on submenu title - do nothing
                    $item.click(function (e) {
                        e.stopPropagation();
                    });
                }
                $item.on("mouseenter", function () {
                    // Скрываем ВСЕ подменю текущего и более глубоких уровней
                    for (let [lvl, subs] of levelMap.entries()) {
                        if (lvl >= level + 1) {
                            subs.forEach($s => $s.hide());
                        }
                    }

                    if (!isObject) {
                        // Показываем текущее
                        $subDiv.show();
                        popperInstance.update();
                    }
                });
            }
        };

        add($div, this.content, 0, [])

    }

    getConfig() {
        console.log('selected', this.selects)
        return this.selects
    }
}
