/**
 * Test Stand for Dataset Visualization
 */

// ============================================================================
// MOCK CONTROLLER
// ============================================================================

class MockController {
    constructor() {
        this.presenter = new MockPresenter();

        this._datasetInfo = null;
        this._datasetData = null;
        this._varData = null;
        this._visiblePart = null;
    }

    async ajaxRequest(url, params) {
        console.log('ajaxRequest:', url, params);

        if (url === '/dataset') {
            if (params.get === 'data') {
                return this._datasetData;
            }
            if (params.get === 'var_data') {
                return this._varData || '';
            }
            if (params.set === 'visible_part') {
                this._visiblePart = JSON.parse(params.part);
                console.log('Visible part set:', this._visiblePart);
                return '';
            }
        }
        if (url === '/model') {
            if (params.get === 'satellites') {
                return '';
            }
        }
        return '';
    }

    setDatasetInfo(info) { this._datasetInfo = info; }
    setDatasetData(data) { this._datasetData = data; }
    setVarData(data) { this._varData = data; }
}

// ============================================================================
// MOCK PRESENTER
// ============================================================================

class MockPresenter extends Presenter {
    constructor() {
        super();
        this.menuDatasetView = { state: MVState.LOCKED };
        this.datasetView = null;
        this.visualsView = null;
    }

    createViews() {
        this.visualsView = new VisualsView(
            $("#menu-visuals-view"), null, ["dc", "dvc"]);
    }
}

// ============================================================================
// GLOBAL CONTROLLER INSTANCE
// ============================================================================

let controller = new MockController();

// ============================================================================
// EXAMPLE DATA
// ============================================================================

const EXAMPLE_NEIGHBORHOOD = {
    datasetInfo: {
        name: "example_neighborhood",
        count: 1,
        directed: false,
        hetero: false,
        nodes: [5],
        node_attributes: {
            names: ["a", "b"],
            types: ["continuous", "categorical"],
            values: [[0, 1], ["A", "B", "C"]]
        },
        labelings: {
            "node-classification": {
                "binary": 2
            },
            "edge-classification": {
                "binary": 2
            }
        }
    },
    datasetData: {
        nodes: [
            [0],
            [1],
            [2, 3, 4]
        ],
        edges: [
            [],
            [[1, 0]],
            [[2, 1], [4, 1], [3, 1]]
        ],
        graphs: null,
        node_attributes: {
            "a": [{"0": 1, "1": 1, "2": 0.6, "3": 0.7, "4": 0.5}],
            "b": [{"0": "A", "1": "A", "2": "B", "3": "C", "4": "A"}]
        }
    },
    datasetVar: {
      "node": {
        "labels": [1, 1, 0, 0, 1],
        "features": [[1], [1], [0], [1], [1]],
        "logits": null,
        "predictions": null,
        "answers": null
      },
      "edge": {
        "features": [[0, 1, 2],[0, 1, 2],[0, 1, 2],[0, 1, 2]],
        "embeddings": [[0.1],[-0.4],[0.1],[0.1]],
        "predictions": [[0.1],[0.4],[0.1],[0.1]],
        "train-test-mask": [1,2,3,1],
        "labels": [0,1,0,1]
      },
      "graph": {
        "labels": null,
        "features": null,
        "logits": null,
        "predictions": null,
        "answers": null
      }
    },
    labeling: "binary",
    // task: "edge-prediction",
    task: "edge-classification",
};

const EXAMPLE_SINGLE_GRAPH = {
    datasetInfo: {
        name: "example_whole_graph",
        count: 1,
        directed: false,
        hetero: false,
        nodes: [8],
        node_attributes: {
            names: ["a", "b"],
            types: ["continuous", "categorical"],
            values: [[0, 1], ["A", "B", "C"]]
        },
        labelings: {
            "node-classification": {
                "binary": 2
            },
            "edge-classification": {
                "binary": 2
            }
        }
    },
    datasetData: {
        nodes: 8,
        edges: [[0,1],[1,2],[1,3],[1,4],[2,3],[2,5],[2,6],[4,5],[4,7],[6,7]],
        graphs: null,
        node_attributes: {
            "a": [{"0": 1, "1": 1, "2": 0.6, "3": 0.7, "4": 0.5, "5": 0.7, "6": 0.5, "7": 0.5}],
            "b": [{"0": "A", "1": "A", "2": "B", "3": "C", "4": "A", "5": "C", "6": "B", "7": "A"}]
        }
    },
    datasetVar: {
      "node": {
        "features": [
            [1],[1],[0.6000000238418579],[0.699999988079071],[0.5],[0.5],[0.699999988079071],[0.5]],
        // "labels": [1,1,1,1,0,0,0,0]
      },
      "edge": {
        "features": [[0, 1, 2],[0, 1, 2],[0, 1, 2],[0, 1, 2],[0, 1, 2],[0, 1, 2],[0, 1, 2],[0, 1, 2],[0, 1, 2],[0, 1, 2]],
        "embeddings": [[0.1],[-0.4],[0.1],[0.1],[-0.1],[0.2],[0.1],[-0.6],[1.1],[0.3]],
        "predictions": [[0.1],[0.4],[0.1],[0.1],[0.1],[0.2],[0.1],[0.1],[0.1],[0.3]],
        "train-test-mask": [1,2,3,1,2,3,1,2,3,1],
        "labels": [0,1,0,1,0,1,0,1,0,1]
      },
      "graph": {}
    },
    // task: "edge-prediction",
    task: "edge-classification",
    labeling: "binary",
    oneHotableFeature: false
}

const EXAMPLE_MULTIPLE_GRAPHS = {
    datasetInfo: {
        name: "example_multi",
        count: 3,
        directed: false,
        hetero: false,
        nodes: [3, 4, 5],
        node_attributes: {
            names: ["type"],
            types: ["categorical"],
            values: [["alpha", "beta", "gamma"]]
        },
        labelings: {
            "graph-classification": {
                "binary": 2
            },
            "graph-regression": {
                "binary": 0
            }
        }
    },
    datasetData: {
      "edges": [
          [[0,1],[1,2],[2,3],[0,3]],
          [[0,1],[0,2],[0,3],[0,4]]]
        ,
      "nodes": [4,5],
      "graphs": [1,2],
      "node_attributes": {
        "type": {
          "1": {
            "0": "gamma",
            "1": "beta",
            "2": "gamma",
            "3": "gamma"
          },
          "2": {
            "0": "beta",
            "1": "gamma",
            "2": "gamma",
            "3": "alpha",
            "4": "beta"
          }
        }
      }
    },
    datasetVar: {
      "node": {
        "features": [
          [[0,0,1],[0,1,0],[0,0,1],[0,0,1]],
          [[0,1,0],[0,0,1],[0,0,1],[1,0,0],[0,1,0]]
        ]
      },
      "edge": {},
      "graph": {
        "labels": [1,0],
        "features": [[0],[0]],
        "embeddings": [[2, 1, 3, 1, 1, 1],[-1, -2, 0.4]],
        "predictions": [[0.1, 0.9],[0.8, 0.2]],
        "train-test-mask": [1,2],
      }
    },
    task: "graph-classification",
    labeling: "binary",
    oneHotableFeature: true
};

const EXAMPLE_HETEROGRAPH = {
    // TODO: fill in later
};

const ALL_EXAMPLES = {
    "neighborhood": EXAMPLE_NEIGHBORHOOD,
    "single-graph": EXAMPLE_SINGLE_GRAPH,
    "multiple-graphs": EXAMPLE_MULTIPLE_GRAPHS,
};

let currentExampleKey = new URLSearchParams(window.location.search).get('example') || "neighborhood";

// ============================================================================
// TEST DATASET VIEW
// ============================================================================

class TestDatasetView extends View {
    constructor($div) {
        super($div, null, []);
        this.visView = controller.presenter.visualsView;
        this._tag = "dv-" + timeBasedId();

        let $svgDiv = $("<div></div>").attr("id", "dataset-svg")
            .css({
                position: "absolute",
                top: 0, bottom: 0, left: 0, right: 0,
                overflow: "scroll"
            });
        this.$div.append($svgDiv);
        this.svgPanel = new SvgPanel($svgDiv[0]);

        this.$upLeftInfoDiv = $('#info-upleft');
        this.$bottomLeftInfoDiv = $('#info-bottomleft');
        this.$upRightInfoDiv = $('#info-upright');
        this.$bottomRightInfoDiv = $('#info-bottomright');

        createColormapImage(this.$bottomLeftInfoDiv[0], IMPORTANCE_COLORMAP);

        this.datasetInfo = null;
        this.datasetVar = null;
        this.explanation = null;
        this.visibleGraph = null;
        this.task = null;
        this.labeling = null;
        this.oneHotableFeature = null;
    }

    setNodeInfo(node, graph) {
        let html = '';
        if (this.visibleGraph) {
            html += this.visibleGraph.getInfo() + '<br>';
        }
        if (node != null) {
            html += "<b>Selected</b>";
            if (graph != null) html += ` graph: ${graph},`;
            html += ` node: ${node}`;
            html += ' ' + this.visibleGraph.getNodeInfo(node, graph);
            let attrString = JSON.stringify(this.visibleGraph.getNodeAttrs(node, graph));
            if (attrString && attrString.length > 120)
                attrString = attrString.slice(0, 120) + '...';
            if (attrString)
                html += `<br>Attributes: ${attrString}`;
        }
        this.$upLeftInfoDiv.html(html);
    }

    async onNodeClick(event, node, graph) {
        if (event === "left") {
            this.setNodeInfo(node, graph);
        }
        else if (event === "double") {
            if (this.visibleGraph instanceof Neighborhood) {
                await this.visibleGraph.setNode(node);
            }
        }
    }

    async setDataset() {
        blockLeftMenu(true);
        this.dropDatasetVar();
        this.dropDataset();

        if (this.datasetInfo.count > 1) {
            // Multiple graphs
            this.visibleGraph = new MultipleGraphs(this.datasetInfo, this.svgPanel);
        }
        else {
            // Single graph - choose mode based on example type
            if (currentExampleKey === "neighborhood") {
                this.visibleGraph = new Neighborhood(this.datasetInfo, this.svgPanel);
            }
            else if (currentExampleKey === "heterograph") {
                this.visibleGraph = new HeteroGraph(this.datasetInfo, this.svgPanel);
            }
            else {
                // single-graph and others default to Graph
                this.visibleGraph = new Graph(this.datasetInfo, this.svgPanel);
            }
        }

        this.visibleGraph.beforeInit = this.beforeInit.bind(this);
        this.visibleGraph.afterInit = this.afterInit.bind(this);
        this.visibleGraph.onNodeClick = this.onNodeClick.bind(this);

        this.visibleGraph.defineVisibleConfig();

        await this.beforeInit();
        await this.visibleGraph.init();
        await this.afterInit();

        blockLeftMenu(false);

        if (this.datasetVar)
            this.setDatasetVar();
    }

    dropDataset() {
        if (this.visibleGraph) this.visibleGraph.drop();
        this.visibleGraph = null;
    }

    setDatasetVar() {
        this.visibleGraph.initVar(this.datasetVar, this.task, this.labeling, this.oneHotableFeature);
    }

    dropDatasetVar() {
        if (this.visibleGraph) this.visibleGraph.dropVar();
    }

    async beforeInit() {
        await controller.ajaxRequest('/dataset',
            {set: "visible_part", part: JSON_stringify(this.visibleGraph.visibleConfig)});

        let data = await controller.ajaxRequest('/dataset', {get: "data"});
        this.visibleGraph.datasetData = data;
    }

    async afterInit() {
        let data = await controller.ajaxRequest('/dataset', {get: "var_data"});
        if (data !== '') {
            this.datasetVar = data;
        }

        this.visibleGraph.checkLightMode();
        this.visibleGraph.draw();
        this.setNodeInfo();
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

let testDatasetView = null;

async  function initTestStand() {
    console.log('Initializing test stand...');
    updateStatus('Initializing...');

    try {
        controller.presenter.createViews();

        let config = ALL_EXAMPLES[currentExampleKey];

        controller.setDatasetInfo(config.datasetInfo);
        controller.setDatasetData(config.datasetData);
        controller.setVarData(config.datasetVar);

        let $container = $('#dataset-container');
        testDatasetView = new TestDatasetView($container);
        controller.presenter.datasetView = testDatasetView;

        testDatasetView.datasetInfo = config.datasetInfo;
        testDatasetView.datasetVar = config.datasetVar;
        testDatasetView.task = config.task;
        testDatasetView.labeling = config.labeling;
        testDatasetView.oneHotableFeature = config.oneHotableFeature;

        await controller.presenter.visualsView.onInit("dvc", config.datasetInfo);

        setupUIHandlers();

        // Populate example selector
        let $exampleSelect = $('#example-select');
        for (let key of Object.keys(ALL_EXAMPLES)) {
            let example = ALL_EXAMPLES[key];
            if (example.datasetInfo) {
                $exampleSelect.append($("<option></option>").val(key).text(key));
            }
        }
        $exampleSelect.val(currentExampleKey);

        // Render
        testDatasetView.setDataset();

        controller.presenter.visualsView.onSubmit("dvc", {});

        updateStatus('Ready: ' + (config.datasetInfo.name || 'test dataset'));

    } catch (e) {
        console.error('Init error:', e);
        updateStatus('Error: ' + e.message, true);
    }
}

function setupUIHandlers() {
    // Example selector - changes URL and reloads
    $('#example-select').on('change', function() {
        let newExample = $(this).val();
        let url = new URL(window.location);
        url.searchParams.set('example', newExample);
        window.location.href = url;
    });

    $('#btn-reinit').on('click', async function() {
        await testDatasetView.setDataset();
        updateStatus('Reinitialized');
    });

    $('#btn-download').on('click', function() {
        if (testDatasetView && testDatasetView.svgPanel) {
            let svgData = testDatasetView.svgPanel.svg.outerHTML;
            let svgBlob = new Blob([svgData], {type: "image/svg+xml;charset=utf-8"});
            let svgUrl = URL.createObjectURL(svgBlob);
            let downloadLink = document.createElement("a");
            downloadLink.href = svgUrl;
            downloadLink.download = "dataset.svg";
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
            updateStatus('SVG downloaded');
        }
    });
}

function updateStatus(message, isError = false) {
    let $status = $('#status');
    $status.text(message);
    if (isError) {
        $status.addClass('error');
    } else {
        $status.removeClass('error');
    }
    console.log('Status:', message);
}