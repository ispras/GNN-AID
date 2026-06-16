#!/bin/bash
PROJECT_ROOT="$(dirname "$0")/../.."
export PYTHONPATH="$PROJECT_ROOT/gnn_aid:$PROJECT_ROOT:$PYTHONPATH"
python "$PROJECT_ROOT/tutorials/04_interpretability/gnn_explainer_example.py"
