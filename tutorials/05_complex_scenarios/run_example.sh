#!/bin/bash
PROJECT_ROOT="$(dirname "$0")/../.."
export PYTHONPATH="$PROJECT_ROOT/gnn_aid:$PROJECT_ROOT:$PYTHONPATH"
python "$PROJECT_ROOT/tutorials/05_complex_scenarios/multi_attack_pipeline.py"
python "$PROJECT_ROOT/tutorials/05_complex_scenarios/multi_defense_pipeline.py"
python "$PROJECT_ROOT/tutorials/05_complex_scenarios/combined workflow.py"
