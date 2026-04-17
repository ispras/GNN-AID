#!/bin/bash
PROJECT_ROOT="$(dirname "$0")/../.."
export PYTHONPATH="$PROJECT_ROOT/gnn_aid:$PROJECT_ROOT:$PYTHONPATH"
python "$PROJECT_ROOT/tutorials/02_poisoning_attack_defense/poisoning_attack.py"
python "$PROJECT_ROOT/tutorials/02_poisoning_attack_defense/defense_against_poisoning.py"