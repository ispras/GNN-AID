#!/bin/bash
PROJECT_ROOT="$(dirname "$0")/../.."
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$PYTHONPATH"
python "$PROJECT_ROOT/tutorials/03_privacy_attack/MI_attack.py"
python "$PROJECT_ROOT/tutorials/03_privacy_attack/MI_defense.py"