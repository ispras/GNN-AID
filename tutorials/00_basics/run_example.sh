#!/bin/bash
PROJECT_ROOT="$(dirname "$0")/../.."
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$PYTHONPATH"
python "$PROJECT_ROOT/tutorials/00_basics/train_gnn.py"