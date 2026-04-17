#!/bin/bash

PROJECT_ROOT="$(dirname "$0")/../.."
export PYTHONPATH="$PROJECT_ROOT/gnn_aid:$PROJECT_ROOT:$PYTHONPATH"

MODE="$1"

if [[ "$MODE" == "attack" ]]; then
  echo ">> Run evasion_attack.py"
  python "$PROJECT_ROOT/tutorials/01_evasion_attack_defense/evasion_attack.py"

elif [[ "$MODE" == "defense" ]]; then
  echo ">> Run (defense_against_evasion.py)"
  python "$PROJECT_ROOT/tutorials/01_evasion_attack_defense/defense_against_evasion.py"

else
  echo "Specify the launch mode: attack или defense"
  echo ""
  echo "Examples:"
  echo "  bash run_example.sh attack"
  echo "  bash run_example.sh defense"
  exit 1
fi