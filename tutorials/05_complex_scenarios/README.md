# Combined attacks and defenses workflow

This experiment demonstrates complex usage scenarios with the framework, including combined attacks, multi-defense pipelines, 
and integrated interpretation workflows. We use the Cora dataset and the GIN model.

---

## Folder contents
- `multi_attack_pipeline.py` — script for running combined attack scenarios.
- `multi_defense_pipeline.py` — script for running multi-defense scenarios.
- `combined_workflow.py` — script for integrated attack-defense-interpretation workflow.
- `README.md` — description of the experiments.
- `run_example.sh` — script for running the experiment.

---

## Quick start

### 1. Multi-attack pipeline
```bash
python multi_attack_pipeline.py
```

### 2. Multi-defense pipeline  
```bash
python multi_defense_pipeline.py
```

### 3. Combined workflow
```bash
python combined_workflow.py
```

---
## Experimental results

### 1. Multi-attack pipeline

This script demonstrates running GIN_2l on Cora dataset with four different configurations:

| Configuration | F1 (macro) | Accuracy |
|---------------|------------|----------|
| Clean model | 0.8379 | 0.8432 |
| CLGA attack only | 0.6469 | 0.6734 |
| FGSM attack only | 0.6877 | 0.7030 |
| Both attacks combined | 0.5537 | 0.5886 |

The results show that combining attacks creates a more powerful attack strategy, degrading model performance more severely than individual attacks.

### 2. Multi-defense pipeline

This script explores building models with simultaneous protection against multiple attack types:

| Configuration | F1 (macro) | Accuracy |
|---------------|------------|----------|
| Defense against evasion attacks | 0.8201 | 0.8173 |
| Defense against both evasion and poisoning attacks | 0.7757 | 0.7749 |

The results demonstrate that defenses against different attack types can be combined without critically degrading model performance.

### 3. Combined workflow

This tutorial provides an integrated workflow example:
- GIN_2l model undergoes combined CLGA-FGSM attack pipeline
- Attack results are interpreted using GNNExplainer
- Demonstrates simultaneous use of attack-defense and interpretation modules