# Evasion attacks and defense

This experiment demonstrates the behavior of the graph model under evasion attack and the defense capabilities of 
Gradient Regularization Defender. We use the Cora dataset and the GNN model.

---

## Folder contents
- `evasion_attack.py` — script for training GNN + attack.
- `defense_against_evasion.py` — script for training GNN + defense + attack.
- `run_example.sh` — script for running the experiment.
- `imgs/` — folder with images.
- `README.md` — description of the experiment.

---

## Quick start

Two launch modes are supported:
1. `attack` — only attack is performed.
2. `defense` — protection is performed, after which an attack is carried out on the protected model.

### 1. only attack
```bash
  bash run_example.sh attack
```

### 2. defense and attack
```bash
  bash run_example.sh defense
```

---
## Description of modes

### 1. only attack

In this mode:

- The model is trained **without any defense**.
- After training, the **FGSM** (Fast Gradient Sign Method) evasion attack is applied.
- Metrics are measured **before and after** the attack.

#### Example of the experimental result

|  Metric      ↓ \ Experiment →            | Before attack | After attack |
|-----------------|---------------|--------------|
| **F1 (macro)**         | \~0.86        | \~0.48       |
| **Accuracy**      | \~0.87        | \~0.49       |

The exact values may differ due to random weight initialization and model training process.

### 2. defense and attack

In this mode:

- The model is trained **with enabled defense**: `GradientRegularizationDefender`.
- The **FGSM** attack is then applied to the trained model.
- Metrics are evaluated **after the attack** on the defended model.

#### Example of the experimental result

| Metric      ↓ \ Experiment →                          | No attack, no defense | attack (no defense) | attack + defense |
|-------------------------------------------------------|------------------------|----------------------|------------------|
| **F1 (macro)**                                        | 0.84                   | 0.48                 | 0.67             |
| **Accuracy**                                          | 0.86                   | 0.53                 | 0.70             |

The exact values may differ due to random weight initialization and model training process.