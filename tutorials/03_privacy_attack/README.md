# Membership Inference attacks and defense

This experiment demonstrates the behavior of the graph model under membership inference attack and the effectiveness of 
output noise-based defense. We use the Cora dataset and the GCN model.

---

## Folder contents
- `MI_attack.py` — script for training GCN + membership inference attack.
- `MI_defense.py` — script for training GCN + defense + membership inference attack.
- `README.md` — description of the experiment.
- `run_example.sh` — script for running the experiment.

---

## Quick start

Two launch modes are supported:
1. `attack` — only membership inference attack is performed.
2. `defense` — defense is applied against membership inference attack.

### 1. Membership inference attack
```bash
python MI_attack.py
```

### 2. Defense against membership inference
```bash
python MI_defense.py
```

---
## Description of modes

### 1. Membership inference attack

In this mode:
- The GCN_2l model is trained on Cora dataset.
- A **naive membership inference attack** is applied, labeling samples as training data based on higher model confidence.
- Attack performance metrics are measured.

#### Performance of Naive MI Attack
Note that this is not model performance, metrics being calculated on results of attack, for example accuracy means percentage of data that are right classified as train by MI Attacker

| Metric               | Value         |
|----------------------|---------------|
| **Accuracy**         | ~0.67         |
| **Precision (train)**| ~0.70         |
| **Recall (train)**   | ~0.94         |
| **F1 (train)**       | ~0.80         |

### 2. Defense against membership inference

In this mode:
- The model is trained **with output noise-based defense** against membership inference attacks.
- The same **membership inference attack** is applied to the defended model.
- Metrics demonstrate the defense effectiveness in reducing attack performance.

#### You will get these results
Note that this is not model performance, metrics being calculated on results of attack, for example accuracy means percentage of data that are right classified as train by MI Attacker
Therefore we see that the defense reduced MI attacker's performance

| Metric               | Attack (no defense) | Attack + defense |
|----------------------|---------------------|------------------|
| **Accuracy**         | 0.67                | 0.49             |
| **Precision (train)**| 0.70                | 0.75             |
| **Recall (train)**   | 0.94                | 0.16             |
| **F1 (train)**       | 0.80                | 0.26             |
