# Poisoning attacks and defense

This experiment demonstrates the behavior of the graph model under poisoning attack and the defense capabilities of 
Jaccard-based defender. We use the Cora dataset and the GIN model.

---

## Folder contents
- `poisoning_attack.py` — script for training GIN + attack.
- `defense_against_poisoning.py` — script for training GIN + defense + attack.
- `README.md` — description of the experiment.
- `run_example.sh` — script for running the experiment.

---

## Quick start

Two launch modes are supported:
1. `clean` — only clean model training without attack.
2. `attack` — poisoning attack is performed.
3. `defense` — defense is applied against poisoning attack.

### 1. Clean model training
```bash
python poisoning_attack.py  # with line 65 commented
```

### 2. Poisoning attack
```bash
python poisoning_attack.py  # with line 65 uncommented
```

### 3. Defense against poisoning
```bash
python defense_against_poisoning.py
```

---
## Description of modes

### 1. Clean model training

In this mode:
- The GIN_2l model is trained **without any attack or defense** on Cora dataset.
- Model quality metrics are measured on the test portion.

#### Performance of clean model

| Metric          | Value         |
|-----------------|---------------|
| **F1 (macro)**  | ~0.84         |
| **Accuracy**    | ~0.84         |

### 2. Poisoning attack

In this mode:
- The **CLGA poisoning attack** is applied during model training.
- Attack parameters are optimized for effectiveness and reasonable training time (~3 minutes).
- Metrics are measured after attack completion.

#### Performance of the attacked model

| Metric          | Clean model | After attack |
|-----------------|-------------|--------------|
| **F1 (macro)**  | ~0.84       | ~0.65        |
| **Accuracy**    | ~0.84       | ~0.67        |

### 3. Defense against poisoning

In this mode:
- The model is trained **with Jaccard-based defense** against poisoning attacks.
- The **CLGA attack** is applied to the defended model.
- JaccardDefense may not be the most prospect poison defense but aim of this tutorial is to demonstrate poison defense usage within GNN-AID

#### The results user will get

| Metric          | Clean model | Attack (no defense) | Attack + defense |
|-----------------|-------------|---------------------|------------------|
| **F1 (macro)**  | 0.84        | 0.65                | 0.68             |
| **Accuracy**    | 0.84        | 0.67                | 0.71             |

