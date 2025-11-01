# Basic GNN training

This experiment demonstrates the following steps of working with GNN within the **GNN-AID** framework:
1. Loading the dataset and preparing the data.
2. Building the GNN model.
3. Training the GNN model.
4. Saving and loading the model weights.
5. Assessing the quality of the trained model.

---

## Folder contents
- `train_gnn.py` — script for training GNN.
- `run_example.sh` — script for running the experiment.
- `imgs/` — folder with images.
- `README.md` — description of the experiment.

---

## Quick start

Run:
```bash
  bash run_example.sh
```

---

## Example of the experimental result
| Metric          | Value  |
| --------------- |--------|
| F1 (macro, test) | \~0.83 |
| Accuracy (test) | \~0.85 |

The exact values may differ due to random weight initialization and model training process.

