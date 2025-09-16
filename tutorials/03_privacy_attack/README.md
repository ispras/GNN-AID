The tutorials in this folder demonstrate the use of privacy attacks, specifically membership inference attacks.

The file `MI_attack.py` runs a GCN_2l model on the Cora dataset, employing a naive MI attack that labels samples as training data based on higher model confidence.

The performance of this naive attack is:
MI Attack accuracy: {'accuracy': 0.67, 'precision_train': 0.6979166666666666, 'recall_train': 0.9436619718309859, 'f1_train': 0.8023952095808383}


The file `MI_defense.py` modifies this pipeline by adding a defense against MI attacks through the introduction of noise to the model's outputs.

The performance of the attack when the defense is applied:
MI Attack accuracy: {'accuracy': 0.49, 'precision_train': 0.75, 'recall_train': 0.15789473684210525, 'f1_train': 0.2608695652173913}
