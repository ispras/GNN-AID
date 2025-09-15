The tutorials in this folder demonstrate the use of privacy attacks, specifically membership inference attacks.

The file `MI_attack.py` runs a GCN_2l model on the Cora dataset, employing a naive MI attack that labels samples as training data based on higher model confidence.

The performance of this naive attack is:
****

The file `MI_defense.py` modifies this pipeline by adding a defense against MI attacks through the introduction of noise to the model's outputs.

The performance of the attack when the defense is applied:
****