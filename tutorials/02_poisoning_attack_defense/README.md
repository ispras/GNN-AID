These tutorials focus on the application of poisoning attacks (and defenses against them) within our framework.

Let's consider poisoning_attack.py:
Running this file with line 65 commented out runs the clean GIN_2l model on the Cora dataset.
At the end of the output, the user receives the model's quality metrics on the test portion of the dataset:

{'test': {'F1': 0.837926471703686, 'Accuracy': 0.8431734317343174}}

Running with line 65 uncommented executes the CLGA poisoning attack.
(The parameters are chosen so that the attack is effective but also has an optimal duration for the tutorial. Approximate training time is 3 minutes).

Similarly, the user receives the following results:

{'test': {'F1': 0.6468543445372832, 'Accuracy': 0.6734317343173432}}

Thus, we can see that using CLGA degrades the model's final performance on the dataset.

Let's consider defense_against_poisoning.py:
This file runs a pipeline similar to the previous one, but with the addition of a Jaccard-based defense against poisoning attacks. While this defense is somewhat naive, it allows for quickly observing the degradation in the CLGA attack's performance.
The user will obtain the following results:
{'test': {'F1': 0.6780780144051877, 'Accuracy': 0.7140221402214022}}
