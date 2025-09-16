The tutorials in this folder provide examples of more complex, combined usage scenarios with the framework.

multi_attack_pipeline.py:
This file demonstrates running GIN_2l on the Cora dataset using two attacks:
CLGA (a poisoning attack) and FGSM (an evasion attack).
By commenting/uncommenting lines 73-74, you can examine four different pipelines:

Running the clean model
{'test': {'F1': 0.837926471703686, 'Accuracy': 0.8431734317343174}}

CLGA attack only
{'test': {'F1': 0.6468543445372832, 'Accuracy': 0.6734317343173432}}

FGSM attack only
{'test': {'F1': 0.6876629611752214, 'Accuracy': 0.7029520295202952}}

Simultaneous application of both attacks
{'test': {'F1': 0.553660031435681, 'Accuracy': 0.5885608856088561}}

Thus, we can observe that combining these attacks can be viewed as a more powerful attack strategy, capable of degrading the model's performance more severely than either attack could achieve individually.

multi_defense_pipeline.py:

This file explores a scenario of building a model with simultaneous protection against both poisoning attacks and evasion attacks.

When line 88 is commented out, a typical pipeline for defending the model against evasion attacks is executed.
{'test': {'F1': 0.8200952562430774, 'Accuracy': 0.8173431734317343}}

With line 88 uncommented, a defense against poisoning attacks (Jaccard Defense) is applied.
{'test': {'F1': 0.7756653295032969, 'Accuracy': 0.7749077490774908}}

Thus, we can see that it is possible to combine defenses against different types of attacks without critically degrading the model's performance.

combined_workflow.py:
Finally, this tutorial provides an example of simultaneously using the attack-defense module and the interpretation module in our framework.

The GIN_2l model undergoes a combined CLGA-FGSM attack pipeline (as in multi_attack_pipeline.py), and then the results of its operation are interpreted using GNNExplainer.