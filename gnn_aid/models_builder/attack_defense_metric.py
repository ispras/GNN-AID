from typing import Union, List

import numpy as np
import sklearn
import torch


def asr(
        y_predict_clean: Union[List, torch.Tensor, np.array],
        y_predict_after_attack_only: Union[List, torch.Tensor, np.array],
        **kwargs
):
    """
    Compute Attack Success Rate: fraction of predictions changed by the attack.

    Args:
        y_predict_clean (Union[List, torch.Tensor, np.array]): Model predictions before attack.
        y_predict_after_attack_only (Union[List, torch.Tensor, np.array]): Model predictions after attack.
        **kwargs: Ignored extra arguments.

    Returns:
        Fraction of predictions that differ between clean and attacked model.
    """
    if isinstance(y_predict_clean, torch.Tensor):
        if y_predict_clean.dim() > 1:
            y_predict_clean = y_predict_clean.argmax(dim=1)
        y_predict_clean.cpu()
    if isinstance(y_predict_after_attack_only, torch.Tensor):
        if y_predict_after_attack_only.dim() > 1:
            y_predict_after_attack_only = y_predict_after_attack_only.argmax(dim=1)
        y_predict_after_attack_only.cpu()
    return 1 - sklearn.metrics.accuracy_score(y_true=y_predict_clean, y_pred=y_predict_after_attack_only)


# TODO Kirill, change for any classic metric
def aucc_change_attack(
        y_predict_clean: Union[List, torch.Tensor, np.array],
        y_predict_after_attack_only: Union[List, torch.Tensor, np.array],
        y_true,
        **kwargs
):
    """
    Compute the accuracy drop caused by the attack.

    Args:
        y_predict_clean (Union[List, torch.Tensor, np.array]): Predictions before attack.
        y_predict_after_attack_only (Union[List, torch.Tensor, np.array]): Predictions after attack.
        y_true: Ground-truth labels.
        **kwargs: Ignored extra arguments.

    Returns:
        Difference between clean accuracy and attacked accuracy (positive means accuracy dropped).
    """
    if isinstance(y_predict_clean, torch.Tensor):
        if y_predict_clean.dim() > 1:
            y_predict_clean = y_predict_clean.argmax(dim=1)
        y_predict_clean.cpu()
    if isinstance(y_predict_after_attack_only, torch.Tensor):
        if y_predict_after_attack_only.dim() > 1:
            y_predict_after_attack_only = y_predict_after_attack_only.argmax(dim=1)
        y_predict_after_attack_only.cpu()
    if isinstance(y_true, torch.Tensor):
        if y_true.dim() > 1:
            y_true = y_true.argmax(dim=1)
        y_true.cpu()
    return (sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_predict_clean) -
            sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_predict_after_attack_only))


# TODO Kirill, change for any classic metric
def aucc_change_defense_only(
        y_predict_clean: Union[List, torch.Tensor, np.array],
        y_predict_after_defense_only: Union[List, torch.Tensor, np.array],
        y_true: Union[List, torch.Tensor, np.array],
        **kwargs
):
    """
    Compute the accuracy change caused by the defense alone (without an attack).

    Args:
        y_predict_clean (Union[List, torch.Tensor, np.array]): Predictions without defense.
        y_predict_after_defense_only (Union[List, torch.Tensor, np.array]): Predictions with defense only.
        y_true (Union[List, torch.Tensor, np.array]): Ground-truth labels.
        **kwargs: Ignored extra arguments.

    Returns:
        Difference between clean accuracy and defense-only accuracy.
    """
    if isinstance(y_predict_clean, torch.Tensor):
        if y_predict_clean.dim() > 1:
            y_predict_clean = y_predict_clean.argmax(dim=1)
        y_predict_clean.cpu()
    if isinstance(y_predict_after_defense_only, torch.Tensor):
        if y_predict_after_defense_only.dim() > 1:
            y_predict_after_defense_only = y_predict_after_defense_only.argmax(dim=1)
        y_predict_after_defense_only.cpu()
    if isinstance(y_true, torch.Tensor):
        if y_true.dim() > 1:
            y_true = y_true.argmax(dim=1)
        y_true.cpu()
    return (sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_predict_clean) -
            sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_predict_after_defense_only))


# TODO Kirill, change for any classic metric
def aucc_change_defense_with_attack(
        y_predict_after_attack_only: Union[List, torch.Tensor, np.array],
        y_predict_after_attack_and_defense: Union[List, torch.Tensor, np.array],
        y_true: Union[List, torch.Tensor, np.array],
        **kwargs
):
    """
    Compute the accuracy recovery achieved by the defense under attack.

    Args:
        y_predict_after_attack_only (Union[List, torch.Tensor, np.array]): Predictions under attack only.
        y_predict_after_attack_and_defense (Union[List, torch.Tensor, np.array]): Predictions under attack + defense.
        y_true (Union[List, torch.Tensor, np.array]): Ground-truth labels.
        **kwargs: Ignored extra arguments.

    Returns:
        Accuracy of attack+defense minus accuracy of attack only (positive means defense helped).
    """
    if isinstance(y_predict_after_attack_only, torch.Tensor):
        if y_predict_after_attack_only.dim() > 1:
            y_predict_after_attack_only = y_predict_after_attack_only.argmax(dim=1)
        y_predict_after_attack_only.cpu()
    if isinstance(y_predict_after_attack_and_defense, torch.Tensor):
        if y_predict_after_attack_and_defense.dim() > 1:
            y_predict_after_attack_and_defense = y_predict_after_attack_and_defense.argmax(dim=1)
        y_predict_after_attack_and_defense.cpu()
    if isinstance(y_true, torch.Tensor):
        if y_true.dim() > 1:
            y_true = y_true.argmax(dim=1)
        y_true.cpu()
    return (sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_predict_after_attack_and_defense) -
            sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_predict_after_attack_only))


class AttackMetric:
    """
    Metric for evaluating the impact of an attack on model predictions.
    Computes scores such as ASR and accuracy drop between clean and attacked outputs.
    """
    available_metrics = {
        "ASR": asr,
        "AuccAttackDiff": aucc_change_attack,
    }

    def __init__(
            self,
            name: str,
            **kwargs
    ):
        self.name = name
        self.kwargs = kwargs

    def compute(
            self,
            y_predict_clean: Union[List, torch.Tensor, np.array, None],
            y_predict_after_attack_only: Union[List, torch.Tensor, np.array, None],
            y_true: Union[List, torch.Tensor, np.array, None],
    ):
        """
        Compute the attack metric value.

        Args:
            y_predict_clean (Union[List, torch.Tensor, np.array, None]): Predictions before attack.
            y_predict_after_attack_only (Union[List, torch.Tensor, np.array, None]): Predictions after attack.
            y_true (Union[List, torch.Tensor, np.array, None]): Ground-truth labels.

        Returns:
            Numeric metric value.
        """
        if self.name in AttackMetric.available_metrics:
            return AttackMetric.available_metrics[self.name](
                y_predict_clean=y_predict_clean,
                y_predict_after_attack_only=y_predict_after_attack_only,
                y_true=y_true,
                **self.kwargs
            )
        raise NotImplementedError()


class DefenseMetric:
    """
    Metric for evaluating how well a defense mitigates an attack.

    Computes scores comparing clean, attack-only, defense-only, and attack+defense predictions.
    """
    available_metrics = {
        "AuccDefenseCleanDiff": aucc_change_defense_only,
        "AuccDefenseAttackDiff": aucc_change_defense_with_attack,
    }

    def __init__(
            self,
            name: str,
            **kwargs
    ):
        self.name = name
        self.kwargs = kwargs

    def compute(
            self,
            y_predict_clean: Union[List, torch.Tensor, np.array, None],
            y_predict_after_attack_only: Union[List, torch.Tensor, np.array, None],
            y_predict_after_defense_only: Union[List, torch.Tensor, np.array, None],
            y_predict_after_attack_and_defense: Union[List, torch.Tensor, np.array, None],
            y_true: Union[List, torch.Tensor, np.array, None],
    ):
        """
        Compute the defense metric value.

        Args:
            y_predict_clean (Union[List, torch.Tensor, np.array, None]): Predictions without attack or defense.
            y_predict_after_attack_only (Union[List, torch.Tensor, np.array, None]): Predictions under attack only.
            y_predict_after_defense_only (Union[List, torch.Tensor, np.array, None]): Predictions with defense only.
            y_predict_after_attack_and_defense (Union[List, torch.Tensor, np.array, None]): Predictions under attack + defense.
            y_true (Union[List, torch.Tensor, np.array, None]): Ground-truth labels.

        Returns:
            Numeric metric value.
        """
        if self.name in DefenseMetric.available_metrics:
            return DefenseMetric.available_metrics[self.name](
                y_predict_clean=y_predict_clean,
                y_predict_after_defense_only=y_predict_after_defense_only,
                y_predict_after_attack_only=y_predict_after_attack_only,
                y_predict_after_attack_and_defense=y_predict_after_attack_and_defense,
                y_true=y_true,
                **self.kwargs
            )
        raise NotImplementedError(f"Metric {self.name} is not implemented")
