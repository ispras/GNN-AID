from typing import Union, List, Callable, Any

import numpy as np
import sklearn
import torch


def asr(
        y_predict_clean: Union[List, torch.Tensor, np.array],
        y_predict_after_attack_only: Union[List, torch.Tensor, np.array],
        **kwargs
):
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
        if self.name in AttackMetric.available_metrics:
            return AttackMetric.available_metrics[self.name](
                y_predict_clean=y_predict_clean,
                y_predict_after_attack_only=y_predict_after_attack_only,
                y_true=y_true,
                **self.kwargs
            )
        raise NotImplementedError()


class DefenseMetric:
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
