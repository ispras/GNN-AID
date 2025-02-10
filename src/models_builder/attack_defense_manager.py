import copy
import json
import os
import warnings
from pathlib import Path
from typing import Type, Union, List, Any

import numpy as np
import torch

from base.datasets_processing import GeneralDataset

for pack in [
    'defense.GNNGuard.gnnguard',
    'defense.JaccardDefense.jaccard_def',
]:
    try:
        __import__(pack)
    except ImportError:
        print(f"Couldn't import Explainer from {pack}")


class FrameworkAttackDefenseManager:
    """
    """

    def __init__(
            self,
            gen_dataset: GeneralDataset,
            gnn_manager,
            device: str = None
    ):
        if device is None:
            device = "cpu"
        self.device = device
        self.gnn_manager = gnn_manager
        self.gen_dataset = gen_dataset

        self.available_attacks = {
            "poison": True if gnn_manager.poison_attacker.name != "EmptyPoisonAttacker" else False,
            "evasion": True if gnn_manager.evasion_attacker.name != "EmptyEvasionAttacker" else False,
            "mi": True if gnn_manager.mi_attacker.name != "EmptyMIAttacker" else False,
        }

        self.available_defense = {
            "poison": True if gnn_manager.poison_defender.name != "EmptyEvasionDefender" else False,
            "evasion": True if gnn_manager.evasion_defender.name != "EmptyPoisonAttacker" else False,
            "mi": True if gnn_manager.mi_defender.name != "EmptyMIDefender" else False,
        }

        self.start_attack_defense_flag_state = {
            "poison_attack": self.gnn_manager.poison_attack_flag,
            "evasion_attack": self.gnn_manager.evasion_attack_flag,
            "mi_attack": self.gnn_manager.mi_attack_flag,
            "poison_defense": self.gnn_manager.poison_defense_flag,
            "evasion_defense": self.gnn_manager.evasion_defense_flag,
            "mi_defense": self.gnn_manager.mi_defense_flag,
        }

    def set_clear_model(
            self
    ) -> None:
        self.gnn_manager.poison_attack_flag = False
        self.gnn_manager.evasion_attack_flag = False
        self.gnn_manager.mi_attack_flag = False
        self.gnn_manager.poison_defense_flag = False
        self.gnn_manager.evasion_defense_flag = False
        self.gnn_manager.mi_defense_flag = False

    def return_attack_defense_flags(
            self
    ) -> None:
        self.gnn_manager.poison_attack_flag = self.start_attack_defense_flag_state["poison_attack"]
        self.gnn_manager.evasion_attack_flag = self.start_attack_defense_flag_state["evasion_attack"]
        self.gnn_manager.mi_attack_flag = self.start_attack_defense_flag_state["mi_attack"]
        self.gnn_manager.poison_defense_flag = self.start_attack_defense_flag_state["poison_defense"]
        self.gnn_manager.evasion_defense_flag = self.start_attack_defense_flag_state["evasion_defense"]
        self.gnn_manager.mi_defense_flag = self.start_attack_defense_flag_state["mi_defense"]

    def evasion_attack_pipeline(
            self,
            metrics_attack: List,
            steps: int,
            save_model_flag: bool = True,
            mask: Union[str, List[bool], torch.Tensor] = 'test',
    ) -> dict:
        metrics_values = {}
        if self.available_attacks["evasion"]:
            self.set_clear_model()
            self.gnn_manager.modification.epochs = 0
            self.gnn_manager.gnn.reset_parameters()
            from models_builder.gnn_models import Metric
            local_gen_dataset_copy = copy.deepcopy(self.gen_dataset)
            self.gnn_manager.train_model(
                gen_dataset=local_gen_dataset_copy,
                steps=steps,
                save_model_flag=False,
                metrics=[Metric("F1", mask='train', average=None)]
            )
            y_predict_clean = self.gnn_manager.run_model(
                gen_dataset=local_gen_dataset_copy,
                mask=mask,
                out='logits',
            )

            self.gnn_manager.evasion_attack_flag = True
            self.gnn_manager.modification.epochs = 0
            self.gnn_manager.gnn.reset_parameters()
            self.gnn_manager.train_model(
                gen_dataset=local_gen_dataset_copy,
                steps=steps,
                save_model_flag=save_model_flag,
                metrics=[Metric("F1", mask='train', average=None)]
            )
            self.gnn_manager.call_evasion_attack(
                gen_dataset=local_gen_dataset_copy,
                mask=mask,
            )
            y_predict_attack = self.gnn_manager.run_model(
                gen_dataset=local_gen_dataset_copy,
                mask=mask,
                out='logits',
            )

            metrics_attack_values, _ = self.evaluate_attack_defense(
                y_predict_after_attack_only=y_predict_attack,
                y_predict_clean=y_predict_clean,
                metrics_attack=metrics_attack,
                mask=mask,
            )
            if save_model_flag:
                self.save_metrics(
                    metrics_attack_values=metrics_attack_values,
                    metrics_defense_values=None,
                )
            self.return_attack_defense_flags()

        else:
            warnings.warn(f"Evasion attack is not available. Please set evasion attack for "
                          f"gnn_model_manager use def set_evasion_attacker")

        return metrics_values

    def evasion_defense_pipeline(
            self,
            metrics_attack: List,
            metrics_defense: List,
            steps: int,
            save_model_flag: bool = True,
            mask: Union[str, List[bool], torch.Tensor] = 'test',
    ) -> dict:
        metrics_values = {}
        if self.available_attacks["evasion"] and self.available_defense["evasion"]:
            from models_builder.gnn_models import Metric
            local_gen_dataset_copy = copy.deepcopy(self.gen_dataset)
            self.set_clear_model()
            self.gnn_manager.modification.epochs = 0
            self.gnn_manager.gnn.reset_parameters()
            self.gnn_manager.train_model(
                gen_dataset=local_gen_dataset_copy,
                steps=steps,
                save_model_flag=False,
                metrics=[Metric("F1", mask='train', average=None)]
            )
            y_predict_clean = self.gnn_manager.run_model(
                gen_dataset=local_gen_dataset_copy,
                mask=mask,
                out='logits',
            )

            self.gnn_manager.evasion_defense_flag = True
            self.gnn_manager.modification.epochs = 0
            self.gnn_manager.gnn.reset_parameters()
            self.gnn_manager.train_model(
                gen_dataset=local_gen_dataset_copy,
                steps=steps,
                save_model_flag=False,
                metrics=[Metric("F1", mask='train', average=None)]
            )
            y_predict_after_defense_only = self.gnn_manager.run_model(
                gen_dataset=local_gen_dataset_copy,
                mask=mask,
                out='logits',
            )

            local_gen_dataset_copy = copy.deepcopy(self.gen_dataset)
            self.gnn_manager.evasion_defense_flag = False
            self.gnn_manager.evasion_attack_flag = True
            self.gnn_manager.modification.epochs = 0
            self.gnn_manager.gnn.reset_parameters()
            self.gnn_manager.train_model(
                gen_dataset=local_gen_dataset_copy,
                steps=steps,
                save_model_flag=False,
                metrics=[Metric("F1", mask='train', average=None)]
            )
            y_predict_after_attack_only = self.gnn_manager.run_model(
                gen_dataset=local_gen_dataset_copy,
                mask=mask,
                out='logits',
            )

            self.gnn_manager.evasion_defense_flag = True
            self.gnn_manager.modification.epochs = 0
            self.gnn_manager.gnn.reset_parameters()
            self.gnn_manager.train_model(
                gen_dataset=local_gen_dataset_copy,
                steps=steps,
                save_model_flag=save_model_flag,
                metrics=[Metric("F1", mask='train', average=None)]
            )
            y_predict_after_attack_and_defense = self.gnn_manager.run_model(
                gen_dataset=local_gen_dataset_copy,
                mask=mask,
                out='logits',
            )

            metrics_attack_values, metrics_defense_values = self.evaluate_attack_defense(
                y_predict_after_attack_only=y_predict_after_attack_only,
                y_predict_clean=y_predict_clean,
                y_predict_after_defense_only=y_predict_after_defense_only,
                y_predict_after_attack_and_defense=y_predict_after_attack_and_defense,
                metrics_attack=metrics_attack,
                metrics_defense=metrics_defense,
                mask=mask,
            )
            if save_model_flag:
                self.save_metrics(
                    metrics_attack_values=metrics_attack_values,
                    metrics_defense_values=metrics_defense_values,
                )
            self.return_attack_defense_flags()
        else:
            warnings.warn(f"Evasion attack and defense is not available. Please set evasion attack for "
                          f"gnn_model_manager use def set_evasion_attacker")

        return metrics_values

    def poison_attack_pipeline(
            self,
            metrics_attack: List,
            steps: int,
            save_model_flag: bool = True,
            mask: Union[str, List[bool], torch.Tensor] = 'test',
    ) -> dict:
        metrics_values = {}
        if self.available_attacks["poison"]:
            self.set_clear_model()
            self.gnn_manager.modification.epochs = 0
            self.gnn_manager.gnn.reset_parameters()
            from models_builder.gnn_models import Metric
            local_gen_dataset_copy = copy.deepcopy(self.gen_dataset)
            self.gnn_manager.train_model(
                gen_dataset=local_gen_dataset_copy,
                steps=steps,
                save_model_flag=False,
                metrics=[Metric("F1", mask='train', average=None)]
            )
            y_predict_clean = self.gnn_manager.run_model(
                gen_dataset=local_gen_dataset_copy,
                mask=mask,
                out='logits',
            )

            self.gnn_manager.poison_attack_flag = True
            self.gnn_manager.modification.epochs = 0
            self.gnn_manager.gnn.reset_parameters()
            self.gnn_manager.train_model(
                gen_dataset=local_gen_dataset_copy,
                steps=steps,
                save_model_flag=save_model_flag,
                metrics=[Metric("F1", mask='train', average=None)]
            )
            y_predict_attack = self.gnn_manager.run_model(
                gen_dataset=local_gen_dataset_copy,
                mask=mask,
                out='logits',
            )

            metrics_attack_values, _ = self.evaluate_attack_defense(
                y_predict_after_attack_only=y_predict_attack,
                y_predict_clean=y_predict_clean,
                metrics_attack=metrics_attack,
                mask=mask,
            )
            if save_model_flag:
                self.save_metrics(
                    metrics_attack_values=metrics_attack_values,
                    metrics_defense_values=None,
                )
            self.return_attack_defense_flags()
        else:
            warnings.warn(f"Poison attack is not available. Please set evasion attack for "
                          f"gnn_model_manager use def set_evasion_attacker")

        return metrics_values

    def poison_defense_pipeline(
            self,
            metrics_attack: List,
            metrics_defense: List,
            steps: int,
            save_model_flag: bool = True,
            mask: Union[str, List[bool], torch.Tensor] = 'test',
    ) -> dict:
        metrics_values = {}
        if self.available_attacks["poison"] and self.available_defense["poison"]:
            from models_builder.gnn_models import Metric
            local_gen_dataset_copy = copy.deepcopy(self.gen_dataset)
            self.set_clear_model()
            self.gnn_manager.modification.epochs = 0
            self.gnn_manager.gnn.reset_parameters()
            self.gnn_manager.train_model(
                gen_dataset=local_gen_dataset_copy,
                steps=steps,
                save_model_flag=False,
                metrics=[Metric("F1", mask='train', average=None)]
            )
            y_predict_clean = self.gnn_manager.run_model(
                gen_dataset=local_gen_dataset_copy,
                mask=mask,
                out='logits',
            )

            self.gnn_manager.poison_defense_flag = True
            self.gnn_manager.modification.epochs = 0
            self.gnn_manager.gnn.reset_parameters()
            self.gnn_manager.train_model(
                gen_dataset=local_gen_dataset_copy,
                steps=steps,
                save_model_flag=False,
                metrics=[Metric("F1", mask='train', average=None)]
            )
            y_predict_after_defense_only = self.gnn_manager.run_model(
                gen_dataset=local_gen_dataset_copy,
                mask=mask,
                out='logits',
            )

            local_gen_dataset_copy = copy.deepcopy(self.gen_dataset)
            self.gnn_manager.poison_defense_flag = False
            self.gnn_manager.poison_attack_flag = True
            self.gnn_manager.modification.epochs = 0
            self.gnn_manager.gnn.reset_parameters()
            self.gnn_manager.train_model(
                gen_dataset=local_gen_dataset_copy,
                steps=steps,
                save_model_flag=False,
                metrics=[Metric("F1", mask='train', average=None)]
            )
            y_predict_after_attack_only = self.gnn_manager.run_model(
                gen_dataset=local_gen_dataset_copy,
                mask=mask,
                out='logits',
            )

            self.gnn_manager.poison_defense_flag = True
            self.gnn_manager.modification.epochs = 0
            self.gnn_manager.gnn.reset_parameters()
            self.gnn_manager.train_model(
                gen_dataset=local_gen_dataset_copy,
                steps=steps,
                save_model_flag=save_model_flag,
                metrics=[Metric("F1", mask='train', average=None)]
            )
            y_predict_after_attack_and_defense = self.gnn_manager.run_model(
                gen_dataset=local_gen_dataset_copy,
                mask=mask,
                out='logits',
            )

            metrics_attack_values, metrics_defense_values = self.evaluate_attack_defense(
                y_predict_after_attack_only=y_predict_after_attack_only,
                y_predict_clean=y_predict_clean,
                y_predict_after_defense_only=y_predict_after_defense_only,
                y_predict_after_attack_and_defense=y_predict_after_attack_and_defense,
                metrics_attack=metrics_attack,
                metrics_defense=metrics_defense,
                mask=mask,
            )
            if save_model_flag:
                self.save_metrics(
                    metrics_attack_values=metrics_attack_values,
                    metrics_defense_values=metrics_defense_values,
                )
            self.return_attack_defense_flags()
        else:
            warnings.warn(f"Poison attack and defense is not available. Please set evasion attack for "
                          f"gnn_model_manager use def set_evasion_attacker")

        return metrics_values

    def save_metrics(
            self,
            metrics_attack_values: Union[dict, None] = None,
            metrics_defense_values: Union[dict, None] = None,
    ):
        attack_metrics_file_name = 'attack_metrics.txt'
        defense_metrics_file_name = 'defense_metrics.txt'
        model_path_info = self.gnn_manager.model_path_info()

        if metrics_attack_values is not None:
            self.update_dictionary_in_file(
                file_path=model_path_info / attack_metrics_file_name,
                new_dict=metrics_attack_values
            )

        if metrics_defense_values is not None:
            self.update_dictionary_in_file(
                file_path=model_path_info / defense_metrics_file_name,
                new_dict=metrics_defense_values
            )

    def evaluate_attack_defense(
            self,
            y_predict_clean: Union[List, torch.Tensor, np.array],
            mask: Union[str, torch.Tensor],
            y_predict_after_attack_only: Union[List, torch.Tensor, np.array, None] = None,
            y_predict_after_defense_only: Union[List, torch.Tensor, np.array, None] = None,
            y_predict_after_attack_and_defense: Union[List, torch.Tensor, np.array, None] = None,
            metrics_attack: Union[List, None] = None,
            metrics_defense: Union[List, None] = None,
    ):

        try:
            mask_tensor = {
                'train': self.gen_dataset.train_mask.tolist(),
                'val': self.gen_dataset.val_mask.tolist(),
                'test': self.gen_dataset.test_mask.tolist(),
                'all': [True] * len(self.gen_dataset.labels),
            }[mask]
        except KeyError:
            assert isinstance(mask, torch.Tensor)
            mask_tensor = mask
        y_true = copy.deepcopy(self.gen_dataset.labels[mask_tensor])
        metrics_attack_values = {mask: {}}
        metrics_defense_values = {mask: {}}
        if metrics_attack is not None and y_predict_after_attack_only is not None:
            for metric in metrics_attack:
                metrics_attack_values[mask][metric.name] = metric.compute(
                    y_predict_clean=y_predict_clean,
                    y_predict_after_attack_only=y_predict_after_attack_only,
                    y_true=y_true,
                )
        if (
                metrics_defense is not None
                and y_predict_after_defense_only is not None
                and y_predict_after_attack_and_defense is not None
        ):
            for metric in metrics_defense:
                metrics_defense_values[mask][metric.name] = metric.compute(
                    y_predict_clean=y_predict_clean,
                    y_predict_after_attack_only=y_predict_after_attack_only,
                    y_predict_after_defense_only=y_predict_after_defense_only,
                    y_predict_after_attack_and_defense=y_predict_after_attack_and_defense,
                    y_true=y_true,
                )
        return metrics_attack_values, metrics_defense_values

    @staticmethod
    def update_dictionary_in_file(
            file_path: Union[str, Path],
            new_dict: dict
    ) -> None:
        def tensor_to_str(key):
            if isinstance(key, torch.Tensor):
                return key.tolist()
            return key

        def str_to_tensor(key):
            if isinstance(key, list):
                return torch.tensor(key, dtype=torch.bool)
            return key

        def prepare_dict_for_json(d):
            return {tensor_to_str(k): v for k, v in d.items()}

        def restore_dict_from_json(d):
            return {str_to_tensor(k): v for k, v in d.items()}

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                file_dict = restore_dict_from_json(json.load(f))
        else:
            file_dict = {}

        for mask, metrics in new_dict.items():
            for metric, value in metrics.items():
                if mask not in file_dict:
                    file_dict[mask] = {}
                file_dict[mask][metric] = value

        with open(file_path, "w") as f:
            print(json.dumps(prepare_dict_for_json(file_dict), indent=2))
            json.dump(prepare_dict_for_json(file_dict), f, indent=2)
