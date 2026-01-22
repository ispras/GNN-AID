import json
from copy import deepcopy

import torch

from attacks.mi_attacks import MIAttacker
from aux.data_info import DataInfo
from data_structures.configs import ConfigPattern, PoisonAttackConfig, PoisonDefenseConfig, EvasionAttackConfig, \
    EvasionDefenseConfig, MIAttackConfig, MIDefenseConfig
from aux.utils import POISON_ATTACK_PARAMETERS_PATH, POISON_DEFENSE_PARAMETERS_PATH, \
    EVASION_ATTACK_PARAMETERS_PATH, EVASION_DEFENSE_PARAMETERS_PATH, MI_ATTACK_PARAMETERS_PATH, \
    MI_DEFENSE_PARAMETERS_PATH
from datasets.gen_dataset import GeneralDataset
from models_builder.attack_defense_manager import FrameworkAttackDefenseManager
from models_builder.gnn_models import GNNModelManager, Metric
from web_interface.back_front.block import Block
from web_interface.back_front.utils import WebInterfaceError

NAME_TO_PATH = {
    "AD-pa": POISON_ATTACK_PARAMETERS_PATH,
    "AD-pd": POISON_DEFENSE_PARAMETERS_PATH,
    "AD-ea": EVASION_ATTACK_PARAMETERS_PATH,
    "AD-ed": EVASION_DEFENSE_PARAMETERS_PATH,
    "AD-ma": MI_ATTACK_PARAMETERS_PATH,
    "AD-md": MI_DEFENSE_PARAMETERS_PATH,
}

NAME_TO_CLASS = {
    "AD-pa": PoisonAttackConfig.__name__,
    "AD-pd": PoisonDefenseConfig.__name__,
    "AD-ea": EvasionAttackConfig.__name__,
    "AD-ed": EvasionDefenseConfig.__name__,
    "AD-ma": MIAttackConfig.__name__,
    "AD-md": MIDefenseConfig.__name__,
}


class BeforeTrainBlock(Block):
    """
    When model manager is defined, we can specify 1 attack and 3 defenses:
    poison attack, poison defense, evasion defense, MI defense.
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.gen_dataset: GeneralDataset = None
        self.model_manager: GNNModelManager = None

        self.ad_configs = {
            "AD-pa": None,
            "AD-pd": None,
            "AD-ed": None,
            "AD-md": None,
        }

    def _init(
            self,
            gen_dataset: GeneralDataset,
            gmm: GNNModelManager
    ) -> dict:
        self.gen_dataset = gen_dataset
        self.model_manager = gmm

        return FrameworkAttackDefenseManager.available_ad_methods(
            self.gen_dataset, self.model_manager)

    def _finalize(
            self
    ) -> bool:
        print(self._config)
        for name, config in self._config.items():
            # FIXME check config

            # Check for inner configs
            for k, v in config["_config_kwargs"].items():
                if isinstance(v, dict) and "_class_name" in v and "_config_kwargs" in v:
                    type = v["params_type"]
                    v = ConfigPattern(
                        _class_name=v["_class_name"],
                        _config_kwargs=v["_config_kwargs"],
                        _import_path=NAME_TO_PATH[type],
                        _config_class=NAME_TO_CLASS[type],
                    )
                    config["_config_kwargs"][k] = v

            self.ad_configs[name] = ConfigPattern(
                **config,
                _import_path=NAME_TO_PATH[name],
                _config_class=NAME_TO_CLASS[name])
        return True

    def _submit(
            self
    ) -> None:
        if self.ad_configs["AD-pa"]:
            self.model_manager.set_poison_attacker(self.ad_configs["AD-pa"])
        if self.ad_configs["AD-pd"]:
            self.model_manager.set_poison_defender(self.ad_configs["AD-pd"])
        if self.ad_configs["AD-ed"]:
            self.model_manager.set_evasion_defender(self.ad_configs["AD-ed"])
        if self.ad_configs["AD-md"]:
            self.model_manager.set_mi_defender(self.ad_configs["AD-md"])

        self._object = self.model_manager

    def _unlock(
            self
    ) -> None:
        # Retract training changes - reset model weights
        self.model_manager.gnn.reset_parameters()
        self.model_manager.modification.epochs = 0
        self.model_manager.compute_stats_data(self.gen_dataset, predictions=True, logits=True)

        stats_data = {k: self.gen_dataset.visible_part.filter(v)
                      for k, v in self.model_manager.stats_data.items()}
        self.model_manager.send_epoch_results(
            stats_data=stats_data, socket=self.socket)


class AfterTrainBlock(Block):
    """
    When model training is over, we can specify 2 attacks:
    evasion attack, MI attack.
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.gen_dataset: GeneralDataset = None
        self.model_manager: GNNModelManager = None
        self.metrics: list = None

        self.ad_configs = {
            "AD-ea": None,
            "AD-ma": None,
        }

        # Copy of the dataset before attacks applied.
        self._gen_dataset_backup: GeneralDataset = None

    def _init(
            self,
            gen_dataset: GeneralDataset,
            gmm_and_metrics: list
    ) -> dict:
        self.gen_dataset = gen_dataset
        self.model_manager, self.metrics = gmm_and_metrics

        return FrameworkAttackDefenseManager.available_ad_methods(
            self.gen_dataset, self.model_manager)

    def _finalize(
            self
    ) -> bool:
        # Make a dataset backup
        if self._gen_dataset_backup is None:
            # Make a dataset backup
            # FIXME This is a bad way - for large datasets very bad. It is a temporary solution
            self._gen_dataset_backup = deepcopy(self.gen_dataset)
        else:
            # Restore dataset
            self._restore_dataset()
        return True

    def _clear_configs(
            self
    ) -> None:
        self.ad_configs = {
            "AD-ea": None,
            "AD-ma": None,
        }

    def do(
            self,
            do,
            params
    ) -> str:
        if do == "run with attacks":
            # Effect of pressing 'accept'
            self._finalize()
            self._is_set = True  # to make diagram call unlock() when we break this block

            self._clear_configs()
            for name, config in json.loads(params.get('configs')).items():
                # FIXME check config
                self.ad_configs[name] = ConfigPattern(
                    **config,
                    _import_path=NAME_TO_PATH[name],
                    _config_class=NAME_TO_CLASS[name])

            self.model_manager.set_evasion_attacker(self.ad_configs["AD-ea"])
            self.model_manager.set_mi_attacker(self.ad_configs["AD-ma"])

            # Apply evasion attack
            if self.ad_configs["AD-ea"] is not None:
                # attack_mask =

                self.model_manager.call_evasion_attack(
                    gen_dataset=self.gen_dataset,
                    mask=torch.empty(1),
                )

            # Evaluate metrics
            metrics_values = {}
            res = self.model_manager.evaluate_model(
                gen_dataset=self.gen_dataset, metrics=self.metrics)
            if self.ad_configs["AD-ea"] is not None:
                metrics_values['After evasion attack'] = res
            else:
                metrics_values = res

            # Apply MI attack and get metrics
            if self.ad_configs["AD-ma"] is not None:
                import numpy as np
                assert not self.gen_dataset.is_multi()
                target_list = np.random.choice(
                    self.gen_dataset.info.nodes[0], size=100, replace=False)
                mask_loc = Metric.create_mask_by_target_list(
                    y_true=self.gen_dataset.labels, target_list=target_list)
                # Apply MI attack on a special mask
                self.model_manager.mi_attacker.attack(
                    gen_dataset=self.gen_dataset, model=self.model_manager.gnn,
                    mask_tensor=mask_loc)
                res = self.model_manager.mi_attacker.results.get(mask_loc)
                if res is not None:
                    metrics_values['MI attack results'] = MIAttacker.compute_single_attack_accuracy(
                        mask_loc, res, self.gen_dataset.train_mask)

            # Update model logits and predictions
            self.model_manager.compute_stats_data(
                gen_dataset=self.gen_dataset, predictions=True, logits=True)
            stats_data = self.model_manager.stats_data

            # print("metrics_values after attacks", metrics_values)
            stats_data = {k: self.gen_dataset.visible_part.filter(v)
                          for k, v in stats_data.items()}

            # Update dataset features
            stats_data["node_features"] = self.gen_dataset.visible_part.get_dataset_var_data().node_features

            self.model_manager.send_epoch_results(
                metrics_values=metrics_values, stats_data=stats_data, socket=self.socket)
            return ''

        elif do == "save attack configs":
            # We want to save the given config
            self._clear_configs()
            for name, config in json.loads(params.get('configs')).items():
                # FIXME check config
                self.ad_configs[name] = ConfigPattern(
                    **config,
                    _import_path=NAME_TO_PATH[name],
                    _config_class=NAME_TO_CLASS[name])
            return self._save_attack_confgis()

        else:
            raise WebInterfaceError(f"Unknown 'do' command '{do}' for model")

    def _save_attack_confgis(
            self
    ) -> str:
        # FIXME discuss scenario with Kirill
        #  no sense to save model, only configs
        path = self.model_manager.save_model_executor()
        self.gen_dataset.save_train_test_mask(path)
        DataInfo.refresh_models_dir_structure()
        return str(path)

    def _unlock(
            self
    ) -> None:
        # Retract changes - reset dataset as before evasion attacks
        # and remove attacks from model manager
        self._restore_dataset()
        self.model_manager.set_evasion_attacker(None)
        self.model_manager.set_mi_attacker(None)

        # # Update dataset features
        # stats_data = {
        #     "node_features": self.gen_dataset.visible_part.get_dataset_var_data().node_features
        # }
        # self.socket.send(block='at', msg=stats_data, tag='node_features', obligate=True)

        # Update model logits and predictions
        self.model_manager.compute_stats_data(
            gen_dataset=self.gen_dataset, predictions=True, logits=True)
        stats_data = self.model_manager.stats_data

        # print("metrics_values after attacks", metrics_values)
        stats_data = {k: self.gen_dataset.visible_part.filter(v)
                      for k, v in stats_data.items()}

        # Update dataset features
        stats_data[
            "node_features"] = self.gen_dataset.visible_part.get_dataset_var_data().node_features

        metrics_values = self.model_manager.evaluate_model(
            gen_dataset=self.gen_dataset, metrics=self.metrics)

        self.model_manager.send_epoch_results(
            metrics_values=metrics_values, stats_data=stats_data, socket=self.socket)

    def _restore_dataset(
            self
    ) -> None:
        # FIXME This is a bad way - for large datasets very bad. It is a temporary solution
        self.gen_dataset = deepcopy(self._gen_dataset_backup)
