import json

from gnn_aid.data_structures.configs import ConfigPattern, PoisonAttackConfig, PoisonDefenseConfig, EvasionAttackConfig, \
    EvasionDefenseConfig, MIAttackConfig, MIDefenseConfig
from gnn_aid.aux.utils import POISON_ATTACK_PARAMETERS_PATH, POISON_DEFENSE_PARAMETERS_PATH, \
    EVASION_ATTACK_PARAMETERS_PATH, EVASION_DEFENSE_PARAMETERS_PATH, MI_ATTACK_PARAMETERS_PATH, \
    MI_DEFENSE_PARAMETERS_PATH
from gnn_aid.datasets.gen_dataset import GeneralDataset
from gnn_aid.models_builder.attack_defense_manager import FrameworkAttackDefenseManager
from gnn_aid.models_builder.gnn_models import GNNModelManager
from .block import Block
from .utils import WebInterfaceError

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

        self.gen_dataset = None
        self.model_manager: GNNModelManager = None
        self.metrics: list = None

        self.ad_configs = {
            "AD-ea": None,
            "AD-ma": None,
        }

    def _init(
            self,
            gen_dataset: GeneralDataset,
            gmm_and_metrics: list
    ) -> dict:
        self.gen_dataset = gen_dataset
        self.model_manager, self.metrics = gmm_and_metrics

        return FrameworkAttackDefenseManager.available_ad_methods(
            self.gen_dataset, self.model_manager)

    def do(
            self,
            do,
            params
    ) -> str:
        if do == "run with attacks":
            for name, config in json.loads(params.get('configs')).items():
                # FIXME check config
                self.ad_configs[name] = ConfigPattern(
                    **config,
                    _import_path=NAME_TO_PATH[name],
                    _config_class=NAME_TO_CLASS[name])

            if self.ad_configs["AD-ea"]:
                self.model_manager.set_evasion_attacker(self.ad_configs["AD-ea"])
            if self.ad_configs["AD-ma"]:
                self.model_manager.set_mi_attacker(self.ad_configs["AD-ma"])

            metrics_values = self.model_manager.evaluate_model(
                self.gen_dataset, metrics=self.metrics)
            self.model_manager.compute_stats_data(self.gen_dataset, predictions=True, logits=True)

            # print("metrics_values after attacks", metrics_values)
            stats_data = {k: self.gen_dataset.visible_part.filter(v)
                          for k, v in self.model_manager.stats_data.items()}
            self.model_manager.send_epoch_results(
                metrics_values=metrics_values, stats_data=stats_data, socket=self.socket)
            return ''

        else:
            raise WebInterfaceError(f"Unknown 'do' command '{do}' for model")

