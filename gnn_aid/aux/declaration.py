import json
import os
from pathlib import Path
from typing import Union, Tuple, Any

from .utils import MODELS_DIR, GRAPHS_DIR, EXPLANATIONS_DIR, hash_data_sha256, \
    SAVE_DIR_STRUCTURE_PATH, DATASETS_DIR


class Declare:
    """
    Forms a filesystem path for accessing, saving, and loading all key objects
    (data, model, explainer).
    """

    @staticmethod
    def obj_info_to_path(
            what_save: str = None,
            previous_path: Union[str, Path] = None,
            obj_info: Union[None, list, tuple, dict, Path] = None
    ) -> Tuple[str | Path, list[Path]]:
        """
        Build a filesystem path for accessing, saving, or loading a key object.

        Args:
            what_save (str): Object type to build the path for.
                Supported: ``'data_root'``, ``'datasets'``, ``'models'``, ``'explanations'``.
            previous_path (Union[str, Path]): Base path to extend with the directory structure.
            obj_info (Union[None, list, tuple, dict, Path]): Object identifiers that must match
                the keys in ``save_dir_structure.json`` (order matters for list/tuple).

        Returns:
            Tuple[str | Path, list[Path]]: Extended path and list of associated technical file paths.
        """
        if obj_info is None:
            obj_info = []
        with open(SAVE_DIR_STRUCTURE_PATH) as f:
            save_dir_structure = json.loads(f.read())
        if what_save is None:
            raise Exception(f"what_save is None, select one of the keys save_dir_structure.json: "
                            f"{save_dir_structure.keys()}")
        save_dir_structure = save_dir_structure[what_save]
        if previous_path is None:
            raise Exception(f"previous_path is None, but it can't be None")
        path = previous_path
        empty_dir_val_shift = 0

        files_paths = []

        correct_len_obj_info = len(
            list(filter(lambda y: y["add_key_name_flag"] is not None, save_dir_structure.values())))
        if isinstance(obj_info, (list, tuple)):
            if len(obj_info) != correct_len_obj_info:
                raise Exception(f"obj_info len don't what_save modes keys from save_dir_structure.json")
            for i, (key, val) in enumerate(save_dir_structure.items()):
                if val["add_key_name_flag"] is None:
                    empty_dir_val_shift += 1
                    loc_dir_name = key
                else:
                    if val["add_key_name_flag"]:
                        loc_dir_name = key + "=" + obj_info[i - empty_dir_val_shift]
                    else:
                        loc_dir_name = obj_info[i - empty_dir_val_shift]
                if val["files_info"] is not None:
                    for file_info in val["files_info"]:
                        if file_info["file_name"] == "origin":
                            loc_file_name = loc_dir_name
                        else:
                            loc_file_name = file_info["file_name"]
                        if file_info["format"] is not None:
                            loc_file_name += file_info["format"]
                        files_paths.append(path / Path(loc_file_name))
                path /= loc_dir_name
        elif isinstance(obj_info, dict):
            if len(obj_info.keys()) != correct_len_obj_info:
                raise Exception(f"obj_info len don't what_save modes keys from save_dir_structure.json")
            for key, val in save_dir_structure.items():
                if val["add_key_name_flag"] is None:
                    loc_dir_name = key
                else:
                    if val["add_key_name_flag"]:
                        loc_dir_name = key + "=" + obj_info[key]
                    else:
                        loc_dir_name = obj_info[key]
                if val["files_info"] is not None:
                    for file_info in val["files_info"]:
                        if file_info["file_name"] == "origin":
                            loc_file_name = loc_dir_name
                        else:
                            loc_file_name = file_info["file_name"]
                        if file_info["format"] is not None:
                            loc_file_name += file_info["format"]
                        files_paths.append(path / Path(loc_file_name))
                path /= loc_dir_name
        else:
            raise Exception("obj_info must be dict, tuple or list")
        return path, files_paths

    @staticmethod
    def dataset_root_dir(
            dataset_config: 'DatasetConfig'
    ) -> Tuple[str | Path, list[Path]]:
        """
        Directory where dataset raw files and metainfo are stored.

        Args:
            dataset_config (DatasetConfig): Dataset configuration object.

        Returns:
            Path to the data folder and list of technical file paths.
        """
        path = GRAPHS_DIR
        obj_info = [
            dataset_config.path(),
        ]
        path, files_paths = Declare.obj_info_to_path(previous_path=path, what_save="data_root",
                                                     obj_info=obj_info)
        return path, files_paths

    @staticmethod
    def dataset_info_path(
            dataset_config: 'DatasetConfig'
    ) -> Path:
        return Declare.dataset_root_dir(dataset_config)[0] / 'metainfo'

    @staticmethod
    def dataset_prepared_dir(
            dataset_config: Union['ConfigPattern', 'DatasetConfig'],
            dataset_var_config: Union['ConfigPattern', 'DatasetVarConfig']
    ) -> Tuple[str | Path, list[Path]]:
        """
        Directory where the var part of a dataset is stored.

        Args:
            dataset_config (ConfigPattern | DatasetConfig): Dataset configuration object.
            dataset_var_config (ConfigPattern | DatasetVarConfig): Dataset variant configuration object.

        Returns:
            Path to the data folder and extra paths for saving dataset_config and dataset_var_config.
        """
        assert dataset_var_config.features is not None

        path = DATASETS_DIR

        obj_info = [
            dataset_config.hash_for_config(),
            dataset_var_config.hash_for_config(),
        ]

        # Find minimal free version if not specified
        if dataset_var_config["dataset_ver_ind"] is None:
            ix = 0
            while True:
                dataset_var_config["dataset_ver_ind"] = ix  # FIXME Kirill 'DatasetVarConfig' object does not support item assignment
                loc_path, files_paths = Declare.obj_info_to_path(what_save="datasets", previous_path=path,
                                                                 obj_info=obj_info)
                if not loc_path.exists():  # if name exists, adding number to it
                    break
                ix += 1
            path = loc_path
        else:
            path, files_paths = Declare.obj_info_to_path(what_save="datasets", previous_path=path,
                                                         obj_info=obj_info)
        return path, files_paths

    @staticmethod
    def models_path(
            class_obj: 'GNNModelManager'
    ) -> Tuple[str | Path, list[Path]]:
        """
        Build the path where the model will be saved.

        If model_ver_ind is not defined, saves with the smallest free integer index
        starting from 0. If defined, the first save uses that version; subsequent ones
        are determined automatically.

        Args:
            class_obj (GNNModelManager): Model manager instance.

        Returns:
            Path where the model is saved and list of technical file paths.
        """
        model_ver_ind_none_flag = \
            class_obj.modification.model_ver_ind is None or \
            class_obj.modification.data_change_flag()
        path = Path(str(class_obj.dataset_path).replace(str(DATASETS_DIR), str(MODELS_DIR)))
        what_save = "models"

        mi_defense_kwargs_hash = class_obj.mi_defense_config.hash_for_config()
        evasion_defense_kwargs_hash = class_obj.evasion_defense_config.hash_for_config()
        poison_defense_kwargs_hash = class_obj.poison_defense_config.hash_for_config()
        mi_attack_kwargs_hash = class_obj.mi_attack_config.hash_for_config()
        evasion_attack_kwargs_hash = class_obj.evasion_attack_config.hash_for_config()
        poison_attack_kwargs_hash = class_obj.poison_attack_config.hash_for_config()

        obj_info = [
            class_obj.gnn.get_hash(), class_obj.get_hash(),
            poison_attack_kwargs_hash, poison_defense_kwargs_hash,
            mi_defense_kwargs_hash, evasion_defense_kwargs_hash,
            evasion_attack_kwargs_hash, mi_attack_kwargs_hash,
            *class_obj.modification.to_savable_dict(compact=True, need_full=False).values()
        ]
        # print(class_obj.modification.to_saveable_dict(compact=True, need_full=False))

        # QUE Kirill, maybe we can make it better
        if model_ver_ind_none_flag:
            ix = 0
            while True:
                obj_info[-1] = str(ix)
                loc_path, files_paths = Declare.obj_info_to_path(what_save=what_save, previous_path=path,
                                                                 obj_info=obj_info)
                if not loc_path.exists():  # if name exists, adding number to it
                    break
                ix += 1
            path = loc_path
            class_obj.modification.model_ver_ind = ix
            class_obj.modification.data_change_flag()
        else:
            path, files_paths = Declare.obj_info_to_path(what_save=what_save, previous_path=path,
                                                         obj_info=obj_info)
        return path, files_paths

    @staticmethod
    def declare_model_by_config(
            dataset_path: str,
            GNNModelManager_hash: str,
            model_ver_ind: int,
            gnn_name: str,
            mi_defense_hash: str,
            evasion_defense_hash: str,
            poison_defense_hash: str,
            mi_attack_hash: str,
            evasion_attack_hash: str,
            poison_attack_hash: str,
            epochs: Union[int, str] = None
    ) -> Tuple[str | Path, list[Path]]:
        """
        Build the model save path from its hyperparameters and features.

        Args:
            dataset_path (str): Dataset path.
            GNNModelManager_hash (str): GNN model manager hash.
            model_ver_ind (int): Model version index.
            gnn_name (str): GNN hash.
            mi_defense_hash (str): MI defense hash.
            evasion_defense_hash (str): Evasion defense hash.
            poison_defense_hash (str): Poison defense hash.
            mi_attack_hash (str): MI attack hash.
            evasion_attack_hash (str): Evasion attack hash.
            poison_attack_hash (str): Poison attack hash.
            epochs (Union[int, str]): Number of training epochs. Default value: `None`.

        Returns:
            Path where the model is saved and list of technical file paths.
        """
        if not isinstance(model_ver_ind, int) or model_ver_ind < 0:
            raise Exception("model_ver_ind must be int type and has value >= 0")
        path = Path(str(dataset_path).replace(str(DATASETS_DIR), str(MODELS_DIR)))
        what_save = "models"
        obj_info = {
            "gnn": gnn_name,
            "gnn_model_manager": GNNModelManager_hash,
            "model_ver_ind": str(model_ver_ind),
            "poison_attacker": str(poison_attack_hash),
            "poison_defender": str(poison_defense_hash),
            "evasion_defender": str(evasion_defense_hash),
            "mi_defender": str(mi_defense_hash),
            "evasion_attacker": str(evasion_attack_hash),
            "mi_attacker": str(mi_attack_hash),
            "epochs": str(epochs),
        }

        path, files_paths = Declare.obj_info_to_path(previous_path=path, what_save=what_save,
                                                     obj_info=obj_info)
        return path, files_paths

    @staticmethod
    def explanation_file_path(
            models_path: str,
            explainer_name: str,
            explainer_ver_ind: int = None,
            explainer_run_kwargs: dict = None,
            explainer_init_kwargs: dict = None,
            create_dir_flag: bool = True
    ) -> Tuple[str | Path, list[Path]]:
        """
        Build the file path for an explanation result.

        Args:
            models_path (str): Model path.
            explainer_name (str): Explainer class name, e.g. ``'Zorro'``.
            explainer_ver_ind (int): Explanation version index. Default value: `None`.
            explainer_run_kwargs (dict): Kwargs passed to the explainer's run method. Default value: `None`.
            explainer_init_kwargs (dict): Kwargs passed to the explainer's constructor. Default value: `None`.
            create_dir_flag (bool): If True, create the directory and write kwargs files. Default value: `True`.

        Returns:
            Path for the explanation result file and list of technical file paths.
        """
        explainer_init_kwargs = explainer_init_kwargs.copy()
        explainer_init_kwargs = dict(sorted(explainer_init_kwargs.items()))
        json_init_object = json.dumps(explainer_init_kwargs)
        explainer_init_kwargs_hash = hash_data_sha256(json_init_object.encode('utf-8'))

        explainer_run_kwargs = explainer_run_kwargs.copy()
        explainer_run_kwargs = dict(sorted(explainer_run_kwargs.items()))
        json_run_object = json.dumps(explainer_run_kwargs)
        explainer_run_kwargs_hash = hash_data_sha256(json_run_object.encode('utf-8'))

        path = Path(str(models_path).replace(str(MODELS_DIR), str(EXPLANATIONS_DIR)))
        what_save = "explanations"
        obj_info = {
            "explainer_name": explainer_name,
            "explainer_init_kwargs": explainer_init_kwargs_hash,
            "explainer_run_kwargs": explainer_run_kwargs_hash,
            "explainer_ver_ind": str(explainer_ver_ind),
        }

        # QUE Kirill, maybe we can make it better
        if explainer_ver_ind is None:
            ix = 0
            while True:
                obj_info["explainer_ver_ind"] = str(ix)
                loc_path, files_paths = Declare.obj_info_to_path(
                    what_save=what_save,
                    previous_path=path,
                    obj_info=obj_info
                )
                if not loc_path.exists():  # if name exists, adding number to it
                    break
                ix += 1
            path = loc_path
        else:
            path, files_paths = Declare.obj_info_to_path(
                what_save=what_save,
                previous_path=path,
                obj_info=obj_info
            )

        if create_dir_flag:
            if not os.path.exists(path):
                os.makedirs(path)
            path = path / Path('explanation.json')
            with open(files_paths[0], "w") as f:
                json.dump(explainer_init_kwargs, f, indent=2)
            with open(files_paths[1], "w") as f:
                json.dump(explainer_run_kwargs, f, indent=2)

        return path, files_paths

    @staticmethod
    def explainer_kwargs_path_full(
            model_path: Union[str, Path],
            explainer_path: Union[str, Path]
    ) -> list[Path]:
        """
        Return the list of technical files associated with an explanation.

        Args:
            model_path (Union[str, Path]): Model path.
            explainer_path (Union[str, Path]): Explanation path.

        Returns:
            Technical file paths (JSON files with init and run kwargs).
        """
        path = Path(str(model_path).replace(str(MODELS_DIR), str(EXPLANATIONS_DIR)))
        what_save = "explanations"
        # BUG Misha, check is correct next line, because in def obj_info_to_path can't be Path or str
        obj_info = explainer_path

        _, files_paths = Declare.obj_info_to_path(
            what_save=what_save,
            previous_path=path,
            obj_info=obj_info
        )

        return files_paths
