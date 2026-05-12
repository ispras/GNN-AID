import collections
import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import List, Tuple, Union, Dict, Mapping

from .prefix_storage import FixedKeysPrefixStorage, TuplePrefixStorage
from gnn_aid.aux.utils import MODELS_DIR, GRAPHS_DIR, EXPLANATIONS_DIR, DATA_INFO_DIR, \
    USER_MODELS_DIR, SAVE_DIR_STRUCTURE_PATH, DATASETS_DIR, root_dir


class DataInfo:
    """
    Responsible for populating prefix access trees to datasets, models, and explanations
    based on the directory structure.
    """

    @staticmethod
    def refresh_all_data_info() -> None:
        """
        Refresh all data info files for saved objects (datasets, models, explanations).
        """
        DATA_INFO_DIR.mkdir(exist_ok=True, parents=True)
        DataInfo.refresh_data_dir_structure()
        DataInfo.refresh_data_var_dir_structure()
        DataInfo.refresh_models_dir_structure()
        DataInfo.refresh_explanations_dir_structure()

    @staticmethod
    def refresh_data_dir_structure() -> None:
        """
        Update the file with information about saved raw datasets.
        """
        DATA_INFO_DIR_data = DATA_INFO_DIR / 'data_dir_structure'
        with open(DATA_INFO_DIR_data, 'w', encoding='utf-8') as f:
            # IMP suggest create a constant for "dataset_ver_ind" and such strings, same for next 2 functions
            prev_path = ''
            for path in Path(GRAPHS_DIR).glob('**/metainfo'):
                path = path.parent
                if prev_path != path and (path / 'raw').exists():
                    f.write(os.path.relpath(path, GRAPHS_DIR) + '\n')
                    prev_path = path

    @staticmethod
    def refresh_models_dir_structure() -> None:
        """
        Update the file with information about saved models.
        """
        DATA_INFO_DIR_models = DATA_INFO_DIR / 'models_dir_structure'
        with open(DATA_INFO_DIR_models, 'w', encoding='utf-8') as f:
            for path in Path(MODELS_DIR).glob('**/model'):
                path = path.parts[len(root_dir.parts) + 1:-1]
                f.write(str(Path(*path)) + '\n')

    @staticmethod
    def refresh_explanations_dir_structure() -> None:
        """
        Update the file with information about saved explanations.
        """
        DATA_INFO_DIR_results = DATA_INFO_DIR / 'explanations_dir_structure'
        with open(DATA_INFO_DIR_results, 'w', encoding='utf-8') as f:
            for path in Path(EXPLANATIONS_DIR).glob('**/explanation.json'):
                path = path.parts[len(root_dir.parts) + 1:-1]
                f.write(str(Path(*path)) + '\n')

    @staticmethod
    def refresh_data_var_dir_structure() -> None:
        """
        Update the file with information about saved prepared datasets.
        """
        DATA_INFO_DIR_results = DATA_INFO_DIR / 'data_var_dir_structure'
        with open(DATA_INFO_DIR_results, 'w', encoding='utf-8') as f:
            for path in Path(DATASETS_DIR).glob('**/data.pt'):
                path = path.parts[len(root_dir.parts) + 1:-1]
                f.write(str(Path(*path)) + '\n')

    @staticmethod
    def take_keys_etc_by_prefix(
            prefix: Tuple
    ) -> Tuple[List, List, dict, int]:
        """
        Extract keys and directory structure info for the given save prefix.

        Args:
            prefix (Tuple): Object types in the order used to form the save path.
                Example: ``("datasets", "models", "explanations")``.

        Returns:
            Tuple of (keys_list, full_keys_list, dir_structure, empty_dir_shift).
            keys_list — meaningful keys; full_keys_list — all keys including technical;
            dir_structure — relevant portion of save_dir_structure.json;
            empty_dir_shift — count of technical (unnamed) directory levels.
        """
        with open(SAVE_DIR_STRUCTURE_PATH) as f:
            save_dir_structure = json.loads(f.read())
        keys_list = []
        full_keys_list = []
        empty_dir_shift = 0
        dir_structure = {}
        for elem in prefix:
            if elem in save_dir_structure.keys():
                dir_structure.update(save_dir_structure[elem])
                for key, val in save_dir_structure[elem].items():
                    full_keys_list.append(key)
                    if val["add_key_name_flag"] is not None:
                        keys_list.append(key)
                    else:
                        empty_dir_shift += 1
            else:
                raise Exception(f"Key {elem} doesn't in save_dir_structure.keys()")
        return keys_list, full_keys_list, dir_structure, empty_dir_shift

    @staticmethod
    def values_list_by_path_and_keys(
            path: Union[str, Path],
            full_keys_list: List,
            dir_structure: dict
    ) -> List[str]:
        """
        Extract object field values from a saved-object path using known keys.

        Args:
            path (Union[str, Path]): Path of the saved object.
            full_keys_list (List): All keys (meaningful and technical) matching path segments.
            dir_structure (dict): Directory structure dict from save_dir_structure.json.

        Returns:
            Object field values extracted from the path segments.
        """
        parts_val = []
        path = Path(path).parts
        for i, part in enumerate(path):
            if dir_structure[full_keys_list[i]]["add_key_name_flag"] is not None:
                if not dir_structure[full_keys_list[i]]["add_key_name_flag"]:
                    parts_val.append(part.strip())
                else:
                    parts_val.append(part.strip().split(f'{full_keys_list[i]}=', 1)[1])
        return parts_val

    @staticmethod
    def values_list_and_technical_files_by_path_and_prefix(
            path: Union[str, Path],
            prefix: Tuple[str, ...]
    ) -> Tuple[List[str], Dict[str, dict]]:
        """
        Extract object field values and technical file references from a saved-object path.

        Args:
            path (Union[str, Path]): Path of the saved object.
            prefix (Tuple[str, ...]): Object types in the order used to form the save path.
                Example: ``("datasets", "models", "explanations")``.

        Returns:
            List of field values extracted from the path, and dict mapping field names
            to their associated technical file paths.
        """
        with open(SAVE_DIR_STRUCTURE_PATH) as f:
            save_dir_structure = json.loads(f.read())
        parts_val = []
        description_info = {}
        path = Path(path)
        if isinstance(path, bytes):  # BUG: unreachable after Path(path); bytes check should precede conversion
            path = path.decode()
        path = path.parts
        parts_parse = 0
        for prefix_part in prefix[:-1]:
            for key, val in save_dir_structure[prefix_part].items():
                if val.get("add_key_name_flag") is not None:
                    if not val["add_key_name_flag"]:
                        parts_val.append(path[parts_parse].strip())
                    else:
                        parts_val.append(path[parts_parse].strip().split(f'{key}=', 1)[1])
                parts_parse += 1
        if len(prefix) > 0:
            last_prefix_part = prefix[-1]
            if last_prefix_part in save_dir_structure:
                for key, val in save_dir_structure[last_prefix_part].items():
                    if val.get("add_key_name_flag") is not None:
                        if not val["add_key_name_flag"]:
                            parts_val.append(path[parts_parse].strip())
                        else:
                            parts_val.append(path[parts_parse].strip().split(f'{key}=', 1)[1])
                    if val.get("files_info") is not None:
                        for file_info_dict in val["files_info"]:
                            if file_info_dict["file_name"] == "origin":
                                file_name = path[parts_parse].strip()
                                file_name += file_info_dict["format"]
                                description_info.update(
                                    {key: {parts_val[-1]: os.path.join(*path[:parts_parse], file_name)}})
                    parts_parse += 1
        return parts_val, description_info

    @staticmethod
    def fill_prefix_storage(
            prefix: Tuple,
            file_with_paths: Union[str, Path]
    ) -> Tuple[FixedKeysPrefixStorage, dict]:
        """
        Fill a FixedKeysPrefixStorage from a file containing saved-object paths.

        Args:
            prefix (Tuple): Object types in the order used to form the save path.
                Example: ``("datasets", "models", "explanations")``.
            file_with_paths (Union[str, Path]): File listing paths of saved objects, one per line.

        Returns:
            Populated FixedKeysPrefixStorage and description info dict keyed by hash.
        """
        keys_list, full_keys_list, dir_structure, empty_dir_shift = \
            DataInfo.take_keys_etc_by_prefix(prefix=prefix)
        ps = FixedKeysPrefixStorage(keys_list)
        with open(file_with_paths, 'r', encoding='utf-8') as f:
            description_info = {}
            for line in f:
                if len(ps.keys) != len(Path(line).parts) - empty_dir_shift:
                    continue
                loc_parts_values, description_info_loc = DataInfo.values_list_and_technical_files_by_path_and_prefix(
                    path=line, prefix=prefix,
                )
                ps.add(loc_parts_values, None)
                description_info = DataInfo.deep_update(description_info, description_info_loc)
        return ps, description_info

    @staticmethod
    def deep_update(
            d: dict,
            u: Union[Mapping, Dict]
    ) -> dict:
        """
        Recursively update dict d with values from u, merging nested dicts.

        Args:
            d (dict): Target dictionary to update in place.
            u (Union[Mapping, Dict]): Source of override values.

        Returns:
            Updated target dictionary.
        """
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = DataInfo.deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    @staticmethod
    def description_info_with_paths_to_description_info_with_files_values(
            description_info: dict,
            root_path: Union[str, Path]
    ) -> dict:
        """
        Replace file paths in description_info with the actual file contents.

        Args:
            description_info (dict): Mapping of field names to {hash: relative_path} dicts.
            root_path (Union[str, Path]): Root directory to resolve relative paths against.

        Returns:
            Updated description_info with file contents in place of file paths.
        """
        for description_info_key, description_info_val in description_info.items():
            for obj_name, obj_file_path in description_info_val.items():
                with open(os.path.join(root_path, obj_file_path)) as f:
                    description_info[description_info_key][obj_name] = f.read()
        return description_info

    @staticmethod
    def explainers_parse() -> Tuple[FixedKeysPrefixStorage, dict]:
        """
        Parse explainer paths from the technical file of all saved explainers.

        Returns:
            Populated FixedKeysPrefixStorage and description info dict.
        """
        DATA_INFO_DIR_results = DATA_INFO_DIR / 'explanations_dir_structure'
        ps, description_info = DataInfo.fill_prefix_storage(
            prefix=("datasets", "models", "explanations"),
            file_with_paths=DATA_INFO_DIR_results)
        description_info = DataInfo.description_info_with_paths_to_description_info_with_files_values(
            description_info=description_info, root_path=EXPLANATIONS_DIR,
        )
        return ps, description_info

    @staticmethod
    def models_parse() -> Tuple[FixedKeysPrefixStorage, dict]:
        """
        Parse model paths from the technical file of all saved models.

        Returns:
            Populated FixedKeysPrefixStorage and description info dict.
        """
        DATA_INFO_DIR_results = DATA_INFO_DIR / 'models_dir_structure'
        ps, description_info = DataInfo.fill_prefix_storage(
            prefix=("datasets", "models"),
            file_with_paths=DATA_INFO_DIR_results)
        description_info = DataInfo.description_info_with_paths_to_description_info_with_files_values(
            description_info=description_info, root_path=MODELS_DIR,
        )
        return ps, description_info

    @staticmethod
    def data_parse() -> TuplePrefixStorage:
        """
        Parse raw dataset paths from the technical file of all saved raw datasets.

        Returns:
            TuplePrefixStorage populated with dataset paths and metainfo descriptions.
        """
        DATA_INFO_DIR_results = DATA_INFO_DIR / 'data_dir_structure'
        ps = TuplePrefixStorage()
        with open(DATA_INFO_DIR_results, 'r', encoding='utf-8') as f:
            for line in f:
                description_info = "Info not available"
                path = Path(line.strip())
                try:
                    with open(GRAPHS_DIR / path / 'metainfo', 'r') as f:
                        metainfo = json.load(f)
                        description_info = f'Graphs {metainfo["count"]}\nNodes: {metainfo["nodes"]}\nDirected: {metainfo["directed"]}\nHetero: {metainfo.get("hetero", False)}'
                except FileNotFoundError:
                    pass
                ps.add(path.parts, description_info)
        return ps

    @staticmethod
    def data_var_parse() -> FixedKeysPrefixStorage:
        """
        Parse prepared dataset paths from the technical file of all saved prepared datasets.

        Returns:
            FixedKeysPrefixStorage populated with prepared dataset paths.
        """
        DATA_INFO_DIR_results = DATA_INFO_DIR / 'data_var_dir_structure'
        ps, description_info = DataInfo.fill_prefix_storage(
            prefix=("datasets",),
            file_with_paths=DATA_INFO_DIR_results)
        return ps

    @staticmethod
    def clean_prepared_data(
            dry_run: bool = False
    ) -> None:
        """
        Remove all prepared data directories for all datasets.

        Args:
            dry_run (bool): If True, only print paths without deleting. Default value: `False`.
        """
        import shutil
        for path in Path(DATASETS_DIR).glob('**/prepared'):
            print(path)
            if not dry_run:
                shutil.rmtree(path)

    @staticmethod
    def all_obj_ver_by_obj_path(
            obj_dir_path: Union[str, Path]
    ) -> set:
        """
        Return all saved version indices for the given object path.

        Args:
            obj_dir_path (Union[str, Path]): Path to the saved object directory.

        Returns:
            Set of integer version indices found in the parent directory.
        """
        obj_dir_path = Path(obj_dir_path).parent
        vers_ind = []
        for dir_path, dir_names, filenames in os.walk(obj_dir_path):
            if not dir_names and filenames:
                vers_ind.append(int(Path(dir_path).parts[-1].rsplit(sep='=', maxsplit=1)[-1]))
        return set(vers_ind)

    @staticmethod
    def del_all_empty_folders(
            dir_path: Union[str, Path]
    ) -> None:
        """
        Delete all empty folders and their associated metadata files in a directory.

        Args:
            dir_path (Union[str, Path]): Root directory to search for empty folders.
        """
        for dir_path, dir_names, filenames in os.walk(dir_path, topdown=False):
            for dir_name in dir_names:
                full_path = os.path.join(dir_path, dir_name)
                if not os.listdir(full_path):
                    for file in filenames:
                        import re
                        # QUE Kirill, maybe should make better check
                        check_math = re.search(f"{dir_name}.*", str(file))
                        if check_math is not None:
                            os.remove(os.path.join(dir_path, check_math.group(0)))
                    os.rmdir(full_path)


class UserCodeInfo:
    """
    Provides utilities for discovering and loading user-defined GNN model objects
    from the user_models_obj directory.
    """

    @staticmethod
    def user_models_list_ref() -> dict:
        """
        Scan the user_models_obj directory and return info about all user GNN model objects.

        Each file must expose a ``models_init()`` function returning a dict of
        ``GNNConstructor`` instances::

            def models_init():
                obj_1 = UserGNNClass_1(any parameters)
                obj_2 = UserGNNClass_2(any parameters)
                return locals()

        Returns:
            Dict mapping class name to ``{'obj_names': list, 'import_path': str}``.
        """
        # Hierarchy of dataset naming
        from gnn_aid.models_builder.gnn_constructor import GNNConstructor

        DATA_INFO_DIR.mkdir(exist_ok=True, parents=True)
        DATA_INFO_USER_MODELS_INFO = DATA_INFO_DIR / 'user_model_list'
        user_models_obj_dict_info = {}
        if os.path.exists(USER_MODELS_DIR):
            for path in os.scandir(USER_MODELS_DIR):
                if path.is_file():
                    user_files_path = USER_MODELS_DIR / path.name
                    file_loc_obj = {}
                    try:
                        spec = importlib.util.spec_from_file_location("models_init", user_files_path)
                        foo = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(foo)
                        file_loc_obj = foo.models_init()
                        file_loc_obj = dict(filter(lambda y: isinstance(y[1], GNNConstructor), file_loc_obj.items()))
                    except:
                        logging.warning(f"\nFile {user_files_path} \n"
                                        f"doesn't contain function models_init, write it to access model objects or \n"
                                        f"returns objects that don't inherit from GNNConstructor.\n"
                                        f"def models_init should have the following logic:\n"
                                        f"def models_init():\n"
                                        f"  obj_1 = UserGNNClass_1(any parameters)\n"
                                        f"  obj_2 = UserGNNClass_2(any parameters)\n"
                                        f"  return locals()\n"
                                        f"where UserGNNClass inherit from GNNConstructor")
                    user_files_path = str(user_files_path)
                    if file_loc_obj:
                        for key, val in file_loc_obj.items():
                            if val.__class__.__name__ in user_models_obj_dict_info:
                                user_models_obj_dict_info[val.__class__.__name__]['obj_names'].append(key)
                            else:
                                user_models_obj_dict_info[val.__class__.__name__] = {'obj_names': [key],
                                                                                     'import_path': user_files_path}
            with open(DATA_INFO_USER_MODELS_INFO, 'w', encoding='utf-8') as f:
                f.write(json.dumps(user_models_obj_dict_info, indent=2))

        return user_models_obj_dict_info

    @staticmethod
    def take_user_model_obj(
            user_file_path: Union[str, Path],
            obj_name: str
    ) -> 'GNNConstructor':
        """
        Load and return a user model object by name from a user file.

        Args:
            user_file_path (Union[str, Path]): Path to the user file containing model objects.
            obj_name (str): Name of the object to retrieve from models_init().

        Returns:
            The requested GNNConstructor instance with obj_name attribute set.

        Raises:
            Exception: If the file does not exist, does not define models_init, or
                does not contain an object named obj_name.
        """
        try:
            spec = importlib.util.spec_from_file_location("models_init", user_file_path)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
            file_loc_obj = foo.models_init()
        except:
            raise Exception(f"\nFile {user_file_path} \n"
                            f"doesn't exists or contain function models_init, write it to access model objects or \n"
                            f"returns objects that don't inherit from GNNConstructor.\n"
                            f"def models_init should have the following logic:\n"
                            f"def models_init():\n"
                            f"  obj_1 = UserGNNClass_1(any parameters)\n"
                            f"  obj_2 = UserGNNClass_2(any parameters)\n"
                            f"  return locals()\n"
                            f"where UserGNNClass inherit from GNNConstructor")
        if obj_name in file_loc_obj:
            obj = file_loc_obj[obj_name]
            obj.obj_name = obj_name
            return obj
        else:
            raise Exception(f"File {user_file_path} doesn't have object {obj_name}")


if __name__ == '__main__':
    DataInfo.refresh_all_data_info()
    ps, info = DataInfo.models_parse()
    print(ps)
    print(json.dumps(info, indent=1))
