import collections
import importlib.util
import json
import logging
from typing import List, Tuple, Union

# from pydantic.utils import deep_update
# from pydantic.v1.utils import deep_update

from aux.utils import MODELS_DIR, GRAPHS_DIR, EXPLANATIONS_DIR, root_dir_len, DATA_INFO_DIR, \
    USER_MODELS_DIR, METAINFO_DIR, SAVE_DIR_STRUCTURE_PATH
import os
from pathlib import Path
from aux.prefix_storage import PrefixStorage

# Hierarchy of dataset naming
from models_builder.gnn_constructor import GNNConstructor

DATASET_KEYS = ("domain", "group", "graph")


class DataInfo:
    """
    The class is responsible for populating prefix access trees
    to datasets, models, and interpretations based on the dir structure
    """

    @staticmethod
    def refresh_all_data_info(
    ) -> None:
        """
        Calling all files to update with information about saved objects
        """
        DATA_INFO_DIR.mkdir(exist_ok=True, parents=True)
        DataInfo.refresh_data_dir_structure()
        DataInfo.refresh_data_var_dir_structure()
        DataInfo.refresh_models_dir_structure()
        DataInfo.refresh_explanations_dir_structure()

    @staticmethod
    def refresh_data_dir_structure(
    ) -> None:
        """
        Calling a file update with information about saved raw datasets
        """
        DATA_INFO_DIR_data = DATA_INFO_DIR / 'data_dir_structure'
        with open(DATA_INFO_DIR_data, 'w', encoding='utf-8') as f:
            # IMP suggest create a constant for "dataset_ver_ind" and such strings, same for next 2 functions
            prev_path = ''
            for path in Path(GRAPHS_DIR).glob('**/raw/.info'):
                path = path.parts[root_dir_len + 1:-2]
                path = str(Path(*path)) + '\n'
                if prev_path != path:
                    f.write(path)
                    prev_path = path

    @staticmethod
    def refresh_models_dir_structure(
    ) -> None:
        """
        Calling a file update with information about saved models
        """
        DATA_INFO_DIR_models = DATA_INFO_DIR / 'models_dir_structure'
        with open(DATA_INFO_DIR_models, 'w', encoding='utf-8') as f:
            for path in Path(MODELS_DIR).glob('**/model'):
                path = path.parts[root_dir_len + 1:-1]
                f.write(str(Path(*path)) + '\n')

    @staticmethod
    def refresh_explanations_dir_structure(
    ) -> None:
        """
        Calling a file update with information about saved explanations
        """
        DATA_INFO_DIR_results = DATA_INFO_DIR / 'explanations_dir_structure'
        with open(DATA_INFO_DIR_results, 'w', encoding='utf-8') as f:
            for path in Path(EXPLANATIONS_DIR).glob('**/explanation.json'):
                path = path.parts[root_dir_len + 1:-1]
                f.write(str(Path(*path)) + '\n')

    @staticmethod
    def refresh_data_var_dir_structure(
    ) -> None:
        """
        Calling a file update with information about saved prepared datasets
        """
        DATA_INFO_DIR_results = DATA_INFO_DIR / 'data_var_dir_structure'
        with open(DATA_INFO_DIR_results, 'w', encoding='utf-8') as f:
            for path in Path(GRAPHS_DIR).glob('**/data.pt'):
                path = path.parts[root_dir_len + 1:-1]
                f.write(str(Path(*path)) + '\n')

    @staticmethod
    def take_keys_etc_by_prefix(
            prefix: Tuple
    ) -> [List, List, dict, int]:
        """

        :param prefix: what data and in what order were used to form the path when saving the object
         Example: modes=("data_root", "data_prepared", "models", "explanations")
        :return: keys_list witch should use specific info about object,
         full_keys_list take all keys (with technical keys),
         dir_structure dict base on save_dir_structure.json without modes keys,
         empty_dir_shift number of technical keys
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
    ) -> List:
        """

        :param path: path of the saved object
        :param full_keys_list: keys witch should use spesific info about object and technical keys
        :param dir_structure: dict base on save_dir_structure.json without modes keys
        :return: object values based on object path
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
            prefix: Tuple
    ) -> [List, dict]:
        """

        :param path: path of the saved object
        :param prefix: what data and in what order were used to form the path when saving the object
         Example: modes=("data_root", "data_prepared", "models", "explanations")
        :return: object values based on object path and dict with technical files
        """
        with open(SAVE_DIR_STRUCTURE_PATH) as f:
            save_dir_structure = json.loads(f.read())
        parts_val = []
        description_info = {}
        path = Path(path).parts
        parts_parse = 0
        for prefix_part in prefix[:-1]:
            for key, val in save_dir_structure[prefix_part].items():
                if val["add_key_name_flag"] is not None:
                    if not val["add_key_name_flag"]:
                        parts_val.append(path[parts_parse].strip())
                    else:
                        parts_val.append(path[parts_parse].strip().split(f'{key}=', 1)[1])
                parts_parse += 1
        if len(prefix) > 0:
            for key, val in save_dir_structure[prefix[-1]].items():
                if val["add_key_name_flag"] is not None:
                    if not val["add_key_name_flag"]:
                        parts_val.append(path[parts_parse].strip())
                    else:
                        parts_val.append(path[parts_parse].strip().split(f'{key}=', 1)[1])
                if val["files_info"] is not None:
                    for file_info_dict in val["files_info"]:
                        if file_info_dict["file_name"] == "origin":
                            file_name = path[parts_parse].strip()
                        else:
                            file_name = file_info_dict["file_name"]
                        file_name += file_info_dict["format"]
                        description_info.update(
                            {key: {parts_val[-1]: os.path.join(os.path.join(*path[:parts_parse]), file_name)}})
                parts_parse += 1
        return parts_val, description_info

    @staticmethod
    def fill_prefix_storage(
            prefix: Tuple,
            file_with_paths: Union[str, Path]
    ) -> [PrefixStorage, dict]:
        """
        Fill prefix storage by file with paths

        :param prefix: what data and in what order were used to form the path when saving the object
         Example: modes=("data_root", "data_prepared", "models", "explanations")
        :param file_with_paths: file with paths of saved objects
        :return: fill prefix storage and dict with description_info about objects use hash
        """
        keys_list, full_keys_list, dir_structure, empty_dir_shift = \
            DataInfo.take_keys_etc_by_prefix(prefix=prefix)
        ps = PrefixStorage(keys_list)
        with open(file_with_paths, 'r', encoding='utf-8') as f:
            description_info = {
            }
            for line in f:
                if len(ps.keys) != len(Path(line).parts) - empty_dir_shift:
                    continue
                # loc_parts_values = DataInfo.values_list_by_path_and_keys(path=line, full_keys_list=full_keys_list,
                #                                                          dir_structure=dir_structure)
                loc_parts_values, description_info_loc = DataInfo.values_list_and_technical_files_by_path_and_prefix(
                    path=line, prefix=prefix,
                )
                ps.add(loc_parts_values)
                description_info = DataInfo.deep_update(description_info, description_info_loc)
        return ps, description_info

    @staticmethod
    def deep_update(
            d: dict,
            u: dict
    ) -> dict:
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
        for description_info_key, description_info_val in description_info.items():
            for obj_name, obj_file_path in description_info_val.items():
                with open(os.path.join(root_path, obj_file_path)) as f:
                    description_info[description_info_key][obj_name] = f.read()
        return description_info

    @staticmethod
    def explainers_parse(
    ) -> [PrefixStorage, dict]:
        """
        Parses the path to explainers from a technical file with the paths of all saved explainers.
        """
        DATA_INFO_DIR_results = DATA_INFO_DIR / 'explanations_dir_structure'
        ps, description_info = DataInfo.fill_prefix_storage(
            prefix=("data_root", "data_prepared", "models", "explanations"),
            file_with_paths=DATA_INFO_DIR_results)
        description_info = DataInfo.description_info_with_paths_to_description_info_with_files_values(
            description_info=description_info, root_path=EXPLANATIONS_DIR,
        )
        return ps, description_info

    @staticmethod
    def models_parse(
    ) -> [PrefixStorage, dict]:
        """
        Parses the path to models from a technical file with the paths of all saved models.
        """
        DATA_INFO_DIR_results = DATA_INFO_DIR / 'models_dir_structure'
        ps, description_info = DataInfo.fill_prefix_storage(
            prefix=("data_root", "data_prepared", "models"),
            file_with_paths=DATA_INFO_DIR_results)
        description_info = DataInfo.description_info_with_paths_to_description_info_with_files_values(
            description_info=description_info, root_path=MODELS_DIR,
        )
        return ps, description_info

    @staticmethod
    def data_parse(
    ) -> [PrefixStorage, dict]:
        """
        Parses the path to raw datasets from a technical file with the paths of all saved raw datasets.
        """
        DATA_INFO_DIR_results = DATA_INFO_DIR / 'data_dir_structure'
        ps, description_info = DataInfo.fill_prefix_storage(
            prefix=("data_root",),
            file_with_paths=DATA_INFO_DIR_results)
        return ps

    @staticmethod
    def data_var_parse(
    ) -> [PrefixStorage, dict]:
        """
        Parses the path to prepared datasets from a technical file with the paths of all saved prepared datasets.
        """
        DATA_INFO_DIR_results = DATA_INFO_DIR / 'data_var_dir_structure'
        ps, description_info = DataInfo.fill_prefix_storage(
            prefix=("data_root", "data_prepared"),
            file_with_paths=DATA_INFO_DIR_results)
        return ps

    @staticmethod
    def clean_prepared_data(
            dry_run: bool = False
    ) -> None:
        """
        Remove all prepared data for all datasets.
        """
        import shutil
        for path in Path(GRAPHS_DIR).glob('**/prepared'):
            print(path)
            if not dry_run:
                shutil.rmtree(path)

    @staticmethod
    def all_obj_ver_by_obj_path(
            obj_dir_path: Union[str, Path]
    ) -> set:
        """

        :param obj_dir_path: path to the saved object
        :return: set of all saved versions of the provided object
        """
        obj_dir_path = Path(obj_dir_path).parent
        vers_ind = []
        for dir_path, dir_names, filenames in os.walk(obj_dir_path):
            if not dir_names and filenames:
                # ver_ind.append()
                vers_ind.append(int(Path(dir_path).parts[-1].rsplit(sep='=', maxsplit=1)[-1]))
        return set(vers_ind)

    @staticmethod
    def del_all_empty_folders(
            dir_path: Union[str, Path]
    ) -> None:
        """
        Deletes all empty folders and files with meta information in the selected directory

        :param dir_path: path to the directory in which empty folders should be deleted
        """
        for dir_path, dir_names, filenames in os.walk(dir_path, topdown=False):
            # print(dir_names, filenames, '\n', dir_path)
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
    @staticmethod
    def user_models_list_ref(
    ) -> dict:
        """
        :return: dict with information about user models objects in directory <project root>/user_model_list
         Contains information about objects class name, objects names and import paths
         Dict structure:
         user_models_obj_dict_info[class_name] = {
             'obj_names': list,
             'import_path': str
         }
         <project root>/user_model_list should have def models_init. Examole:
         def models_init():
             obj_1 = UserGNNClass_1(any parameters)
             obj_2 = UserGNNClass_2(any parameters)
             return locals()
         where UserGNNClass inherits from GNNConstructor
        """
        DATA_INFO_DIR.mkdir(exist_ok=True, parents=True)
        DATA_INFO_USER_MODELS_INFO = DATA_INFO_DIR / 'user_model_list'
        user_models_obj_dict_info = {}
        for path in os.scandir(USER_MODELS_DIR):
            if path.is_file():
                # print(path.name)
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
                    # user_models_obj_dict_info[user_files_path] = {}
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
    ) -> object:
        """
        :param user_file_path: path to the user file with user model
        :param obj_name: user object name
        :return: import user object from user file
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
    print(ps.to_json())
    # ps, info = DataInfo.fill_prefix_storage(modes=("data_root", "data_prepared", "models"),
    #                                         file_with_paths=DATA_INFO_DIR / 'models_dir_structure')
    # print(info)
    # DataInfo.clean_prepared_data()
    # DataInfo.del_all_empty_folders(MODELS_DIR)
    # UserCodeInfo.user_models_list_ref()
    pass
