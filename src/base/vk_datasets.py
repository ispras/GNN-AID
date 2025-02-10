import ast
import datetime
import json
from bisect import bisect_right
from numbers import Number
from operator import itemgetter
from pathlib import Path
from typing import Union

import numpy as np

from aux.utils import GRAPHS_DIR
from base.custom_datasets import CustomDataset
from aux.configs import DatasetConfig

AGE_GROUPS = [15, 20, 25, 30, 35, 40, 50, 60]


class AttrInfo:
    """
    """
    _attribute_vals_cache = {}  # (full_name, attribute) -> attribute_vals

    @staticmethod
    def vk_attr(
    ) -> dict:
        vk_dict = {
            ('age',): list(range(0, len(AGE_GROUPS) + 1)),
            ('sex',): [1, 2],
            ('relation',): list(range(1, 9)),
            # ('occupation', 'type'): ['work', 'university', 'school'],
            ('personal', 'political'): list(range(1, 10)),
            ('personal', 'people_main'): list(range(1, 7)),
            ('personal', 'life_main'): list(range(1, 9)),
            ('personal', 'smoking'): list(range(1, 6)),
            ('personal', 'alcohol'): list(range(1, 6)),
            ('schools', 'type_str'): list(range(0, 14)),
        }
        return vk_dict

    @staticmethod
    def attribute_vals(
            full_name: tuple,
            attribute: [str, tuple, list]
    ) -> list:
        """ Get a set of possible attribute values or None for textual or continuous attributes.
        """
        # Convert to tuple
        if isinstance(attribute, list):
            attribute = tuple(attribute)
        elif isinstance(attribute, str):
            attribute = tuple(attribute.split())

        # Check if cached
        if (full_name, attribute) in AttrInfo._attribute_vals_cache:
            return AttrInfo._attribute_vals_cache[(full_name, attribute)]

        vk_dict = {
            ('age',): list(range(0, len(AGE_GROUPS) + 1)),
            ('sex',): [1, 2],
            ('relation',): list(range(1, 9)),
            # ('occupation', 'type'): ['work', 'university', 'school'],
            ('personal', 'political'): list(range(1, 10)),
            ('personal', 'people_main'): list(range(1, 7)),
            ('personal', 'life_main'): list(range(1, 9)),
            ('personal', 'smoking'): list(range(1, 6)),
            ('personal', 'alcohol'): list(range(1, 6)),
            ('schools', 'type_str'): list(range(0, 14)),
        }
        res = None
        if full_name[1] == 'vk_samples':
            res = vk_dict[attribute]

        elif full_name == ('attributed', 'vk_10_classes'):
            if attribute in vk_dict:
                res = vk_dict[attribute]
            elif attribute == "community_class":
                raise NotImplementedError()
            else:  # Бизнес, Программирование, etc
                res = [1, 0]

        elif full_name == ('test', 'toy'):
            # TODO Misha read from ...info file
            res = {
                'a': None,  # continuous
                'b': ["A", "B", "CCC"]
            }[attribute[0]]

        else:
            raise NotImplementedError(f"Attribute values for graph {full_name}")

        AttrInfo._attribute_vals_cache[(full_name, attribute)] = res
        return res

    @staticmethod
    def one_hot(
            full_name: tuple,
            attribute: [str, tuple, list],
            value: int,
            add_none: bool = False
    ) -> Union[np.ndarray, list]:
        """ 1-hot encoding feature. If no such value, return all zeros or with 1 it in last element.
        :param full_name:
        :param attribute: attribute name, e.g. 'sex', ('personal', 'smoking').
        :param value: value of this attribute. If a list, a multiple-hot vector will be constructed.
        :param add_none: if True, last element of returned vector encodes undefined or
         not-in-the-list attribute value. If False, in case of undefined value, vector of all zeros
         will be returned.
        :return: vector of length = len(AttrHelper.attribute_vals(graph, attribute)) or +1 if
         add_none is True.
        """
        # TODO if called often, we can cache it
        allowed_vals = AttrInfo.attribute_vals(full_name, attribute)

        if allowed_vals is None:  # continuous - return as list of this value
            assert isinstance(value, Number)
            return [value]

        if isinstance(value, list):  # multiple-hot vector
            vals = set(value)
            if add_none:
                none = True
                res = np.zeros(len(allowed_vals) + 1)
            else:
                res = np.zeros(len(allowed_vals))
            for pos, val in enumerate(allowed_vals):
                if val in vals:
                    res[pos] = 1
                    none = False
            if add_none and none:
                res[-1] = 1
            return res

        if add_none:
            res = np.zeros(len(allowed_vals) + 1)
            for pos, val in enumerate(allowed_vals):
                if val == value:
                    res[pos] = 1
                    return res
            res[-1] = 1  # last element means value is undefined or not in the list
            return res
        else:
            res = np.zeros(len(allowed_vals))
            for pos, val in enumerate(allowed_vals):  # NOTE: iteration over set not list
                # FIXME str or int?
                if val == value:
                    res[pos] = 1
                    break
            return res  # all zeros here


class ConfigPatter:
    pass


class VKDataset(
    CustomDataset
):
    """
    Custom dataset of VK samples with specific attributes processing and features creation.
    """
    def __init__(
            self,
            dataset_config: Union[ConfigPatter, DatasetConfig],
            add_none: bool = False
    ):
        """
        Args:
            dataset_config: DatasetConfig dict from frontend
            add_none: flag indicating whether to add a value under the none class in the feature
             vector.
        """
        super().__init__(dataset_config)
        self.add_none = add_none

    def _compute_dataset_data(
            self
    ) -> None:
        """ Get DatasetData for VK graph
        """
        super()._compute_dataset_data()

        # TODO Misha do we want add node attributes to send to front? See attr name
        #  Problem is that attr names are diff in attrs folder and in .info
        self.dataset_data["node_attributes"] = {}

        # # Add node labelings present in folder
        # labelings = {}
        # for filename in os.listdir(self.labels_dir):
        #     with open(self.labels_dir / filename, 'r') as f:
        #         d = json.load(f)
        #         # TODO Misha get unique
        #         labelings[filename] = max([-1 if x is None else x for x in d.values()]) + 1
        # self.dataset_data["info"]["labelings"] = labelings

    def _feature_tensor(
            self,
            g_ix=None
    ) -> list:
        # FIXME Misha self.node_map[graph] ...
        x = [[] for _ in range(len(self.node_map))]
        features = self.dataset_var_config.features
        if not features:
            raise RuntimeError("features_dict is empty, need any feature, to create feature tensor")

        # TODO Kirill support all features types

        if "str_g" in features and features["str_g"] == "one_hot":
            raise NotImplementedError()

        if 'attr' in features and features['attr']:
            if str(('age',)) in features['attr'] and features['attr']:
                self.bdate_to_age(attr_dir_path=self.node_attributes_dir, node_map=self.node_map)
            for key, value in features['attr'].items():
                key_loc = ast.literal_eval(key)
                with open(self.node_attributes_dir / Path(key_loc[0]), 'r') as f:
                    attr_dict = json.load(f)
                    for i, node in self._iter_nodes():
                        # print(node)
                        one_hot = list(AttrInfo.one_hot(full_name=self.dataset_config.full_name(),
                                                        attribute=key_loc,
                                                        value=attr_dict[node],
                                                        add_none=self.add_none))
                        x[i] += one_hot

        return x

    @staticmethod
    def bdate_to_age(
            attr_dir_path: str,
            node_map: list
    ) -> None:
        with open(attr_dir_path / Path('bdate'), 'r') as f:
            age_dict = json.load(f)
            node_age = {}
            for key, value in age_dict.items():
                dmy = value.split('.')
                if len(dmy) == 3:
                    age = datetime.date.today().year - int(dmy[2])
                    if 10 < age < 90:
                        group = bisect_right(AGE_GROUPS, age)
                        node_age[key] = group
            for key in node_map:
                if key not in node_age:
                    node_age[key] = -1

            with open(attr_dir_path / Path('age'), 'w', encoding='utf-8') as f1:
                json.dump(node_age, f1)


def make_vk_labeling(
        attr_path: str,
        labeling_path: str,
        attr_val: int = 1
) -> None:
    """
    Creates a markup file where the attribute's target value is set to 1 and the rest to 0
    Args:
        attr_path: path to file with attributes
        labeling_path: path to save markup file
        attr_val: attribute target value

    Returns:

    """
    with open(attr_path, 'r') as f:
        attr_dict = json.load(f)
    labeling_dict = {}
    for key, elem in attr_dict.items():
        if elem == attr_val:
            labeling_dict[key] = 1
        elif elem == 0:
            labeling_dict[key] = None
        else:
            labeling_dict[key] = 0

    Path(labeling_path).parent.mkdir(parents=True, exist_ok=True)
    with open(labeling_path, 'w') as f:
        f.write(json.dumps(labeling_dict))


if __name__ == '__main__':
    # Print VK attributes and their values
    names, values = list(zip(*sorted(AttrInfo.vk_attr().items(), key=itemgetter(0))))
    print(list(map(str, names)))
    print(list(values))

    # Create labels
    graph_name = 'vk2-ff20-N10000-A.1611943634'
    attrib_path = GRAPHS_DIR / (
            'single-graph/vk_samples/' + graph_name + '/raw/' + graph_name + '.node_attributes/sex')
    label_path = GRAPHS_DIR / (
            'single-graph/vk_samples/' + graph_name + '/raw/' + graph_name + '.labels/sex2')
    make_vk_labeling(attr_path=attrib_path, labeling_path=label_path, attr_val=2)
