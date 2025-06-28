import json
from json.encoder import JSONEncoder
from pathlib import Path
from typing import Union, Dict, Any


class PrefixStorage:
    """ Prefix-tree like structure for keeping a set of tuples with important order.
    Supports:

    * adding, removing, filtering, iterating elements;
    * gathering contents from file structure.
    """
    def __init__(
            self,
            keys: Union[tuple, list]
    ):
        assert isinstance(keys, (tuple, list))
        assert len(keys) >= 1
        self._keys = keys
        self.content = {} if len(keys) > 1 else set()

    @property
    def depth(
            self
    ) -> int:
        return len(self._keys)

    @property
    def keys(
            self
    ) -> tuple:
        return tuple(self._keys)

    def size(
            self
    ) -> int:
        def count(obj):
            return sum(count(_) for _ in obj.values()) if isinstance(obj, dict) else len(obj)
        return count(self.content)

    def add(
            self,
            values: Union[dict, tuple, list]
    ) -> None:
        """
        Add one list of values.
        """
        if isinstance(values, dict):
            assert set(values.keys()) == set(self._keys)
            self.add([values[k] for k in self._keys])

        elif isinstance(values, (list, tuple)):
            assert len(values) == len(self._keys)

            def add(obj, depth):
                v = values[depth]
                if depth < self.depth - 1:
                    if v not in obj:
                        obj[v] = {} if depth < self.depth-2 else set()
                    add(obj[v], depth + 1)
                else:  # set
                    if v in obj:
                        raise ValueError(f"Element '{values}' already present")
                    obj.add(v)

            add(self.content, 0)

        else:
            raise TypeError("dict, tuple, or list were expected")

    def merge(
            self,
            ps,
            ignore_conflicts: bool = False
    ) -> None:
        """
        Extend this with another PrefixStorage with same keys.
        if ignore_conflicts=True, do not raise Exception when values sets intersect.
        While merging objects are not copied.
        """
        assert isinstance(ps, PrefixStorage)
        assert self._keys == ps.keys

        def merge(content1, content2):
            if isinstance(content1, set):  # set
                for item in content2:
                    if not ignore_conflicts and item in content1:
                        raise ValueError(f"Item '{item}' occurs in both PrefixStorages")
                    content1.add(item)
            else:
                for key, value in content2.items():
                    if key in content1:
                        merge(content1[key], content2[key])
                    else:
                        content1[key] = content2[key]

        merge(self.content, ps.content)

    def remove(
            self,
            values: Union[dict, tuple, list]
    ) -> None:
        """
        Remove one tuple of values if it is present.
        """
        assert len(values) == len(self._keys)
        if isinstance(values, dict):
            assert set(values.keys()) == set(self._keys)
            self.remove([values[k] for k in self._keys])

        elif isinstance(values, (list, tuple)):
            def rm(obj, depth):
                v = values[depth]
                if v in obj:
                    rm(obj[v], depth+1) if isinstance(obj, dict) else obj.remove(v)

            rm(self.content, 0)

    def filter(
            self,
            key_values: dict
    ):
        """
        Find all items satisfying specified key values. Returns a new PrefixStorage.
        """
        assert all(k in self._keys for k in key_values.keys())

        def filter(obj, depth):
            key = self._keys[depth]
            if isinstance(obj, dict):

                if key in key_values:  # take the value for 1 key
                    value = key_values[key]
                    if isinstance(value, dict):  # value could be a dict
                        value = json.dumps(value)
                    if value in obj:
                        return filter(obj[value], depth + 1)
                    else:  # all the rest is empty
                        return set()

                else:  # filter all
                    return {k: filter(v, depth+1) for k, v in obj.items()}
            else:  # set
                if key in key_values:  # 0 or 1 element
                    if key_values[key] in obj:
                        return {key_values[key]}
                    else:
                        return set()
                else:  # copy of full set
                    return set(obj)

        # Remove filtered keys
        ps = PrefixStorage([k for k in self.keys if k not in key_values])
        ps.content = filter(self.content, 0)
        return ps

    def check(
            self,
            values: Union[dict, tuple, list]
    ) -> bool:
        """
        Check if a tuple of values is present.
        """
        assert len(values) == len(self._keys)
        if isinstance(values, dict):
            assert set(values.keys()) == set(self._keys)
            return self.check([values[k] for k in self._keys])

        elif isinstance(values, (list, tuple)):
            data = self.content
            for i, v in enumerate(values):
                if v in data:
                    if i == self.depth-1:
                        return True
                    data = data[v]
                else:
                    return False

    def __iter__(
            self
    ):
        def enum(obj, elems):
            if isinstance(obj, (set, list)):
                for e in obj:
                    yield elems + [e]
            else:
                for k, v in obj.items():
                    for _ in enum(v, elems + [k]):
                        yield _

        for _ in enum(self.content, []):
            yield _

    @staticmethod
    def from_json(
            string: str
    ):
        """
        Construct PrefixStorage object from a json string.
        """
        data = json.loads(string)
        ps = PrefixStorage(data["keys"])
        ps.content = data["content"]
        return ps

    def to_json(
            self,
            **dump_args
    ) -> str:
        """ Return json string. """
        class Encoder(JSONEncoder):
            def default(self, obj):
                if isinstance(obj, set):
                    return sorted(list(obj))
                return json.JSONEncoder.default(self, obj)
        return json.dumps({"keys": self.keys, "content": self.content}, cls=Encoder, **dump_args)

    def fill_from_folder(
            self,
            path: Path,
            file_pattern: str = r".*"
    ) -> None:
        """
        Recursively walk over the given folder and repeat its structure.
        The content will be replaced.
        """
        import os
        import re

        res = []

        def walk(p, elems):
            for name in os.listdir(p):
                if os.path.isdir(p / name):
                    walk(p / name, elems + [name])
                else:
                    if re.fullmatch(file_pattern, name):
                        res.append(elems + [Path(name).stem])

        walk(path, [])
        self.content.clear()
        for e in res:
            if len(e) == self.depth:
                self.add(e)
        print(f"Added {self.size()} items of {len(res)} files found.")

    def remap(
            self,
            mapping,
            only_values: bool = False
    ):
        """
        Change keys order and combination.
        """
        # Check consistency
        ms = set()
        for m in mapping:
            if isinstance(m, (list, tuple)):
                ms.update(m)
            else:
                ms.add(m)
        assert ms == set(range(len(self.keys))),\
            f"mapping should contain key indices from 0 to {len(self.keys)-1}"

        keys = [",".join(self.keys[i] for i in m) if isinstance(m, (list, tuple)) else self.keys[m]
                for m in mapping]
        ps = PrefixStorage(keys)

        for item in self:
            values = []
            for m in mapping:
                if isinstance(m, (list, tuple)):
                    if only_values:
                        v = ",".join(str(item[i]) for i in m)
                    else:
                        v = ",".join(f"{self.keys[i]}={item[i]}" for i in m)
                else:
                    v = item[m]
                values.append(v)
            ps.add(values)

        return ps

class TuplePrefixStorage():
    """ Prefix-tree like structure for keeping objects with index as a tuple of keys.
    Supports:

    * adding, removing, filtering, iterating elements;
    * gathering contents from file structure.
    """
    def __init__(
            self,
    ):
        # {key -> value}, key is str, value is a tuple with final obj or a dict with same structure
        self.content: Dict[str, Union[str, Any]] = {}

    # @property
    # def depth(
    #         self
    # ) -> int:
    #     return len(self._keys)
    #
    # @property
    # def keys(
    #         self
    # ) -> tuple:
    #     return tuple(self._keys)

    def size(
            self
    ) -> int:
        def count(obj):
            return sum(count(_) for _ in obj.values()) if isinstance(obj, dict) else 1
        return count(self.content)

    def add(
            self,
            key: Union[list, tuple],
            obj: Any
    ) -> None:
        """
        Add one object.
        """
        assert len(key) > 0

        def _add(content, a_key):
            k, *tail = a_key
            if k in content:
                obj_or_key = content[k]
                if isinstance(obj_or_key, tuple):
                    raise KeyError(f"Storage already contains object by key {key}")
                if isinstance(obj_or_key, dict):
                    if len(a_key) == 1:
                        raise KeyError(f"Storage contains key {key}, so '{k}' cannot be object name")
                    else:
                        _add(obj_or_key, tuple(tail))
            else:
                if len(a_key) > 1:
                    next = {}
                    content[k] = next
                    _add(next, tuple(tail))
                else:
                    content[k] = (obj,)

        _add(self.content, key)

    def merge(
            self,
            ps: 'TuplePrefixStorage',
    ) -> None:
        """
        Extend this with another PrefixStorage with same keys.
        While merging objects are NOT copied.
        """

        def _merge(content1, content2):
            if isinstance(content1, tuple) or isinstance(content2, tuple):  # obj
                raise RuntimeError(f"Cannot merge, ")

            for k, v in content2.items():
                if k in content1:
                    _merge(content1[k], content2[k])
                else:
                    content1[k] = v  # FIXME copy?

        _merge(self.content, ps.content)

    def remove(
            self,
            key: tuple
    ) -> None:
        """
        Remove one tuple of values if it is present.
        """
        content = self.content
        parent = None
        for k in key:
            if k not in content:
                return False
            parent = content
            content = content[k]

        if isinstance(content, tuple):
            # We remove object
            del parent[key[-1]]
        if isinstance(content, dict):
            # We remove substructure
            del parent[key[-1]]

        return True

    # def filter(
    #         self,
    #         key_values: dict
    # ):
    #     """
    #     Find all items satisfying specified key values. Returns a new PrefixStorage.
    #     """
    #     assert all(k in self._keys for k in key_values.keys())
    #
    #     def filter(obj, depth):
    #         key = self._keys[depth]
    #         if isinstance(obj, dict):
    #
    #             if key in key_values:  # take the value for 1 key
    #                 value = key_values[key]
    #                 if isinstance(value, dict):  # value could be a dict
    #                     value = json.dumps(value)
    #                 if value in obj:
    #                     return filter(obj[value], depth + 1)
    #                 else:  # all the rest is empty
    #                     return set()
    #
    #             else:  # filter all
    #                 return {k: filter(v, depth+1) for k, v in obj.items()}
    #         else:  # set
    #             if key in key_values:  # 0 or 1 element
    #                 if key_values[key] in obj:
    #                     return {key_values[key]}
    #                 else:
    #                     return set()
    #             else:  # copy of full set
    #                 return set(obj)
    #
    #     # Remove filtered keys
    #     ps = PrefixStorage([k for k in self.keys if k not in key_values])
    #     ps.content = filter(self.content, 0)
    #     return ps

    def check(
            self,
            key: tuple
    ) -> bool:
        """
        Check if a tuple of values is present.
        """
        content = self.content
        for k in key:
            if k not in content:
                return False
            content = content[k]
        return True

    def __iter__(
            self
    ):
        def enum(obj, elems):
            if isinstance(obj, tuple):
                yield elems, obj[0]
            else:
                for k, v in obj.items():
                    for _ in enum(v, elems + [k]):
                        yield _

        for _ in enum(self.content, []):
            yield _

    @staticmethod
    def from_json(
            string: str
    ):
        """
        Construct PrefixStorage object from a json string.
        """
        ps = PrefixStorage()
        ps.content = json.loads(string)
        return ps

    def to_json(
            self,
            **dump_args
    ) -> str:
        """ Return json string. """
        return json.dumps(self.content, **dump_args)

    def fill_from_folder(
            self,
            path: Path,
            file_pattern: str = r".*"
    ) -> None:
        """
        Recursively walk over the given folder and repeat its structure.
        The content will be replaced.
        """
        import os
        import re

        res = []

        def walk(p, elems):
            for name in os.listdir(p):
                if os.path.isdir(p / name):
                    walk(p / name, elems + [name])
                else:
                    if re.fullmatch(file_pattern, name):
                        res.append(elems + [Path(name).stem])

        walk(path, [])
        self.content.clear()
        for e in res:
            self.add(e, None)
        print(f"Added {self.size()} items of {len(res)} files found.")

    # def remap(
    #         self,
    #         mapping,
    #         only_values: bool = False
    # ):
    #     """
    #     Change keys order and combination.
    #     """
    #     # Check consistency
    #     ms = set()
    #     for m in mapping:
    #         if isinstance(m, (list, tuple)):
    #             ms.update(m)
    #         else:
    #             ms.add(m)
    #     assert ms == set(range(len(self.keys))),\
    #         f"mapping should contain key indices from 0 to {len(self.keys)-1}"
    #
    #     keys = [",".join(self.keys[i] for i in m) if isinstance(m, (list, tuple)) else self.keys[m]
    #             for m in mapping]
    #     ps = PrefixStorage(keys)
    #
    #     for item in self:
    #         values = []
    #         for m in mapping:
    #             if isinstance(m, (list, tuple)):
    #                 if only_values:
    #                     v = ",".join(str(item[i]) for i in m)
    #                 else:
    #                     v = ",".join(f"{self.keys[i]}={item[i]}" for i in m)
    #             else:
    #                 v = item[m]
    #             values.append(v)
    #         ps.add(values)
    #
    #     return ps

    def __str__(
            self
    ) -> str:

        def _print(content, lvl):
            res = ""
            offset = "  " * lvl
            for k, v in content.items():
                if isinstance(v, tuple):
                    res += f"{offset}- {k} : {str(v[0])[:10]}\n"
                else:
                    res += f"{offset}- {k}\n"
                    res += _print(v, lvl+1)
            return res

        res = _print(self.content, 0)
        return res.rstrip()


def test_tuple_storage():
    d = TuplePrefixStorage()

    # Adding items
    d.add(("single", "test"), "OBJ1")
    d.add(("multiple-graphs", "custom", "example_gml"), "OBJ2")
    d.add(("multiple-graphs", "custom", "small"), "OBJ3")
    d.add(("multiple-graphs", "custom", "example"), "OBJ4")
    d.add(("single-graph", "Planetoid", "Cora"), "OBJ5")
    d.add(("single-graph", "example"), "OBJ6")
    d.add(("single-graph", "custom", "example_gml"), "OBJ7")
    d.add(("single-graph", "custom", "example"), "OBJ8")

    print(d)
    print(d.size())

    # Removing items
    d.remove(("multiple-graphs", "custom", "example_gml"))
    d.remove(("multiple-graphs", "custom", "unknown"))  # no effect
    d.remove(("single-graph", "custom"))

    print(d)
    print(d.size())

    # Checking items
    print(d.check(("single-graph", "Planetoid", "Cora")))  # True
    print(d.check(("single-graph", "custom", "example")))  # False

    # Merging
    d1 = TuplePrefixStorage()
    d1.add(("single-graph", "custom", "example"), "OBJ11")
    d1.add(("single-graph", "Planetoid", "Citeseer"), "OBJ12")
    d1.add(("single-graph", "test"), "OBJ13")
    print(d1.size())
    d.merge(d1)
    print(d)
    print(d.size())

    # Iterating items
    for ps_item in d:
        print(ps_item)

    # # Filtering by key-values
    # f = d.filter({"graph": "10K", "value": 2})
    #
    # Get as JSON
    print(d.to_json(indent=2))

    # Parse from a folder
    from aux.utils import root_dir
    ps_test = TuplePrefixStorage()
    ps_test.fill_from_folder(root_dir / 'data', file_pattern=r".*metainfo")
    print(ps_test)

    # # Remapping
    # r = d.remap([2, [0, 1, 3]])
    # print(r.to_json(indent=2))


def test_prefix_storage():
    print("PrefixStorage demo")
    d = PrefixStorage(("group", "graph", "attribute", "value"))

    # Adding items
    d.add(("VK", "10K", "sex", 1))
    d.add(("VK", "10K", "sex", 2))
    d.add(("VK", "10K", "age", 2))
    d.add(("VK", "100K", "sex", 1))
    d.add(("VK", "100K", "sex", 2))
    d.add(("VK", "100K", "sex", 3))

    # Removing items
    d.remove(("VK", "100K", "sex", 2))
    d.remove(("VK", "100K", "sex", 5))  # no effect

    # Checking items
    print(d.check(("VK", "100K", "sex", 1)))  # True
    print(d.check(("VK", "100K", "sex", 2)))  # False

    # Merging
    d1 = PrefixStorage(("group", "graph", "attribute", "value"))
    d1.add(("VK", "10K", "sex", 2))
    d1.add(("VK", "10K", "sex", 3))
    print(d.size())
    d.merge(d1, ignore_conflicts=True)
    print(d.size())

    # Iterating items
    for ps_item in d:
        print(ps_item)

    # Filtering by key-values
    f = d.filter({"graph": "10K", "value": 2})

    # Get as JSON
    print(f.to_json(indent=2))

    # Parse from a folder
    from aux.utils import root_dir
    ps_test = PrefixStorage(("folder", "name"))
    ps_test.fill_from_folder(root_dir / 'src', file_pattern=r".*\.py")
    print(ps_test.to_json(indent=2))

    # Remapping
    r = d.remap([2, [0, 1, 3]])
    print(r.to_json(indent=2))


if __name__ == '__main__':
    # test_tuple_storage()
    # test_prefix_storage()

    d = TuplePrefixStorage()

    # Adding items
    d.add(("single", "test"), "OBJ1")
    d.add(("multiple-graphs", "custom", "example_gml"), "OBJ2")
    d.add(("multiple-graphs", "custom", "small"), "OBJ3")
    d.add(("multiple-graphs", "custom", "example"), "OBJ4")
    d.add(("single-graph", "Planetoid", "Cora"), "OBJ5")
    d.add(("single-graph", "example"), "OBJ6")
    d.add(("single-graph", "custom", "examples1"), "OBJ7")
    d.add(("single-graph", "custom", "examples", "test1"), "OBJ7")
    d.add(("single-graph", "custom", "examples", "test2"), "OBJ7")
    d.add(("single-graph", "custom", "examples", "tests", "test3"), "OBJ7")
    d.add(("single-graph", "custom", "example"), "OBJ8")

    print(d.to_json(indent=2))
