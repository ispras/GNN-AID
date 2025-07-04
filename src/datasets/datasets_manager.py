import json
import shutil
import os
from pathlib import Path
from typing import Union

import torch
import torch_geometric
from torch import default_generator, randperm
from torch_geometric.data import Dataset, InMemoryDataset, Data
from torch_geometric.datasets import BAShapes

from datasets.dataset_info import DatasetInfo
from datasets.gen_dataset import GeneralDataset
from data_structures.configs import DatasetConfig, DatasetVarConfig, ConfigPattern
from aux.custom_decorators import timing_decorator
from aux.declaration import Declare
from aux.utils import TORCH_GEOM_GRAPHS_PATH, tmp_dir, import_by_name, SOURCE_DIR


class DatasetManager:
    """
    Class for working with datasets. Methods: get - loads dataset in torch_geometric format for gnn
    along the path specified in full_name as tuple. Currently also supports automatic loading and
    processing of all datasets from torch_geometric.datasets
    """

    # @staticmethod
    # def register_torch_geometric_local(
    #         dataset: InMemoryDataset,
    #         name: str = None
    # ) -> GeneralDataset:
    #     """
    #     Save a given PTG dataset locally.
    #     Dataset is then always available for use by its config.
    #
    #     :param dataset: torch_geometric.data.Dataset object.
    #     :param name: user defined dataset name. If not specified, a timestamp will be used.
    #     :return: GeneralDataset
    #     """
    #     gen_dataset = DatasetManager._register_torch_geometric(
    #         dataset, name=name, group='local', exists_ok=False, copy_data=True)
    #
    #     return gen_dataset

    @staticmethod
    def get_by_config(
            dataset_config: DatasetConfig,
            **params
    ) -> GeneralDataset:
        """ Get GeneralDataset by dataset config. Convenient to use from the frontend.
        """
        path = Declare.dataset_info_path(dataset_config)

        # Check special cases when there is no metainfo file but we know where to get class
        if not path.exists():
            from datasets.ptg_datasets import LibPTGDataset
            if dataset_config.full_name[0] == LibPTGDataset.data_folder:
                class_name = LibPTGDataset.__name__
                import_from = LibPTGDataset.__module__
            else:
                raise RuntimeError(f"No metainfo file found for dataset with config {dataset_config}")

        else:
            # Read metainfo
            info = DatasetInfo.read(path)
            class_name = info.class_name
            import_from = info.import_from
            if class_name is None or import_from is None:
                raise RuntimeError(f"Metainfo file does not contain field 'class_name' or 'import_from'."
                                   f" They must be specified in metainfo file, check it {path}")

        klass = import_by_name(class_name, [import_from])
        return klass(dataset_config=dataset_config, **params)

    # @staticmethod
    # def get_by_config_and_var_config(
    #         dataset_config: DatasetConfig,
    # ) -> GeneralDataset:
    #     """ Get GeneralDataset by dataset config and var config. Convenient for test in backend
    #     """
    #     info = DatasetInfo.read(Declare.dataset_info_path(dataset_config))
    #     klass = import_by_name(info.class_name, [info.import_from])
    #     return klass(**dataset_config.init_kwargs)

    @staticmethod
    @timing_decorator
    def get_by_full_name(
            full_name=None,
            **kwargs
    ) -> [GeneralDataset, torch_geometric.data.Data, Path]:
        """
        Get PTG dataset by its full name tuple.
        Starts the creation of an object from raw data or takes already saved datasets in prepared
        form.

        Args:
            full_name: full name of graph data
            **kwargs: other arguments required to create datasets (dataset_var_config)

        Returns: GeneralDataset, a list of tensors with data, and
        the path where the dataset is saved.

        """
        from datasets.ptg_datasets import PTGDataset
        dataset = DatasetManager.get_by_config(DatasetConfig(full_name))
        cfg = PTGDataset.default_dataset_var_config.to_savable_dict()
        cfg.update(**kwargs)
        dataset_var_config = DatasetVarConfig(**cfg)
        dataset.build(dataset_var_config=dataset_var_config)
        dataset.train_test_split(percent_train_class=kwargs.get("percent_train_class", 0.8),
                                 percent_test_class=kwargs.get("percent_test_class", 0.2))
        # IMP Kirill suggest to return only dataset, else is its parts
        return dataset, dataset.data, dataset.results_dir

    # @staticmethod
    # def register_torch_geometric_api(
    #         dataset: Dataset,
    #         name: str = None,
    #         obj_name: str = 'DATASET_TO_EXPORT'
    # ) -> GeneralDataset:
    #     """
    #     Register a user defined code implementing a PTG dataset.
    #     This function should be called at each framework run to make the dataset available for use.
    #
    #     :param dataset: torch_geometric.data.Dataset object.
    #     :param name: user defined dataset name. If not specified, a timestamp will be used.
    #     :param obj_name: variable name to locate when import. DATASET_TO_EXPORT is default
    #     :return: GeneralDataset
    #     """
    #     gen_dataset = DatasetManager._register_torch_geometric(
    #         dataset, name=name, group='api', exists_ok=False, copy_data=False)
    #
    #     # Save api info
    #     import inspect
    #     import_path = Path(inspect.getfile(dataset.__class__))
    #     api = {
    #         'import_path': str(import_path),
    #         'obj_name': obj_name,
    #     }
    #     with gen_dataset.api_path.open('w') as f:
    #         json.dump(api, f, indent=1)
    #
    #     return gen_dataset
    #
    # @staticmethod
    # def _register_torch_geometric(
    #         dataset: Dataset,
    #         name: Union[str, None] = None,
    #         group: str = None,
    #         exists_ok: bool = False,
    #         copy_data: bool = False
    # ) -> GeneralDataset:
    #     """
    #     Create GeneralDataset from an externally specified torch geometric dataset.
    #
    #     :param dataset: torch_geometric.data.Dataset object.
    #     :param name: dataset name.
    #     :param group: group name, preferred options are: 'local', 'exported', etc.
    #     :param exists_ok: if True raise Exception if graph with same name exists, otherwise the data
    #      will be overwritten.
    #     :param copy_data: if True processed data will be copied, otherwise a symbolic link is
    #      created.
    #     :return: GeneralDataset
    #     """
    #     info = DatasetInfo.induce(dataset)
    #     if name is None:
    #         import time
    #         info.name = 'graph_' + str(time.time())
    #     else:
    #         assert isinstance(name, str) and os.sep not in name
    #         info.name = name
    #
    #     # Define graph configuration
    #     dataset_config = DatasetConfig(
    #         domain="single-graph" if info.count == 1 else "multiple-graphs",
    #         group=group or 'exported', graph=info.name
    #     )
    #
    #     # Check if exists
    #     root_dir, _ = Declare.dataset_root_dir(dataset_config)
    #     if root_dir.exists():
    #         if exists_ok:
    #             shutil.rmtree(root_dir)
    #         else:
    #             raise FileExistsError(
    #                 f"Graph with config {dataset_config.full_name} already exists!")
    #
    #     from datasets.ptg_datasets import PTGDataset
    #     gen_dataset = PTGDataset(dataset_config=dataset_config)
    #     info.save(gen_dataset.info_path)
    #
    #     # Link or copy original contents to our path
    #     results_dir = gen_dataset.results_dir
    #     results_dir.parent.mkdir(parents=True, exist_ok=True)
    #     if copy_data:
    #         shutil.copytree(os.path.abspath(dataset.processed_dir), results_dir,
    #                         dirs_exist_ok=True)
    #     else:  # Create symlink
    #         # FIXME what will happen if we modify graph and its data.pt ?
    #         results_dir.symlink_to(os.path.abspath(dataset.processed_dir),
    #                                target_is_directory=True)
    #
    #     gen_dataset.dataset = dataset
    #     gen_dataset.info = info
    #     print(f"Registered graph '{info.name}' as {dataset_config.full_name}")
    #     return gen_dataset


if __name__ == '__main__':
    # TODO remove
    print("test configs")

    # DataInfo.refresh_data_dir_structure()

    # dc = DatasetConfig(('single', 'test1'), init_kwargs={"a": 10, "b": 'line'})
    # print(Declare.dataset_root_dir(dc))

    # ba1 = BAShapes()
    # ba2 = BAShapes(connection_distribution="uniform")

    class UserLocalDataset(InMemoryDataset):
        def __init__(self, root, data_list, transform=None):
            self.data_list = data_list
            super().__init__(root, transform)
            # NOTE: it is important to define self.slices here, since it is used to calculate len()
            self.data, self.slices = torch.load(self.processed_paths[0])

        @property
        def processed_file_names(self):
            return 'data.pt'

        def process(self):
            torch.save(self.collate(self.data_list), self.processed_paths[0])


    dc = DatasetConfig(('single', 'test1'), init_kwargs={"a": 10, "b": 'line'})

    from torch import tensor
    x = tensor([[0, 0], [1, 0], [1, 0]])
    edge_index = tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    y = tensor([0, 1, 1])

    # Single
    data_list = [Data(x=x, edge_index=edge_index, y=y)]
    tmp_dir = tmp_dir(Path('.')).tmp_dir
    dataset = UserLocalDataset(tmp_dir / 'test_dataset_single', data_list)
    gen_dataset = DatasetManager.register_torch_geometric_local(dataset)

    gen_dataset._compute_dataset_data()
    print(json.dumps(gen_dataset.dataset_data, indent=1))
