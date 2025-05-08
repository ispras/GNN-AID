import json
import shutil
import os
from pathlib import Path
from typing import Union

import torch_geometric
from torch_geometric.data import Dataset, InMemoryDataset

from aux.configs import DatasetConfig, DatasetVarConfig
from aux.custom_decorators import timing_decorator
from aux.declaration import Declare
from aux.utils import tmp_dir
from base.gen_dataset import DatasetInfo, GeneralDataset


class DatasetManager:
    """
    Class for working with datasets. Methods: get - loads dataset in torch_geometric format for gnn
    along the path specified in full_name as tuple. Currently also supports automatic loading and
    processing of all datasets from torch_geometric.datasets
    """

    @staticmethod
    def register_torch_geometric_local(
            dataset: InMemoryDataset,
            name: str = None
    ) -> GeneralDataset:
        """
        Save a given PTG dataset locally.
        Dataset is then always available for use by its config.

        :param dataset: torch_geometric.data.Dataset object.
        :param name: user defined dataset name. If not specified, a timestamp will be used.
        :return: GeneralDataset
        """
        gen_dataset = DatasetManager._register_torch_geometric(
            dataset, name=name, group='local', exists_ok=False, copy_data=True)

        return gen_dataset

    # QUE Misha, Kirill - can we use get_by_config always instead of it?
    @staticmethod
    @timing_decorator
    def get_by_config(
            dataset_config: DatasetConfig,
            dataset_var_config: DatasetVarConfig = None
    ) -> GeneralDataset:
        """ Get GeneralDataset by dataset config. Used from the frontend.
        """
        dataset_group = dataset_config.group
        # TODO misha - better make a more hierarchical grouping?
        if dataset_group in ["custom"]:
            from base.custom_datasets import CustomDataset
            gen_dataset = CustomDataset(dataset_config)

        elif dataset_group in ["hetero"]:
            # TODO misha - it is a kind of custom?
            from base.heterographs import CustomHeteroDataset
            gen_dataset = CustomHeteroDataset(dataset_config)

        elif dataset_group in ["vk_samples"]:
            # TODO misha - it is a kind of custom?
            from base.vk_datasets import VKDataset
            gen_dataset = VKDataset(dataset_config)

        else:
            from base.ptg_datasets import PTGDataset
            gen_dataset = PTGDataset(dataset_config)

        if dataset_var_config is not None:
            gen_dataset.build(dataset_var_config)

        return gen_dataset

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
        from base.ptg_datasets import PTGDataset
        dataset = DatasetManager.get_by_config(DatasetConfig(full_name))
        cfg = PTGDataset.dataset_var_config.to_saveable_dict()
        cfg.update(**kwargs)
        dataset_var_config = DatasetVarConfig(**cfg)
        dataset.build(dataset_var_config=dataset_var_config)
        dataset.train_test_split(percent_train_class=kwargs.get("percent_train_class", 0.8),
                                 percent_test_class=kwargs.get("percent_test_class", 0.2))
        # IMP Kirill suggest to return only dataset, else is its parts
        return dataset, dataset.data, dataset.results_dir

    @staticmethod
    def register_torch_geometric_api(
            dataset: Dataset,
            name: str = None,
            obj_name: str = 'DATASET_TO_EXPORT'
    ) -> GeneralDataset:
        """
        Register a user defined code implementing a PTG dataset.
        This function should be called at each framework run to make the dataset available for use.

        :param dataset: torch_geometric.data.Dataset object.
        :param name: user defined dataset name. If not specified, a timestamp will be used.
        :param obj_name: variable name to locate when import. DATASET_TO_EXPORT is default
        :return: GeneralDataset
        """
        gen_dataset = DatasetManager._register_torch_geometric(
            dataset, name=name, group='api', exists_ok=False, copy_data=False)

        # Save api info
        import inspect
        import_path = Path(inspect.getfile(dataset.__class__))
        api = {
            'import_path': str(import_path),
            'obj_name': obj_name,
        }
        with gen_dataset.api_path.open('w') as f:
            json.dump(api, f, indent=1)

        return gen_dataset

    @staticmethod
    def register_custom(
            dataset_config: DatasetConfig,
            format: str = 'ij',
            default_node_attr_value: dict = None,
            default_edge_attr_value: dict = None,
    ) -> GeneralDataset:
        """
        Create GeneralDataset from user created files in one of the supported formats.
        Attribute files created by user (in <name>.node_attributes and <name>.edge_attributes) have
        priority over attributes extracted from the raw graph file (<name>.<format>).

        :param dataset_config: config for a new dataset. Files will be searched for in the folder
         defined by this config.
        :param format: one of the supported formats.
        :param default_node_attr_value: dict with default node attributes values to apply where
         missing.
        :param default_edge_attr_value: dict with default edge attributes values to apply where
         missing.
        :return: CustomDataset
        """
        # Create empty CustomDataset
        from base.custom_datasets import CustomDataset
        gen_dataset = CustomDataset(dataset_config)

        # Look for obligate files: .info, graph(s), a dir with labels
        # info_file = None
        label_dir = None
        graph_files = []
        path = gen_dataset.raw_dir
        for p in path.iterdir():
            # if p.is_file() and p.name == '.info':
            #     info_file = p
            if p.is_file() and p.name.endswith(f'.{format}'):
                graph_files.append(p)
            if p.is_dir() and p.name.endswith('.labels'):
                label_dir = p
        # if info_file is None:
        #     raise RuntimeError(f"No .info file was found at {path}")
        if len(graph_files) == 0:
            raise RuntimeError(f"No files with extension '.{format}' found at {path}. "
                               f"If your graph is heterograph, use 'register_custom_hetero()'")
        if label_dir is None:
            raise RuntimeError(f"No file with extension '.label' found at {path}")

        # Order of files is important, should be consistent with .info, we suppose they are sorted
        graph_files = sorted(graph_files)

        # Create a temporary dir to store converted data
        with tmp_dir(path) as tmp:
            # Convert the data if necessary, write it to an empty directory
            if format != 'ij':
                from base.dataset_converter import DatasetConverter
                DatasetConverter.format_to_ij(gen_dataset.info, graph_files, format, tmp,
                                              default_node_attr_value, default_edge_attr_value)

            # Move or copy original contents to a temporary dir
            merge_directories(path, tmp, True)

            # Rename the newly created dir to the original one
            tmp.rename(path)

        # Check that data is valid
        gen_dataset.check_validity()

        return gen_dataset

    @staticmethod
    def register_custom_hetero(
            dataset_config: DatasetConfig,
    ) -> GeneralDataset:
        """
        Create heterogeneous GeneralDataset from user created files in 'ij' formats.
        Only basic 'ij' format is supported. No conversion.

        :param dataset_config: config for a new dataset. Files will be searched for in the folder
         defined by this config.
        :return: CustomHeteroDataset
        """
        # Create empty CustomDataset
        from base.heterographs import CustomHeteroDataset
        gen_dataset = CustomHeteroDataset(dataset_config)

        # Check that data is valid
        gen_dataset.check_validity()

        return gen_dataset

    @staticmethod
    def _register_torch_geometric(
            dataset: Dataset,
            name: Union[str, None] = None,
            group: str = None,
            exists_ok: bool = False,
            copy_data: bool = False
    ) -> GeneralDataset:
        """
        Create GeneralDataset from an externally specified torch geometric dataset.

        :param dataset: torch_geometric.data.Dataset object.
        :param name: dataset name.
        :param group: group name, preferred options are: 'local', 'exported', etc.
        :param exists_ok: if True raise Exception if graph with same name exists, otherwise the data
         will be overwritten.
        :param copy_data: if True processed data will be copied, otherwise a symbolic link is
         created.
        :return: GeneralDataset
        """
        info = DatasetInfo.induce(dataset)
        if name is None:
            import time
            info.name = 'graph_' + str(time.time())
        else:
            assert isinstance(name, str) and os.sep not in name
            info.name = name

        # Define graph configuration
        dataset_config = DatasetConfig(
            domain="single-graph" if info.count == 1 else "multiple-graphs",
            group=group or 'exported', graph=info.name
        )

        # Check if exists
        root_dir, files_paths = Declare.dataset_root_dir(dataset_config)
        if root_dir.exists():
            if exists_ok:
                shutil.rmtree(root_dir)
            else:
                raise FileExistsError(
                    f"Graph with config {dataset_config.full_name} already exists!")

        from base.ptg_datasets import PTGDataset
        gen_dataset = PTGDataset(dataset_config=dataset_config)
        info.save(gen_dataset.info_path)

        # Link or copy original contents to our path
        results_dir = gen_dataset.results_dir
        results_dir.parent.mkdir(parents=True, exist_ok=True)
        if copy_data:
            shutil.copytree(os.path.abspath(dataset.processed_dir), results_dir,
                            dirs_exist_ok=True)
        else:  # Create symlink
            # FIXME what will happen if we modify graph and its data.pt ?
            results_dir.symlink_to(os.path.abspath(dataset.processed_dir),
                                   target_is_directory=True)

        gen_dataset.dataset = dataset
        gen_dataset.info = info
        print(f"Registered graph '{info.name}' as {dataset_config.full_name}")
        return gen_dataset


def merge_directories(
        source_dir: Union[Path, str],
        destination_dir: Union[Path, str],
        remove_source: bool = False
) -> None:
    """
    Merge source directory into destination directory, replacing existing files.

    :param source_dir: Path to the source directory to be merged
    :param destination_dir: Path to the destination directory
    :param remove_source: if True, remove source directory (empty folders)
    """
    for root, _, files in os.walk(source_dir):
        # Calculate relative path
        relative_path = os.path.relpath(root, source_dir)

        # Create destination path
        dest_path = os.path.join(destination_dir, relative_path)
        os.makedirs(dest_path, exist_ok=True)

        # Move files
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_path, file)
            shutil.move(src_file, dest_file)

    if remove_source:
        shutil.rmtree(source_dir)


if __name__ == '__main__':
    # TODO remove
    print("test configs")

    # DataInfo.refresh_data_dir_structure()

    # class UserLocalDataset(InMemoryDataset):
    #     def __init__(self, root, data_list, transform=None):
    #         self.data_list = data_list
    #         super().__init__(root, transform)
    #         # NOTE: it is important to define self.slices here, since it is used to calculate len()
    #         self.data, self.slices = torch.load(self.processed_paths[0])
    #
    #     @property
    #     def processed_file_names(self):
    #         return 'data.pt'
    #
    #     def process(self):
    #         torch.save(self.collate(self.data_list), self.processed_paths[0])
    #
    #
    # dc = DatasetConfig(('single', 'test1'), init_kwargs={"a": 10, "b": 'line'})
    #
    # from torch import tensor
    # x = tensor([[0, 0], [1, 0], [1, 0]])
    # edge_index = tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    # y = tensor([0, 1, 1])
    #
    # # Single
    # data_list = [Data(x=x, edge_index=edge_index, y=y)]
    # dataset = UserLocalDataset(tmp_dir / 'test_dataset_single', data_list)
    # gen_dataset = DatasetManager.register_torch_geometric_local(dataset)
    #
    # gen_dataset._compute_dataset_data()
    # print(json.dumps(gen_dataset.dataset_data, indent=1))