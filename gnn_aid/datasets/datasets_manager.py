import json

from gnn_aid.aux.declaration import Declare
from gnn_aid.aux.utils import import_by_name
from gnn_aid.data_structures.configs import DatasetConfig, DatasetVarConfig, Task
from .dataset_info import DatasetInfo
from .gen_dataset import GeneralDataset
from .ptg_datasets import LibPTGDataset


class DatasetManager:
    """
    Provides methods for loading and managing datasets in torch_geometric format.

    Supports automatic loading of datasets by config, including all datasets from
    torch_geometric.datasets via LibPTGDataset.
    """

    @staticmethod
    def get_by_config(
            dataset_config: DatasetConfig,
            dataset_var_config: DatasetVarConfig = None,
            **params
    ) -> GeneralDataset:
        """
        Load a GeneralDataset by its config. Convenient to use from the frontend.

        Args:
            dataset_config (DatasetConfig): Config identifying the dataset location and type.
            dataset_var_config (DatasetVarConfig): Optional config for building features and labels.
            **params: Additional parameters forwarded to the dataset class constructor.

        Returns:
            Loaded GeneralDataset, built with dataset_var_config if provided.
        """
        path = Declare.dataset_info_path(dataset_config)

        # Check special cases when there is no metainfo file but we know where to get class
        if not path.exists():
            if dataset_config.full_name[0] == LibPTGDataset.data_folder:
                class_name = LibPTGDataset.__name__
                import_from = LibPTGDataset.__module__
            else:
                raise RuntimeError(f"No metainfo file found at '{path}'.")

        else:
            # Read metainfo
            info = DatasetInfo.read(path)
            class_name = info.class_name
            import_from = info.import_from
            if class_name is None or import_from is None:
                raise RuntimeError(f"Metainfo file does not contain field 'class_name' or 'import_from'."
                                   f" They must be specified in metainfo file, check it {path}")

        klass = import_by_name(class_name, [import_from])
        dataset = klass(dataset_config=dataset_config, **params)

        # Build dataset
        if dataset_var_config:
            dataset.build(dataset_var_config)

        return dataset

    @staticmethod
    def add_labeling(
            dataset_config: DatasetConfig,
            task: Task,
            labeling_name: str,
            labeling_dict: dict,
            value: int | list | None = None,
            force_rewrite=False
    ) -> None:
        """
        Add a new labeling to the dataset identified by dataset_config.

        Args:
            dataset_config (DatasetConfig): Config identifying the target dataset.
            task (Task): Task type for the new labeling.
            labeling_name (str): Name for the new labeling.
            labeling_dict (dict): Labels as {node/edge/graph_id → value}.
            value (int | list | None): Possible values depending on the task: number of classes
                for classification, or [min, max] bounds for regression. Induced if omitted.
            force_rewrite (bool): If True, overwrite an existing labeling with the same name.
        """
        info = DatasetInfo.read(Declare.dataset_info_path(dataset_config))

        if task in info.labelings and labeling_name in info.labelings[task] and not force_rewrite:
            raise NameError(f"Labeling '{labeling_name}' for task {task} already exists for dataset"
                            f" {dataset_config}.")

        # Some checks
        if task.is_node_level():
            assert info.count == 1
            assert len(labeling_dict) == info.nodes[0]
        elif task.is_edge_level():
            assert info.count == 1
        elif task.is_graph_level():
            assert len(labeling_dict) == info.count
        else:
            raise ValueError(f"Adding labelings for task {task} is not supported.")

        if value is None:
            if task.is_classification():
                value = max(labeling_dict.values()) + 1
            if task.is_regression():
                value = [min(labeling_dict.values()), max(labeling_dict.values())]

        # Save labeling_dict to file and update metainfo
        path = Declare.dataset_root_dir(dataset_config)[0] / 'raw' / 'labels' / task / labeling_name
        path.parent.mkdir(parents=True)
        with open(path, 'w') as f:
            json.dump(labeling_dict, f, indent=1)

        if task not in info.labelings:
            info.labelings[task] = {}
        info.labelings[task][labeling_name] = value
        info.save(Declare.dataset_info_path(dataset_config))
