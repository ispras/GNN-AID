from data_structures.configs import DatasetConfig
from datasets.ptg_datasets import PTGDataset
from user_datasets.power_graph.powergrid import PowerGrid


class PowerGraphDataset(
    PTGDataset
):
    """
    PowerGraph dataset from https://github.com/PowerGraph-Datasets/PowerGraph-Graph

    .. code-block:: python

        dc = DatasetConfig(('example', 'custom', 'powergraph'), {'name': 'uk'})
        dataset = PowerGraphDataset(dc)

        dataset.set_visible_part({'center': 0, 'depth': 0})
        print(dataset.visible_part.get_dataset_data())
        >>> DatasetData[
        >>>  edges: [[(0, 1), (0, 1), (0, 2), (0, 2), (1, 2), (1, 3), (1, 3), (2, 3), (2, 3), (3, 4), (3, 4), (3, 5), (3, 5), (3, 6), (...]]
        >>>  nodes: [29]
        >>>  graphs: [0]
        >>>  node_attributes: {'unknown': {0: [[0.12097656726837158, -0.14951694011688232, 0.06792455166578293], [-0.040750112384557724, -0.26544...}}
        >>> ]
    """
    def _define_ptg_dataset(
            self
    ) -> None:
        """ Build graph(s) structure - edge index
        """
        # TODO misha this is temporary until init_kwargs is fully implemented at front
        default_init_kwargs = {'name': 'uk'}
        default_init_kwargs.update(self.dataset_config.init_kwargs)

        # Creates or loads the ptg dataset
        self.dataset = PowerGrid(root=str(self.raw_dir),
                                 **default_init_kwargs)


if __name__ == '__main__':

    dc = DatasetConfig(('example', 'custom', 'powergraph'), {'name': 'uk'})
    dataset = PowerGraphDataset(dc)

    dataset.set_visible_part({'center': 0, 'depth': 0})
    dd = dataset.visible_part.get_dataset_data()
    print(dd)

    dvd = dataset.visible_part.get_dataset_var_data()
    print(dvd)
