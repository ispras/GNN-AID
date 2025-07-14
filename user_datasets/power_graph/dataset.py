from data_structures.configs import DatasetConfig
from datasets.ptg_datasets import PTGDataset
from user_datasets.power_graph.powergrid import PowerGrid


class PowerGraphDataset(
    PTGDataset
):
    """
    PowerGraph dataset from https://github.com/PowerGraph-Datasets/PowerGraph-Graph
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
