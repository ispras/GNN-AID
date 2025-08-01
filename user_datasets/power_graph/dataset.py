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
    
    from models_builder.models_zoo import model_configs_zoo
    from models_builder.gnn_models import ModelModificationConfig, ModelConfig, ConfigPattern, FrameworkGNNModelManager, Metric

    dataset.train_test_split(percent_train_class=0.6, percent_test_class=0.2)
    results_dataset_path = dataset.results_dir
    default_config = ModelModificationConfig(
        model_ver_ind=0,
    )

    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "mask_features": [],
            # "optimizer": {
            #     # "_config_class": "Config",
            #     "_class_name": "Adam",
            #     # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
            #     # "_class_import_info": ["torch.optim"],
            #     "_config_kwargs": {},
            # }
        }
    )

    gin3_lin2_mg_pg = model_configs_zoo(dataset=dataset,
                                           model_name='gin_gin_gin_lin_lin')

    gnn_mm_mg_small = FrameworkGNNModelManager(
        gnn=gin3_lin2_mg_pg,
        dataset_path=results_dataset_path,
        modification=default_config,
        manager_config=manager_config,
    )

    gnn_mm_mg_small.train_model(gen_dataset=dataset, steps=100,
                                metrics=[Metric("F1", mask='val'),
                                         Metric("F1", mask='test')])
    metric_loc = gnn_mm_mg_small.evaluate_model(
        gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro')])
    print(metric_loc)