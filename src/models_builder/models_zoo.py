from datasets.gen_dataset import GeneralDataset
from models_builder.gnn_constructor import FrameworkGNNConstructor
from data_structures.configs import ModelConfig, ModelStructureConfig


def model_configs_zoo(
        dataset: GeneralDataset,
        model_name: str
):
    gin_gin = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': dataset.num_node_features,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': dataset.num_classes,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'LogSoftmax',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                    },
                ]
            )
        )
    )

    gat_gin_lin = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GATConv',
                            'layer_kwargs': {
                                'in_channels': dataset.num_node_features,
                                'out_channels': 16,
                                'heads': 3,
                            },
                        },
                        'batchNorm': {
                            'batchNorm_name': 'BatchNorm1d',
                            'batchNorm_kwargs': {
                                'num_features': 48,
                                'eps': 1e-05,
                            }
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                        'dropout': {
                            'dropout_name': 'Dropout',
                            'dropout_kwargs': {
                                'p': 0.5,
                            }
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 48,
                                            'out_features': 48,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 48,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 48,
                                            'out_features': 48,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 48,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'Linear',
                            'layer_kwargs': {
                                'in_features': 48,
                                'out_features': dataset.num_classes,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },

                ]
            )
        )
    )

    sage_sage = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'SAGEConv',
                            'layer_kwargs': {
                                'in_channels': dataset.num_node_features,
                                'out_channels': 16,
                            },
                        },
                        'batchNorm': {
                            'batchNorm_name': 'BatchNorm1d',
                            'batchNorm_kwargs': {
                                'num_features': 16,
                                'eps': 1e-05,
                            }
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'SAGEConv',
                            'layer_kwargs': {
                                'in_channels': 16,
                                'out_channels': dataset.num_classes,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },
                ]
            )
        )
    )

    sage_sage_sage = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'SAGEConv',
                            'layer_kwargs': {
                                'in_channels': dataset.num_node_features,
                                'out_channels': 16,
                            },
                        },
                        'batchNorm': {
                            'batchNorm_name': 'BatchNorm1d',
                            'batchNorm_kwargs': {
                                'num_features': 16,
                                'eps': 1e-05,
                            }
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'SAGEConv',
                            'layer_kwargs': {
                                'in_channels': 16,
                                'out_channels': 16,
                            },
                        },
                        'batchNorm': {
                            'batchNorm_name': 'BatchNorm1d',
                            'batchNorm_kwargs': {
                                'num_features': 16,
                                'eps': 1e-05,
                            }
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'SAGEConv',
                            'layer_kwargs': {
                                'in_channels': 16,
                                'out_channels': dataset.num_classes,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },
                ]
            )
        )
    )

    gat_gat = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GATConv',
                            'layer_kwargs': {
                                'in_channels': dataset.num_node_features,
                                'out_channels': 16,
                                'heads': 3,
                            },
                        },
                        'batchNorm': {
                            'batchNorm_name': 'BatchNorm1d',
                            'batchNorm_kwargs': {
                                'num_features': 48,
                                'eps': 1e-05,
                            }
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GATConv',
                            'layer_kwargs': {
                                'in_channels': 48,
                                'out_channels': dataset.num_classes,
                                'heads': 1,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },
                ]
            )
        )
    )

    gat_gcn_sage_gcn_gcn = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GATConv',
                            'layer_kwargs': {
                                'in_channels': dataset.num_node_features,
                                'out_channels': 16,
                                'heads': 3,
                            },
                        },
                        'batchNorm': {
                            'batchNorm_name': 'BatchNorm1d',
                            'batchNorm_kwargs': {
                                'num_features': 48,
                                'eps': 1e-05,
                            }
                        },
                        'activation': {
                            'activation_name': 'LeakyReLU',
                            'activation_kwargs': {
                                "negative_slope": 0.01
                            },
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': 48,
                                'out_channels': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'SAGEConv',
                            'layer_kwargs': {
                                'in_channels': 16,
                                'out_channels': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'Tanh',
                            'activation_kwargs': None,
                        },
                        'dropout': {
                            'dropout_name': 'Dropout',
                            'dropout_kwargs': {
                                'p': 0.5,
                            }
                        }
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': 16,
                                'out_channels': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'Sigmoid',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': 16,
                                'out_channels': dataset.num_classes,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },
                ]
            )
        )
    )

    test_gnn = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'CGConv',
                            'layer_kwargs': {
                                'channels': dataset.num_node_features
                                # 'in_channels': dataset.num_node_features,
                                # 'dim': 2,
                                # 'kernel_size': 2,
                                # 'out_channels': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'Linear',
                            'layer_kwargs': {
                                'in_features': dataset.num_node_features,
                                'out_features': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'CGConv',
                            'layer_kwargs': {
                                'channels': 16
                                # 'in_channels': 16,
                                # 'dim': 2,
                                # 'kernel_size': 2,
                                # 'out_channels': dataset.num_classes,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'Linear',
                            'layer_kwargs': {
                                'in_features': 16,
                                'out_features': dataset.num_classes,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },
                ]
            )
        )
    )

    gcn_gcn_xor_task = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        "label": "n",
                        "layer": {
                            "layer_name": "GCNConv",
                            "layer_kwargs": {
                                "in_channels": 1,
                                "out_channels": 3,
                                "aggr": "add",
                                "improved": False,
                                "add_self_loops": False,
                                "normalize": False,
                                "bias": True
                            }
                        },
                        "connections": []
                    },
                    {
                        "label": "n",
                        "layer": {
                            "layer_name": "GCNConv",
                            "layer_kwargs": {
                                "in_channels": 3,
                                "out_channels": 2,
                                "aggr": "add",
                                "improved": False,
                                "add_self_loops": False,
                                "normalize": False,
                                "bias": True
                            }
                        },
                        "activation": {
                            "activation_name": "LogSoftmax",
                            "activation_kwargs": {}
                        },
                        "connections": []
                    }
                ]
            )
        )
    )

    gcn_gcn = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': dataset.num_node_features,
                                'out_channels': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': 16,
                                'out_channels': dataset.num_classes,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },
                ]
            )
        )
    )

    gcn_gcn_gcn = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': dataset.num_node_features,
                                'out_channels': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': 16,
                                'out_channels': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': 16,
                                'out_channels': dataset.num_classes,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },
                ]
            )
        )
    )

    gcn_gcn_no_self_loops = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': dataset.num_node_features,
                                'out_channels': 16,
                                'add_self_loops': False
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': 16,
                                'out_channels': dataset.num_classes,
                                'add_self_loops': False
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },
                ]
            )
        )
    )

    gcn_gcn_linearized = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': dataset.num_node_features,
                                'out_channels': 16,
                                'bias': True,
                            },
                        },
                        # 'activation': {
                        #     'activation_name': 'ReLU',
                        #     'activation_kwargs': None,
                        # },
                        'dropout': {
                            'dropout_name': 'Dropout',
                            'dropout_kwargs': {
                                'p': 0.5,
                            }
                        }
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': 16,
                                'out_channels': dataset.num_classes,
                                'bias': True,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },
                ]
            )
        )
    )

    gcn = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        "label": "n",
                        "layer": {
                            "layer_name": "GCNConv",
                            "layer_kwargs": {
                                "in_channels": dataset.num_node_features,
                                "out_channels": dataset.num_classes,
                                "aggr": "add",
                                "improved": False,
                                "add_self_loops": True,
                                "normalize": True,
                                "bias": True
                            }
                        },
                        "activation": {
                            "activation_name": "LogSoftmax",
                            "activation_kwargs": {}
                        },
                        "connections": []
                    },
                ]
            )
        )
    )

    gcn_test = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        "label": "n",
                        "layer": {
                            "layer_name": "GCNConv",
                            "layer_kwargs": {
                                "in_channels": dataset.num_node_features,
                                "out_channels": dataset.num_classes,
                                "aggr": "add",
                                "improved": False,
                                "add_self_loops": False,
                                "normalize": False,
                                "bias": False
                            }
                        },
                        "activation": {
                            "activation_name": "LogSoftmax",
                            "activation_kwargs": {}
                        },
                        "connections": []
                    },
                ]
            )
        )
    )

    gcn_gcn_lin = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': dataset.num_node_features,
                                'out_channels': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                        'connections': [
                            {
                                'into_layer': 2,
                                'connection_kwargs': {
                                    'aggregation_type': 'cat',
                                },
                            },
                        ],
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': 16,
                                'out_channels': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'Linear',
                            'layer_kwargs': {
                                'in_features': 16 * 2,
                                'out_features': dataset.num_classes,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },

                ]
            )
        )
    )

    gin_gin_gin_lin_lin = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': dataset.num_node_features,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                        'connections': [
                            {
                                'into_layer': 3,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'cat',
                                },
                            },
                        ],
                    },

                    {
                        'label': 'g',
                        'layer': {
                            'layer_name': 'Linear',
                            'layer_kwargs': {
                                'in_features': 16,
                                'out_features': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                        'dropout': {
                            'dropout_name': 'Dropout',
                            'dropout_kwargs': {
                                'p': 0.5,
                            }
                        },
                    },

                    {
                        'label': 'g',
                        'layer': {
                            'layer_name': 'Linear',
                            'layer_kwargs': {
                                'in_features': 16,
                                'out_features': dataset.num_classes,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },

                ]
            )
        )
    )

    gin_gin_gin_lin_lin_con = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': dataset.num_node_features,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                        'connections': [
                            {
                                'into_layer': 3,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'cat',
                                },
                            },
                        ],
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                        'connections': [
                            {
                                'into_layer': 3,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'cat',
                                },
                            },
                        ],
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                        'connections': [
                            {
                                'into_layer': 3,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'cat',
                                },
                            },
                        ],
                    },

                    {
                        'label': 'g',
                        'layer': {
                            'layer_name': 'Linear',
                            'layer_kwargs': {
                                'in_features': 16 * 3,
                                'out_features': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                        'dropout': {
                            'dropout_name': 'Dropout',
                            'dropout_kwargs': {
                                'p': 0.5,
                            }
                        },
                    },

                    {
                        'label': 'g',
                        'layer': {
                            'layer_name': 'Linear',
                            'layer_kwargs': {
                                'in_features': 16,
                                'out_features': dataset.num_classes,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },

                ]
            )
        )
    )

    # FIXME tmp - assert (self.num_prototypes % num_classes == 0) fails
    gin_gin_gin_lin_lin_prot = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': dataset.num_node_features,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                        'connections': [
                            {
                                'into_layer': 3,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'cat',
                                },
                            },
                        ],
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                        'connections': [
                            {
                                'into_layer': 3,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'cat',
                                },
                            },
                        ],
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                        'connections': [
                            {
                                'into_layer': 3,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'cat',
                                },
                            },
                        ],
                    },

                    {
                        'label': 'g',
                        'layer': {
                            'layer_name': 'Linear',
                            'layer_kwargs': {
                                'in_features': 16 * 3,
                                'out_features': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                        'dropout': {
                            'dropout_name': 'Dropout',
                            'dropout_kwargs': {
                                'p': 0.5,
                            }
                        },
                    },

                    {
                        'label': 'g',
                        'layer': {
                            'layer_name': 'Linear',
                            'layer_kwargs': {
                                'in_features': 16,
                                'out_features': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'g',
                        'layer': {
                            'layer_name': 'Prot',
                            'layer_kwargs': {
                                'in_features': 16,
                                'num_prototypes_per_class': 3,
                                'num_classes': dataset.num_classes,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },

                ]
            )
        )
    )

    dummy_gin_gin_gin_gsat_lin_lin = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'DummyLayer',
                            'layer_kwargs': None,
                        },
                        'connections': [
                            {
                                'into_layer': 4,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'stack',
                                },
                            },
                        ],
                    },
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': dataset.num_node_features,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                        'connections': [
                            {
                                'into_layer': 5,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'cat',
                                },
                            },
                        ],
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                        'connections': [
                            {
                                'into_layer': 5,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'cat',
                                },
                            },
                        ],
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                        'connections': [
                            {
                                'into_layer': 5,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'cat',
                                },
                            },
                        ],
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GSAT',
                            'layer_kwargs': {
                                'in_features': 16,
                            },
                        },
                        'connections': [
                            {
                                'into_layer': 5,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'cat',
                                },
                            },
                        ],
                    },

                    {
                        'label': 'g',
                        'layer': {
                            'layer_name': 'Linear',
                            'layer_kwargs': {
                                'in_features': 16 * 4,
                                'out_features': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                        'dropout': {
                            'dropout_name': 'Dropout',
                            'dropout_kwargs': {
                                'p': 0.5,
                            }
                        },
                    },

                    {
                        'label': 'g',
                        'layer': {
                            'layer_name': 'Linear',
                            'layer_kwargs': {
                                'in_features': 16,
                                'out_features': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },
                ]
            )
        )
    )

    dummy_gcn_gcn_gsat = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'DummyLayer',
                            'layer_kwargs': None,
                        },
                        'connections': [
                            {
                                'into_layer': 3,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'stack',
                                },
                            },
                        ],
                    },
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': dataset.num_node_features,
                                'out_channels': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': 16,
                                'out_channels': dataset.num_classes,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GSAT',
                            'layer_kwargs': {
                                'in_features': dataset.num_classes,
                            },
                        },
                    },
                ]
            )
        )
    )

    dummy_gcn_gcn_gcn_gsat = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'DummyLayer',
                            'layer_kwargs': None,
                        },
                        'connections': [
                            {
                                'into_layer': 4,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'stack',
                                },
                            },
                        ],
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': dataset.num_node_features,
                                'out_channels': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': 16,
                                'out_channels': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GCNConv',
                            'layer_kwargs': {
                                'in_channels': 16,
                                'out_channels': dataset.num_classes,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GSAT',
                            'layer_kwargs': {
                                'in_features': dataset.num_classes,
                            },
                        },
                    },
                ]
            )
        )
    )

    dummy_sage_sage_gsat = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'DummyLayer',
                            'layer_kwargs': None,
                        },
                        'connections': [
                            {
                                'into_layer': 3,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'stack',
                                },
                            },
                        ],
                    },
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'SAGEConv',
                            'layer_kwargs': {
                                'in_channels': dataset.num_node_features,
                                'out_channels': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'SAGEConv',
                            'layer_kwargs': {
                                'in_channels': 16,
                                'out_channels': dataset.num_classes,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GSAT',
                            'layer_kwargs': {
                                'in_features': dataset.num_classes,
                            },
                        },
                    },
                ]
            )
        )
    )

    dummy_sage_sage_sage_gsat = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'DummyLayer',
                            'layer_kwargs': None,
                        },
                        'connections': [
                            {
                                'into_layer': 4,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'stack',
                                },
                            },
                        ],
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'SAGEConv',
                            'layer_kwargs': {
                                'in_channels': dataset.num_node_features,
                                'out_channels': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'SAGEConv',
                            'layer_kwargs': {
                                'in_channels': 16,
                                'out_channels': 16,
                            },
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'SAGEConv',
                            'layer_kwargs': {
                                'in_channels': 16,
                                'out_channels': dataset.num_classes,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GSAT',
                            'layer_kwargs': {
                                'in_features': dataset.num_classes,
                            },
                        },
                    },
                ]
            )
        )
    )

    dummy_gin_gin_gsat = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'DummyLayer',
                            'layer_kwargs': None,
                        },
                        'connections': [
                            {
                                'into_layer': 3,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'stack',
                                },
                            },
                        ],
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': dataset.num_node_features,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': dataset.num_classes,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'LogSoftmax',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GSAT',
                            'layer_kwargs': {
                                'in_features': dataset.num_classes,
                            },
                        },
                    },
                ]
            )
        )
    )

    dummy_gat_gat_gsat = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'DummyLayer',
                            'layer_kwargs': None,
                        },
                        'connections': [
                            {
                                'into_layer': 3,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'stack',
                                },
                            },
                        ],
                    },
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GATConv',
                            'layer_kwargs': {
                                'in_channels': dataset.num_node_features,
                                'out_channels': 16,
                                'heads': 3,
                            },
                        },
                        'batchNorm': {
                            'batchNorm_name': 'BatchNorm1d',
                            'batchNorm_kwargs': {
                                'num_features': 48,
                                'eps': 1e-05,
                            }
                        },
                        'activation': {
                            'activation_name': 'ReLU',
                            'activation_kwargs': None,
                        },
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GATConv',
                            'layer_kwargs': {
                                'in_channels': 48,
                                'out_channels': dataset.num_classes,
                                'heads': 1,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GSAT',
                            'layer_kwargs': {
                                'in_features': dataset.num_classes,
                            },
                        },
                    },
                ]
            )
        )
    )

    gin_gin_gin_lin = FrameworkGNNConstructor(
        model_config=ModelConfig(
            structure=ModelStructureConfig(
                [
                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': dataset.num_node_features,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                        'connections': [
                            {
                                'into_layer': 3,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'cat',
                                },
                            },
                        ],
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                        'connections': [
                            {
                                'into_layer': 3,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'cat',
                                },
                            },
                        ],
                    },

                    {
                        'label': 'n',
                        'layer': {
                            'layer_name': 'GINConv',
                            'layer_kwargs': None,
                            'gin_seq': [
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'batchNorm': {
                                        'batchNorm_name': 'BatchNorm1d',
                                        'batchNorm_kwargs': {
                                            'num_features': 16,
                                            'eps': 1e-05,
                                        }
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                                {
                                    'layer': {
                                        'layer_name': 'Linear',
                                        'layer_kwargs': {
                                            'in_features': 16,
                                            'out_features': 16,
                                        },
                                    },
                                    'activation': {
                                        'activation_name': 'ReLU',
                                        'activation_kwargs': None,
                                    },
                                },
                            ],
                        },
                        'connections': [
                            {
                                'into_layer': 3,
                                'connection_kwargs': {
                                    'pool': {
                                        'pool_type': 'global_add_pool',
                                    },
                                    'aggregation_type': 'cat',
                                },
                            },
                        ],
                    },

                    {
                        'label': 'g',
                        'layer': {
                            'layer_name': 'Linear',
                            'layer_kwargs': {
                                'in_features': 16 * 3,
                                'out_features': dataset.num_classes,
                            },
                        },
                        'activation': {
                            'activation_name': 'LogSoftmax',
                            'activation_kwargs': None,
                        },
                    },

                ]
            )
        )
    )

    if model_name in locals():
        return locals()[model_name]
    else:
        raise Exception(f"{model_name} no in models zoo now. Make this model or use one of {locals().keys()}")
