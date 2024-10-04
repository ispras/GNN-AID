from models_builder.gnn_constructor import FrameworkGNNConstructor
from aux.configs import ModelConfig, ModelStructureConfig


def model_configs_zoo(dataset, model_name):
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
                                'heads': 2,
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
