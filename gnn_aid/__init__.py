""" Just a docstring """

__version__ = ""

import gnn_aid.auxil
import gnn_aid.data_structures
import gnn_aid.datasets
import gnn_aid.models_builder
import gnn_aid.explainers
import gnn_aid.attacks
import gnn_aid.defenses

# Suppress some warnings
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Using .* without a 'pyg-lib' installation is deprecated.*",
    category=UserWarning,
    module=r"torch_geometric\.sampler\.neighbor_sampler",
)
