Welcome to GNN-AID documentation!
=================================

GNN-AID is an open framework for **A**\ nalysis, **I**\ nterpretation, and **D**\ efensing Graph Neural Networks.
It is built on `PyTorch-Geometric <https://pytorch-geometric.readthedocs.io>`_ and:

- includes preloaded `datasets from PyG <https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/data_cheatsheet.html>`_
- extendable API for custom architectures: datasets, models, explainers, attacks and defense methods
- MLOps features for experiments reproducibility

GNN-AID has a web interface which supports:

- graph visualization and analysis tools
- no-code model building
- visualization of models explanations

.. Check out :doc:`installation` of the project.

.. .. note::

..    This project is under active development.


.. toctree::
    :maxdepth: 1
    :caption: Getting started

    getting_started

.. toctree::
    :maxdepth: 1
    :caption: Tutorials

    tutorials

.. toctree::
    :maxdepth: 1
    :caption: User guide

    user_guide/pipeline
    user_guide/backend
    user_guide/frontend
    user_guide/datasets
    user_guide/interpretation
    user_guide/attack
    user_guide/defense

.. toctree::
    :maxdepth: 1
    :caption: Developer guide

    dev_guide/arch_back.rst
    dev_guide/arch_front.rst
    dev_guide/extend.rst
    dev_guide/code_style.rst

.. toctree::
    :maxdepth: 1
    :caption: Package reference

    api/aux
    api/data_structures
    api/datasets
    api/models_builder
    api/explainers
    api/attacks
    api/defenses
    api/user_datasets

