Добро пожаловать в документацию GNN-AID!
=========================================

**GNN-AID** — это открытый фреймворк для визуализации, анализа, интерпретации графовых нейронных сетей с возможность применять атаки и защиты.
Фреймворк построен на базе `PyTorch-Geometric <https://pytorch-geometric.readthedocs.io>`_ и:

- включает готовые `датасеты из PyG <https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/data_cheatsheet.html>`_
- расширяемый API для пользовательских архитектур на уровне датасетов, моделей, методов интерпретации, атак и защит
- MLOps техники для воспроизводимых экспериментов

GNN-AID имеет веб-интерфейс, который поддерживает:

- интерактивную визуализацию графов
- конструктор модели "без кода"
- визуализация объяснений решения модели
- визуализация результатов атак на модель


.. Check out :doc:`installation` of the project.

.. .. note::

..    Проект находится в разработке.


.. toctree::
    :maxdepth: 1
    :caption: Начало

    getting_started

.. toctree::
    :maxdepth: 1
    :caption: Примеры

    tutorials

.. toctree::
    :maxdepth: 1
    :caption: Руководство пользователя

    user_guide/pipeline
    user_guide/backend
    user_guide/frontend
    user_guide/datasets
    user_guide/interpretation
    user_guide/attack
    user_guide/defense

.. toctree::
    :maxdepth: 1
    :caption: Руководство разработчика

    dev_guide/arch_back.rst
    dev_guide/arch_front.rst
    dev_guide/extend.rst
    dev_guide/code_style.rst

.. toctree::
    :maxdepth: 1
    :caption: Обзор пакетов

    api/aux
    api/data_structures
    api/datasets
    api/models_builder
    api/explainers
    api/attacks
    api/defenses
    api/user_datasets
    api/web_interface

