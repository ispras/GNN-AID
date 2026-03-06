Методы интерпретации
********************

.. contents::
    :local:


Фреймворк **GNN-AID** предоставляет набор алгоритмов интерпретации графовых нейронных сетей (GNN), позволяющих объяснять предсказание модели и выявлять важные элементы графовой структуры.

По области применения методы интерпретации делятся на локальные и глобальные.

* **Локальные методы** объясняют конкретное предсказание модели. Например, они позволяют определить, какие элементы графа оказали наибольшее влияние на классификацию выбранной вершины.
* **Глобальные методы** формируют общее представление о том, как модель принимает решения. Они выявляют структурные закономерности или прототипы, характерные для определенных классов.

По уровню доступа к внутреннему устройству модели методы интерпретации делятся на методы черного и белого ящика.

* **Методы черного ящика** работают только с входными данными и предсказаниями модели и не используют внутреннюю информацию модели. Это позволяет применять их к моделям любой архитектуры.
* **Методы белого ящика** используют внутренние характеристики модели, например градиенты или скрытые представления. Это позволяет получать более точные объяснения, но требует доступа к модели.

Методы интерпретации могут выделять различные типы элементов графа: вершины, ребра, признаки вершин, подграфы.

В текущей версии фреймворка реализованы следующие методы постфактум-интерпретации:

* GNNExplainer
* PGExplainer
* PGMExplainer
* SubgraphX
* Zorro
* GraphMask

Помимо методов постфактум-интерпретации, фреймворк поддерживает **самоинтерпретируемые архитектуры**, в которых механизм объяснения встроен непосредственно в модель.

В фреймворке реализованы:

* ProtGNN
* NeuralAnalysis

Таблица методов интерпретации
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 18 18 28 18

   * - Метод
     - Тип объяснения
     - Доступ к модели
     - Поддерживаемые задачи
     - Элементы объяснения
   * - GNNExplainer
     - локальный
     - белый ящик
     - классификация графов, вершин, предсказание ребер
     - рёбра, признаки
   * - PGExplainer
     - локальный
     - белый ящик
     - классификация графов, вершин
     - рёбра
   * - PGMExplainer
     - локальный
     - чёрный ящик
     - классификация графов, вершин
     - вершины
   * - SubgraphX
     - локальный
     - чёрный ящик
     - классификация графов, вершин
     - подграф
   * - Zorro
     - локальный
     - чёрный ящик
     - классификация графов, вершин
     - вершины, признаки
   * - GraphMask
     - локальный
     - белый ящик
     - классификация вершин
     - рёбра
   * - ProtGNN
     - глобальный
     - встроено в модель
     - классификация графов
     - прототипы
   * - NeuralAnalysis
     - локальный
     - встроено в модель
     - классификация графов
     - логические выражения

GNNExplainer
------------

GNNExplainer из статьи `"GNNExplainer: Generating Explanations for Graph Neural Networks" <https://proceedings.neurips.cc/paper_files/paper/2019/file/d80b7040b773199015de6d3b4293c8ff-Paper.pdf>`_. Определяет подграф, который максимизирует взаимную информацию между предсказаниями модели и распределением возможных структур подграфов, путем обучения дифференцируемой маски по ребрам.

Вы можете задать конфигурацию метода GNNExplainer, указав нужные значения параметров в ``ExplainerInitConfig`` в поле ``_config_kwargs``. Метод GNNExplainer имеет следующие параметры:

* ``epochs`` — количество эпох оптимизации маски объяснения.
* ``lr`` — скорость обучения при оптимизации маски.
* ``node_mask_type`` — тип маски, применяемой к узлам графа.
* ``edge_mask_type`` — тип маски, применяемой к рёбрам графа.
* ``mode`` — тип задачи, для которой строится объяснение (бинарная, многоклассовая, регрессия).
* ``return_type`` — формат выхода модели (логарифмы вероятностей, вероятности или сырые логиты).
* ``edge_size`` — коэффициент регуляризации, контролирующий разреженность маски рёбер.
* ``edge_reduction`` — способ агрегации регуляризации для маски рёбер.
* ``node_feat_size`` — коэффициент регуляризации размера маски признаков узлов.
* ``node_feat_reduction`` — способ агрегации регуляризации для маски признаков узлов.
* ``edge_ent`` — коэффициент энтропийной регуляризации маски рёбер.
* ``node_feat_ent`` — коэффициент энтропийной регуляризации маски признаков узлов.
* ``EPS`` — малое значение для обеспечения численной стабильности вычислений.

Вы можете задать конфигурацию запуска метода GNNExplainer, указав нужные значения параметров в ``ExplainerRunConfig`` в поле ``_config_kwargs``. Метод GNNExplainer имеет следующие параметры запуска:

* ``element_idx`` — индекс элемента (узла или графа) или ребро, для которого строится объяснение.

Пример кода запуска GNNExplainer:

.. code-block:: python

   from gnn_aid.datasets import DatasetManager, LibPTGDataset
   from gnn_aid.data_structures import DatasetConfig, Task
   from gnn_aid.data_structures.configs import ConfigPattern
   from gnn_aid.models_builder import model_configs_zoo, Metric
   from gnn_aid.models_builder.model_managers import FrameworkGNNModelManager
   from gnn_aid.explainers import FrameworkExplainersManager
   from gnn_aid.aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH

   # Define a dataset
   gen_dataset = DatasetManager.get_by_config(
       DatasetConfig((LibPTGDataset.data_folder, "Homogeneous", "Planetoid", "Cora")),
       LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.NODE_CLASSIFICATION})
   )

   gen_dataset.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
   results_dataset_path = gen_dataset.prepared_dir

   # Define a model
   gnn = model_configs_zoo(dataset=gen_dataset, model_name='gcn_gcn')

   gnn_model_manager_config = ConfigPattern(
       _config_class="ModelManagerConfig",
       _config_kwargs={
           "batch": 10000,
           "mask_features": []
       }
   )
   gnn_model_manager = FrameworkGNNModelManager(
       gnn=gnn,
       dataset_path=results_dataset_path,
       manager_config=gnn_model_manager_config
   )

   # Train model
   gnn_model_manager.train_model(gen_dataset=gen_dataset, steps=100, save_model_flag=False,
                                 metrics=[Metric("F1", mask='test')])

   # Define an explainer
   explainer_init_config = ConfigPattern(
       _class_name="GNNExplainer(torch-geom)",
       _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
       _config_class="ExplainerInitConfig",
       _config_kwargs={
       }
   )
   explainer_run_config = ConfigPattern(
       _config_class="ExplainerRunConfig",
       _config_kwargs={
           "mode": "local",
           "kwargs": {
               "_class_name": "GNNExplainer(torch-geom)",
               "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
               "_config_class": "Config",
               "_config_kwargs": {
               },
           }
       }
   )
   explainer = FrameworkExplainersManager(
       init_config=explainer_init_config,
       dataset=gen_dataset, gnn_manager=gnn_model_manager,
       explainer_name='GNNExplainer(torch-geom)',
   )

   # Run explainer
   explanation = explainer.conduct_experiment(explainer_run_config)
   print(explanation)

Метод GNNExplainer в текущей реализации позволяет строить объяснение предсказания модели для задач классификации узлов, классификации графов и предсказания ребер.

PGExplainer
-----------

PGExplainer из статьи `"Parameterized Explainer for Graph Neural Network" <https://proceedings.neurips.cc/paper/2020/file/e37b08dd3015330dcbb5d6663667b8b8-Paper.pdf>`_. Использует параметрическую модель для идентификации влиятельных ребер и узлов путем изучения вероятностей их вклада в прогнозы. Он применяет пороговое значение на основе прогнозируемых вероятностей ребер, чтобы выделить наиболее значимые компоненты графа.

Вы можете задать конфигурацию метода PGExplainer, указав нужные значения параметров в ``ExplainerInitConfig`` в поле ``_config_kwargs``. Метод PGExplainer имеет следующие параметры:

* ``epochs`` — количество эпох оптимизации маски объяснения.
* ``lr`` — скорость обучения при оптимизации маски.

Вы можете задать конфигурацию запуска метода PGExplainer, указав нужные значения параметров в ``ExplainerRunConfig`` в поле ``_config_kwargs``. Метод PGExplainer имеет следующие параметры запуска:

* ``element_idx`` — индекс элемента (узла или графа), для которого строится объяснение.

Пример кода запуска PGExplainer:

.. code-block:: python

   from gnn_aid.datasets import DatasetManager, LibPTGDataset
   from gnn_aid.data_structures import DatasetConfig, Task, DatasetVarConfig
   from gnn_aid.data_structures.configs import ConfigPattern, FeatureConfig
   from gnn_aid.models_builder import model_configs_zoo, Metric
   from gnn_aid.models_builder.model_managers import FrameworkGNNModelManager
   from gnn_aid.explainers import FrameworkExplainersManager
   from gnn_aid.aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH


   # Defining a dataset
   gen_dataset = DatasetManager.get_by_config(
       DatasetConfig(("example", "example")),
       DatasetVarConfig(task=Task.NODE_CLASSIFICATION,
                        features=FeatureConfig(node_attr=['a']),
                        labeling='binary',
                        dataset_ver_ind=0)
   )
   gen_dataset.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
   results_dataset_path = gen_dataset.prepared_dir

   # Defining a model
   gnn = model_configs_zoo(dataset=gen_dataset, model_name='gcn_gcn')

   gnn_model_manager_config = ConfigPattern(
       _config_class="ModelManagerConfig",
       _config_kwargs={
           "batch": 10000,
           "mask_features": []
       }
   )
   gnn_model_manager = FrameworkGNNModelManager(
       gnn=gnn,
       dataset_path=results_dataset_path,
       manager_config=gnn_model_manager_config
   )

   # Train model
   gnn_model_manager.train_model(gen_dataset=gen_dataset, steps=100, save_model_flag=False,
                                 metrics=[Metric("F1", mask='test')])

   # Define an explainer
   explainer_init_config = ConfigPattern(
       _class_name="PGExplainer(dig)",
       _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
       _config_class="ExplainerInitConfig",
       _config_kwargs={
       }
   )
   explainer_run_config = ConfigPattern(
       _config_class="ExplainerRunConfig",
       _config_kwargs={
           "mode": "local",
           "kwargs": {
               "_class_name": "PGExplainer(dig)",
               "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
               "_config_class": "Config",
               "_config_kwargs": {

               },
           }
       }
   )
   explainer = FrameworkExplainersManager(
       init_config=explainer_init_config,
       dataset=gen_dataset, gnn_manager=gnn_model_manager,
       explainer_name='PGExplainer(dig)',
   )

   # Run explainer
   explanation = explainer.conduct_experiment(explainer_run_config)
   print(explanation)

Метод PGExplainer в текущей реализации позволяет строить объяснение предсказания модели для задач классификации узлов и классификации графов.

PGMExplainer
------------

PGMExplainer из статьи `"PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks" <https://arxiv.org/pdf/2010.05788>`_. Cтроит вероятностную модель для определения важных подграфов, влияющих на прогнозы конкретных узлов. Минимизируя расхождение в прогнозах между полным графом и подграфами, он оценивает важность узлов и ребер.

Вы можете задать конфигурацию метода PGMExplainer, указав нужные значения параметров в ``ExplainerInitConfig`` в поле ``_config_kwargs``. Метод PGMExplainer имеет следующие параметры:

* ``perturbation_mode`` — способ генерации возмущений признаков при оценке значимости узлов (случайные целые, среднее значение, нули, максимум или равномерное распределение).
* ``perturbations_is_positive_only`` — если ``True``, ограничивает возмущённые значения только положительными числами.
* ``is_perturbation_scaled`` — если ``True``, нормализует диапазон возмущённых признаков перед подачей в модель.
* ``num_samples`` — количество сгенерированных возмущённых выборок, используемых для статистической проверки влияния узлов на предсказание.
* ``max_subgraph_size`` — максимальное число соседних узлов, учитываемых при построении объясняющего подграфа.
* ``significance_threshold`` — порог статистической значимости (p-value), при котором узел считается влияющим на предсказание.
* ``pred_threshold`` — допустимое отклонение выхода модели (в диапазоне ``[0,1]``), при котором результат на возмущённых данных считается отличным от исходного.
* ``mode`` — тип задачи модели (бинарная или многоклассовая классификация).
* ``return_type`` — формат выхода модели (логарифмы вероятностей, вероятности или сырые логиты).

Вы можете задать конфигурацию запуска метода PGMExplainer, указав нужные значения параметров в ``ExplainerRunConfig`` в поле ``_config_kwargs``. Метод PGMExplainer имеет следующие параметры запуска:

* ``element_idx`` — индекс элемента (узла или графа), для которого строится объяснение.

Пример кода запуска PGMExplainer:

.. code-block:: python

   from gnn_aid.datasets import DatasetManager, LibPTGDataset
   from gnn_aid.data_structures import DatasetConfig, Task, DatasetVarConfig
   from gnn_aid.data_structures.configs import ConfigPattern, FeatureConfig
   from gnn_aid.models_builder import model_configs_zoo, Metric
   from gnn_aid.models_builder.model_managers import FrameworkGNNModelManager
   from gnn_aid.explainers import FrameworkExplainersManager
   from gnn_aid.aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH


   # Defining a dataset
   gen_dataset = DatasetManager.get_by_config(
       DatasetConfig((LibPTGDataset.data_folder, "Homogeneous", "Planetoid", "Cora")),
       LibPTGDataset.default_dataset_var_config.clone_with({"task": Task.NODE_CLASSIFICATION})
   )
   gen_dataset.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
   results_dataset_path = gen_dataset.prepared_dir

   # Defining a model
   gnn = model_configs_zoo(dataset=gen_dataset, model_name='gcn_gcn')

   gnn_model_manager_config = ConfigPattern(
       _config_class="ModelManagerConfig",
       _config_kwargs={
           "batch": 10000,
           "mask_features": []
       }
   )
   gnn_model_manager = FrameworkGNNModelManager(
       gnn=gnn,
       dataset_path=results_dataset_path,
       manager_config=gnn_model_manager_config
   )

   # Train model
   gnn_model_manager.train_model(gen_dataset=gen_dataset, steps=100, save_model_flag=False,
                                 metrics=[Metric("F1", mask='test')])

   # Define an explainer
   explainer_init_config = ConfigPattern(
       _class_name="PGMExplainer",
       _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
       _config_class="ExplainerInitConfig",
       _config_kwargs={
       }
   )
   explainer_run_config = ConfigPattern(
       _config_class="ExplainerRunConfig",
       _config_kwargs={
           "mode": "local",
           "kwargs": {
               "_class_name": "PGMExplainer",
               "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
               "_config_class": "Config",
               "_config_kwargs": {

               },
           }
       }
   )
   explainer = FrameworkExplainersManager(
       init_config=explainer_init_config,
       dataset=gen_dataset, gnn_manager=gnn_model_manager,
       explainer_name='PGMExplainer',
   )

   # Run explainer
   explanation = explainer.conduct_experiment(explainer_run_config)
   print(explanation)

Метод PGMExplainer в текущей реализации позволяет строить объяснение предсказания модели для задач классификации узлов и классификации графов.

SubgraphX
---------

SubgraphX из статьи `"On Explainability of Graph Neural Networks via Subgraph Explorations" <https://proceedings.mlr.press/v139/yuan21c.html>`_. Использует значение Шепли и поиск по дереву Монте-Карло (MCTS) для генерации связного подграфа, который наилучшим образом объясняет предсказания модели. Он оценивает важность подграфа на основе его вклада в выходные данные модели.

Вы можете задать конфигурацию метода SubgraphX, указав нужные значения параметров в ``ExplainerInitConfig`` в поле ``_config_kwargs``. Метод SubgraphX имеет следующие параметры:

* ``rollout`` — количество итераций поиска (rollout) в алгоритме Monte Carlo Tree Search для оценки важности подграфов.
* ``min_atoms`` — минимальное число узлов (атомов) в листовых вершинах дерева поиска.
* ``c_puct`` — коэффициент, управляющий балансом между исследованием новых подграфов и использованием уже найденных перспективных вариантов в MCTS.
* ``expand_atoms`` — количество узлов, добавляемых при расширении дочерних вершин в дереве поиска.
* ``local_radius`` — радиус локальной окрестности, используемый при формировании подграфов для объяснения.
* ``sample_num`` — число выборок, используемых в Monte Carlo аппроксимации Shapley-значений.
* ``reward_method`` — метод вычисления функции награды, определяющий вклад подграфа в предсказание модели.
* ``high2low`` — если ``True``, расширение дерева поиска выполняется начиная с узлов с наибольшей степенью.
* ``subgraph_building_method`` — способ построения подграфа при оценке объяснения (обнуление признаков удалённых узлов или разделение графа).

Вы можете задать конфигурацию запуска метода SubgraphX, указав нужные значения параметров в ``ExplainerRunConfig`` в поле ``_config_kwargs``. Метод SubgraphX имеет следующие параметры запуска:

* ``max_nodes`` — максимальное количество узлов, которое может содержать объясняющий подграф.
* ``element_idx`` — индекс узла (или графа), для которого строится объяснение.

Пример кода запуска SubgraphX:

.. code-block:: python

   from gnn_aid.datasets import DatasetManager, LibPTGDataset
   from gnn_aid.data_structures import DatasetConfig, Task, DatasetVarConfig
   from gnn_aid.data_structures.configs import ConfigPattern, FeatureConfig
   from gnn_aid.models_builder import model_configs_zoo, Metric
   from gnn_aid.models_builder.model_managers import FrameworkGNNModelManager
   from gnn_aid.explainers import FrameworkExplainersManager
   from gnn_aid.aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH


   # Defining a dataset
   gen_dataset = DatasetManager.get_by_config(
       DatasetConfig(("example", "example")),
       DatasetVarConfig(task=Task.NODE_CLASSIFICATION,
                        features=FeatureConfig(node_attr=['a']),
                        labeling='binary',
                        dataset_ver_ind=0)
   )
   gen_dataset.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
   results_dataset_path = gen_dataset.prepared_dir

   # Defining a model
   gnn = model_configs_zoo(dataset=gen_dataset, model_name='gcn_gcn')

   gnn_model_manager_config = ConfigPattern(
       _config_class="ModelManagerConfig",
       _config_kwargs={
           "batch": 10000,
           "mask_features": []
       }
   )
   gnn_model_manager = FrameworkGNNModelManager(
       gnn=gnn,
       dataset_path=results_dataset_path,
       manager_config=gnn_model_manager_config
   )

   # Train model
   gnn_model_manager.train_model(gen_dataset=gen_dataset, steps=100, save_model_flag=False,
                                 metrics=[Metric("F1", mask='test')])

   # Define an explainer
   explainer_init_config = ConfigPattern(
       _class_name="SubgraphX",
       _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
       _config_class="ExplainerInitConfig",
       _config_kwargs={
       }
   )
   explainer_run_config = ConfigPattern(
       _config_class="ExplainerRunConfig",
       _config_kwargs={
           "mode": "local",
           "kwargs": {
               "_class_name": "SubgraphX",
               "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
               "_config_class": "Config",
               "_config_kwargs": {

               },
           }
       }
   )
   explainer = FrameworkExplainersManager(
       init_config=explainer_init_config,
       dataset=gen_dataset, gnn_manager=gnn_model_manager,
       explainer_name='SubgraphX',
   )

   # Run explainer
   explanation = explainer.conduct_experiment(explainer_run_config)
   print(explanation)

Метод SubgraphX в текущей реализации позволяет строить объяснение предсказания модели для задач классификации узлов и классификации графов.

Zorro
-----

Алгоритм ZORRO из статьи `"Hard Masking for Explaining Graph Neural Networks" <https://openreview.net/forum?id=uDN8pRAdsoC>`_. Выбирает важные узлы и признаки из вычислительного подграфа путем маскирования компонентов и введения шума. Алгоритм жадно перебирает узлы и признаки исходного вычислительного графа и создает минимальное множество ``S``, которое соответствовало бы множеству значений метрики точности.

Вы можете задать конфигурацию метода ZORRO, указав нужные значения параметров в ``ExplainerInitConfig`` в поле ``_config_kwargs``. Метод ZORRO имеет следующие параметры:

* ``greedy`` — если ``True``, используется жадная стратегия выбора элементов при построении объяснения.
* ``add_noise`` — если ``True``, при построении объяснения к входным данным добавляется случайный шум.
* ``samples`` — количество случайных шумовых выборок, используемых для оценки устойчивости и точности объяснения (fidelity).

Вы можете задать конфигурацию запуска метода ZORRO, указав нужные значения параметров в ``ExplainerRunConfig`` в поле ``_config_kwargs``. Метод ZORRO имеет следующие параметры запуска:

* ``element_idx`` — индекс узла (или графа), для которого строится объяснение.
* ``tau`` — параметр τ-метрики, используемый для оценки качества объяснения согласно метрике, предложенной в статье.
* ``recursion_depth`` — максимальная глубина рекурсии при поиске объясняющего подграфа (значение ``Infinity`` означает отсутствие ограничения).
* ``save_initial_improve`` — если ``True``, сохраняет промежуточные улучшения объяснения, найденные на начальных этапах поиска.

Пример кода запуска ZORRO:

.. code-block:: python

   from gnn_aid.datasets import DatasetManager, LibPTGDataset
   from gnn_aid.data_structures import DatasetConfig, Task, DatasetVarConfig
   from gnn_aid.data_structures.configs import ConfigPattern, FeatureConfig
   from gnn_aid.models_builder import model_configs_zoo, Metric
   from gnn_aid.models_builder.model_managers import FrameworkGNNModelManager
   from gnn_aid.explainers import FrameworkExplainersManager
   from gnn_aid.aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH


   # Defining a dataset
   gen_dataset = DatasetManager.get_by_config(
       DatasetConfig(("example", "example")),
       DatasetVarConfig(task=Task.NODE_CLASSIFICATION,
                        features=FeatureConfig(node_attr=['a']),
                        labeling='binary',
                        dataset_ver_ind=0)
   )
   gen_dataset.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
   results_dataset_path = gen_dataset.prepared_dir

   # Defining a model
   gnn = model_configs_zoo(dataset=gen_dataset, model_name='gcn_gcn')

   gnn_model_manager_config = ConfigPattern(
       _config_class="ModelManagerConfig",
       _config_kwargs={
           "batch": 10000,
           "mask_features": []
       }
   )
   gnn_model_manager = FrameworkGNNModelManager(
       gnn=gnn,
       dataset_path=results_dataset_path,
       manager_config=gnn_model_manager_config
   )

   # Train model
   gnn_model_manager.train_model(gen_dataset=gen_dataset, steps=100, save_model_flag=False,
                                 metrics=[Metric("F1", mask='test')])

   # Define an explainer
   explainer_init_config = ConfigPattern(
       _class_name="Zorro",
       _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
       _config_class="ExplainerInitConfig",
       _config_kwargs={
       }
   )
   explainer_run_config = ConfigPattern(
       _config_class="ExplainerRunConfig",
       _config_kwargs={
           "mode": "local",
           "kwargs": {
               "_class_name": "Zorro",
               "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
               "_config_class": "Config",
               "_config_kwargs": {

               },
           }
       }
   )
   explainer = FrameworkExplainersManager(
       init_config=explainer_init_config,
       dataset=gen_dataset, gnn_manager=gnn_model_manager,
       explainer_name='Zorro',
   )

   # Run explainer
   explanation = explainer.conduct_experiment(explainer_run_config)
   print(explanation)

Метод ZORRO в текущей реализации позволяет строить объяснение предсказания модели для задачи классификации узлов.

GraphMask
---------

Метод GraphMask из статьи `"Interpreting Graph Neural Networks for NLP With Differentiable Edge Masking" <https://arxiv.org/abs/2010.00577>`_. Идентифицирует ребра в графе, которые важны для предсказания модели. Метод обучается функции управления, которая решает, следует ли сохранять исходное сообщение, передаваемое по ребру, или заменять его базовым значением. Это решение принимается на основе скрытых представлений соединенных узлов и сообщения ребра, что позволяет модели обнаруживать и удалять избыточные ребра, сохраняя при этом предсказание.

Вы можете задать конфигурацию метода GraphMask, указав нужные значения параметров в ``ExplainerInitConfig`` в поле ``_config_kwargs``. Метод GraphMask имеет следующие параметры:

* ``epochs`` — количество эпох обучения модели, оптимизирующей маску рёбер.
* ``lr`` — скорость обучения, используемая при оптимизации параметров объяснения.
* ``coff_size`` — коэффициент регуляризации, контролирующий размер и разреженность маски рёбер.
* ``coff_ent`` — коэффициент, определяющий вклад энтропийной регуляризации маски рёбер в функцию потерь.
* ``allowance`` — параметр, используемый в лагранжевой оптимизации для регулирования допустимого отклонения при обучении маски.

Вы можете задать конфигурацию запуска метода GraphMask, указав нужные значения параметров в ``ExplainerRunConfig`` в поле ``_config_kwargs``. Метод GraphMask имеет следующие параметры запуска:

* ``element_idx`` — индекс узла (или графа), для которого строится объяснение.

Пример кода запуска GraphMask:

.. code-block:: python

   from gnn_aid.datasets import DatasetManager, LibPTGDataset
   from gnn_aid.data_structures import DatasetConfig, Task, DatasetVarConfig
   from gnn_aid.data_structures.configs import ConfigPattern, FeatureConfig
   from gnn_aid.models_builder import model_configs_zoo, Metric
   from gnn_aid.models_builder.model_managers import FrameworkGNNModelManager
   from gnn_aid.explainers import FrameworkExplainersManager
   from gnn_aid.aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH

   # Defining a dataset
   gen_dataset = DatasetManager.get_by_config(
       DatasetConfig(("example", "example")),
       DatasetVarConfig(task=Task.NODE_CLASSIFICATION,
                        features=FeatureConfig(node_attr=['a']),
                        labeling='binary',
                        dataset_ver_ind=0)
   )
   gen_dataset.train_test_split(percent_train_class=0.6, percent_test_class=0.4)
   results_dataset_path = gen_dataset.prepared_dir

   # Defining a model
   gnn = model_configs_zoo(dataset=gen_dataset, model_name='gcn_gcn')

   gnn_model_manager_config = ConfigPattern(
       _config_class="ModelManagerConfig",
       _config_kwargs={
           "batch": 10000,
           "mask_features": []
       }
   )
   gnn_model_manager = FrameworkGNNModelManager(
       gnn=gnn,
       dataset_path=results_dataset_path,
       manager_config=gnn_model_manager_config
   )

   # Train model
   gnn_model_manager.train_model(gen_dataset=gen_dataset, steps=100, save_model_flag=False,
                                 metrics=[Metric("F1", mask='test')])

   # Define an explainer
   explainer_init_config = ConfigPattern(
       _class_name="GraphMask",
       _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
       _config_class="ExplainerInitConfig",
       _config_kwargs={
       }
   )
   explainer_run_config = ConfigPattern(
       _config_class="ExplainerRunConfig",
       _config_kwargs={
           "mode": "local",
           "kwargs": {
               "_class_name": "GraphMask",
               "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
               "_config_class": "Config",
               "_config_kwargs": {

               },
           }
       }
   )
   explainer = FrameworkExplainersManager(
       init_config=explainer_init_config,
       dataset=gen_dataset, gnn_manager=gnn_model_manager,
       explainer_name='GraphMask',
   )

   # Run explainer
   explanation = explainer.conduct_experiment(explainer_run_config)
   print(explanation)

Метод GraphMask в текущей реализации позволяет строить объяснение предсказания модели для задачи классификации узлов.
