Работа с датасетами
*********************

.. contents::
    :local:


Основная идея
=============

Первым шагом работы с фреймворком GNN-AID является выбор датасета. Мы разделяем его на 2 этапа:

1. выбор графа или набора графов (определяется ``DatasetConfig``)
2. определение вариативной части датасета: признаки, разметка, задача (определяется ``DatasetVarConfig``).

Такое разделение обусловлено тем, что на одном и том же графе, как объекте реального мира, можно решать множество задач, различающихся способом формирования числовых признаков из исходных атрибутов вершин или ребер, задания меток классов или сменить постановку задачи с классификации вершин на предсказание ребер. В этой связи мы различаем атрибуты и признаки. Атрибуты — это характеристики вершин или ребер, например: возраст человека, указанный в аккаунте соцсети, для социального графа, набор ключевых слов статьи для графа цитирования, тип атома и тип связи в графе, представляющем молекулу. Разметка также является вариативной величиной. Например, для известного графа ``karate-club`` существует как минимум 3 вида разметки вершин для классификации: на 2, 3 и 4 класса. Третья составляющая — тип задачи. Например, на графе цитирования можно предсказывать какому разделу науки принадлежит статья (классификация вершин), а также рекомендовать возможные цитирования (предсказание ребра). После
выбора вариативной части датасета в системе создаются тензоры признаков (объекты pytorch.tensor) для вершин и, при необходимости, ребер графа, пригодные для подачи на вход графовой нейросети. При первом выборе параметров датасета тензоры сохраняются в хранилище датасетов, чтобы при последующих обращениях к такому датасету не создавать их заново, а загрузить с диска.

Основные классы
===============

GeneralDataset и подклассы
---------------------------------

В GNN-AID все датасеты являются объектами абстрактного класса ``GeneralDataset``. Основные поля:

- ``dataset_config`` - конфиг графа (набора графов), ``DatasetConfig``
- ``dataset_var_config`` - конфиг вариативной части датасета, ``DatasetVarConfig``
- ``info`` - метаинформация ``DatasetInfo``
- ``dataset`` - датасет как тензоры ``torch_geometric.data.Dataset``

Основные properties:

- ``root_dir`` - папка с файлами датасета, в ней лежат папка с исходными файлами и файл ``metainfo``
- ``raw_dir`` - путь к папке с исходными файлами датасета ``raw/``
- ``prepared_dir`` - путь к папке, где лежат тензоры
- ``metainfo_path`` - путь к файлу ``metainfo``
- ``data`` - весь датасет как ``torch_geometric.data.Data``
- ``edges`` - список ``edge_index`` для каждого графа датасета
- ``node_attributes`` - атрибуты вершин как словарь
- ``labels`` - метки (вершин или графов) для классификации
- ``node_features`` - признаки вершин как тензоры

Основные функции:

- ``__init__(dataset_config)`` - фиксирует ``metainfo``, вызывает ``_compute_dataset_data()``
- ``build(dataset_var_config)`` - в подклассе строит Dataset на основе dataset_var_config, вызывает ``_compute_dataset_var_data()``
- ``set_visible_part(center, depth)`` - задать область видимости датасета для отрисовки на фронте

Расширяемые функции (переопределяются в подклассах)

- ``_compute_dataset_data()`` - построить граф (edge_index и мб сразу тензоры) на основе ``dataset_config``
- ``_compute_dataset_var_data()``- построить тензоры (``X``, ``Y`` и тд) на основе ``dataset_var_config``
- ``node_attributes(attrs)`` - возвращает атрибуты вершин
- ``edge_attributes(attrs)``- возвращает атрибуты ребер

Имеет два основных подкласса: ``PTGDataset`` - когда есть один вариант для тензоров в стиле PyG, и ``KnownFormatDataset`` - когда есть граф с атрибутами и несколько вариантов их кодирования в признаки. В большинстве случаев пользовательские датасеты наследуются от этих двух классов.

PTGDataset
~~~~~~~~~~

Generalization and a wrapper over a single ptg Dataset. Features and labels are defined at initialization. Extend this class if you have a dataset extending ``torch_geometric.data.Dataset``. У датасета единственный признак с названием PTG_FEATURE_NAME = "unknown", одна разметка с названием "origin".

Основные функции:

- ``_compute_dataset_data()`` - вызывает ``_define_ptg_dataset()``. Если обращение первый раз, автоматически генерирует метаинфо на основе тензоров.
- ``_define_ptg_dataset()`` - инициализирует соответствующий PyG датасет. При первом обращении построит тензоры и сохранит их в папку по умолчанию (``processed``).

Построенные тензоры сохраняются там же, где создаются PyG по умолчанию (``<root_dir>/processed``) для того, чтобы не генерировались при каждом обращении. В директории ``datasets`` (где лежат все тензоры) на эту папку создается ссылка.

Имеет два реализованных подкласса для упрощения использования. ``LocalPTGDataset`` - на вход принимает готовые тензоры, ``LibPTGDataset`` - обертка для существующих датасетов из PyG библиотеки.

LocalPTGDataset
~~~~~~~~~~~~~~~

Нужен для случая быстрого создания датасета из тензоров, полученных в коде. Работает в 2 режимах. Если на вход подаются тензоры - сохраняет новый граф. Если ``dataset_config`` то загружает граф с данным конфигом.

Функция ``__init__`` принимает 3 аргумента:

- ``data_list`` - список объектов ``torch_geometric.data.data.Data``
- ``name`` - опционально, имя датасета, по умолчанию использует временную метку
- ``dataset_config`` - если задан, то игнорирует тензоры и пытается загрузить датасет

Созданный датасет сохранится в папку с датасетами по пути ``locally-created-graphs/<name>/<sub-name>/...``.

LibPTGDataset
~~~~~~~~~~~~~

``LibPTGDataset`` - упрощает работу с библиотечными датасетами из PyG.
Общий список <https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/data_cheatsheet.html>.
Функция ``__init__()`` дополнительно принимает ``**params`` для инициализации класса PyG датасетов, где есть параметры. Датасеты сохраняются в папку по пути ``ptg-library-graphs/<domain>/[<group>]/<name>``, где

- ``domain`` - "Homogeneous", "Heterogeneous" или "Synthetic"
- ``group`` - (если есть) название группы датасетов в PyG, например ``TUDatasets``
- ``name`` - название графа в группе датасетов, например ``MUTAG``

Основная нагруженная функция ``_define_ptg_dataset()``, которая разбирает разные случаи инициализации PyG датасета.

Не все датасеты из PyG поддерживаются.
Список доступных есть в файле ``metainfo/torch_geom_index.json``.
Функция ``is_in_torch_geometric_datasets(full_name)`` позволяет проверить наличие датасета в списке по полному имени.

KnownFormatDataset
~~~~~~~~~~~~~~~~~~

Класс для датасетов в одном из фиксированных форматов со стандартизированной обработкой. Датасет может быть задан во внутреннем формате ``ij`` (подробное описание формата см в [[#Формат датасетов ``ij``]].) или в одном из поддерживаемых: ``gml``, ``adjlist``, ``g6`` и другие (конвертируются в ``ij``).

Предполагается, что сырые данные датасета содержат список ребер и файлы с атрибутами и метками. Также необходим готовый файл ``metainfo``. Построение вариативной части регулируется параметром ``dataset_var_config`` и реализуется в функции ``_compute_dataset_var_data()``. В классе ``KnownFormatDataset`` способы построения признаков из атрибутов предлагают несколько стандартных вариантов и описываются в конфиге ``FeatureConfig``. Подробнее см в [[#Building features from attributes]]. Для более кастомного задания датасета необходимо наследоваться от класса ``GeneralDataset``.

Функция ``__init__`` дополнительно принимает словари со значениями атрибутов по умолчанию - используются если в исходных данных есть пропущенные атрибуты.

Основные properties:

- ``node_attributes_dir`` - путь к папке с атрибутами вершин
- ``labels_dir`` - путь к папке с разметками
- ``edges_path`` - путь к файлу со списком ребер
- ``edge_index_path`` - путь к файлу с индексами ребер по графам (для датасетов из нескольких графов)

Основные функции класса:

- ``check_validity()`` - проверяет что сырые данные валидны и согласованы с метаинфо
- ``_convert_to_ij()`` - конвертирует иные поддерживаемые форматы в ``ij``
- ``_create_ptg()`` - отвечает за формирование тензоров
- ``_feature_tensor()`` - создание тензоров признаков
- ``_labeling_tensor()`` - создание тензоров меток
- ``_read_attributes()`` - чтение атрибутов из сырых данных
- ``_read_single()`` - чтение ребер из сырых данных в тензоры, перенумеровка вершин, случай 1 графа
- ``_read_multi()`` - чтение ребер из сырых данных в тензоры, перенумеровка вершин, случай нескольких графов
- ``_infer_feature_slices_form_attributes()``

DatasetInfo - метаинформация о датасете
---------------------------------------

Класс ``DatasetInfo`` является описанием всех свойств датасета: формат, направление ребер, число графов и вершин и т.д. Основные поля:

- ``class_name`` - имя класса для инициализации объекта (не требуется, если датасет создается через соответствующий класс)
- ``import_from`` - название модуля откуда импортировать класс (не требуется, если датасет создается через соответствующий класс)
- ``name`` - имя датасета (опционально, по умолчанию использует путь)
- ``format`` - формат сырых данных (опционально, по умолчанию ij)
- ``count`` - число графов
- ``directed`` - флаг направленности ребер (опционально, по умолчанию false)
- ``hetero`` - флаг гетерографа (опционально, по умолчанию false)
- ``nodes`` - список числа вершин в графах
- ``remap`` - флаг перенумерации вершин (опционально, по умолчанию false)

Он также задает типы атрибутов для того, чтобы определить как их можно конвертировать в признаки (см [[#Building features from attributes]]).

- ``node_attributes`` - инфо об атрибутах вершин (опционально), словарь с ключами (или с промежуточными ключами с типами вершин - для гетерографов)

  - ``names`` - список названий
  - ``types`` - список типов из 4 вариантов

    - ``"continuous"`` - непрерывная величина
    - ``"categorical"`` - категориальный
    - ``"vector"`` - вектор непрерывных значений (как в торче)
    - ``"other"`` - другой тип, например строка

  - ``values`` - список возможных значений атрибутов

    - ``continuous`` - мин и макс значения, список
    - ``categorical`` - перечисление возможных значений, список
    - ``vector`` - мин и макс значения, список
    - ``other`` - пустой список или какие-то указания на то как обрабатывать значения

- ``edge_attributes`` - инфо об атрибутах ребер (опционально), аналогично
- ``labelings`` - инфо о разметках, словарь с ключами (или с промежуточными ключами с типами вершин - для гетерографов)

  - название разметки -> значение

    - для классификации - число классов
    - для регрессии на вершинах - мин и макс значения или 0

Пример - граф из 8 вершин с атрибутами: ``a`` (непрерывный) и ``b`` (категориальный) у вершин, ``weight`` (непрерывный) у ребер.

.. code:: json

   {
    "class_name": "KnownFormatDataset",
    "import_from": "datasets.known_format_datasets",
    "name": "example",
    "count": 1,
    "directed": false,
    "nodes": [8],
    "remap": true,
    "node_attributes": {
     "names": ["a","b"],
     "types": ["continuous","categorical"],
     "values": [[0,1],["A","B","C"]]
    },
    "edge_attributes": {
     "names": ["weight"],
     "types": ["continuous"],
     "values": [[0,1]]
    },
    "labelings": {
     "binary": 2,
     "threeClasses": 3
    }
   }

Configs
-------

При работе с датасетами необходимы 3 конфига - ``DatasetConfig``, ``DatasetVarConfig`` и ``FeatureConfig``

DatasetConfig
~~~~~~~~~~~~~

Определяет местонахождение датасета (путь к сырым файлам) и параметры инициализации конструктора при наличии (например, вероятность ребра генерации синтетического графа).

- ``full_name`` - кортеж из имен подпапок, длины не менее 2
- ``init_kwargs`` - словарь параметров для конструктора

DatasetVarConfig
~~~~~~~~~~~~~~~~

Определяет как будут построена вариативная часть датасета.

- ``features`` - объект ``FeatureConfig``
- ``labeling`` - название разметки или словарь-инструкция для построения разметки
- ``task`` - задача на датасете, объект ``Task``
- ``dataset_ver_ind`` - номер версии датасета. Если тензоры с такими параметрами уже создавали, можно переиспользовать

FeatureConfig - конфиг для признаков
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Определяет как будут построены признаки из атрибутов. ``FeatureConfig`` - стандартный вариант

- ``node_struct`` - структурный признак вершины. Список опций: "degree" (степень вершины), "clustering" (коэффициент кластеризации вершины), "1-hot" (вектор длины равной числу вершин), "10 ones" (вектор из 10 единиц)
- ``node_attr`` - атрибутный признак вершины. Список опций из имен атрибутов
- ``edge_attr`` - атрибутный признак ребра. Список опций из имен атрибутов
- ``graph_attr`` - атрибутный признак графа. Список опций из имен атрибутов (пока недоступно)

Пример.

.. code:: python

   from gnn_aid.data_structures.configs import FeatureConfig

   feature_config = FeatureConfig(
       node_struct=[FeatureConfig.one_hot],
       node_attr=['a', 'b'])

Данный конфиг определяет для каждой вершин признак, состоящий из конкатенации ее 1-hot представления, векторного представления атрибута ``a`` и атрибута ``b`` (их размерность зависит от указанного в metainfo их типа и способа обработки).
Стоит иметь в виду что для больших графов 1-hot представление вершины может потребовать много памяти.

Для максимально кастомных датасетов в конфиге можно задавать более широкие инструкции по построению признаков, которые должны интерпретироваться в функции ``_compute_dataset_var_data()`` в пользовательском классе наследованном от ``GeneralDataset``. Например, это может быть применение SVD к графу или доп обработка атрибутов. ``FeatureConfig`` - обобщенный вариант (для полностью кастомных датасетов).

- ``node_struct`` - строка/список/словарь (ключ - строка, значение - число, строка, флаг, список, или такой же словарь)
- ``node_attr`` - строка/список/словарь
- ``edge_attr`` - строка/список/словарь
- ``graph_attr`` - строка/список/словарь

Словари могут быть с произвольной разрешенной структурой, которая обрабатывается соответствующим пользовательским классом датасета. Для произвольных словарей требования:

- ключ - строка
- значение - число, строка, флаг, список, или такой же словарь Для задания таких словарей на фронте нужно подготовить json с описанием вариантов

Task - задача
~~~~~~~~~~~~~

Перечислитель, определяет решаемую задачу для датасета. Бывает 7 вариантов:

- `NODE_CLASSIFICATION` - классификация вершин
- `NODE_REGRESSION` - регрессия на вершинах
- `GRAPH_CLASSIFICATION` - классификация графов
- `GRAPH_REGRESSION` - регрессия на графах
- `EDGE_PREDICTION` - предсказание ребер
- `EDGE_CLASSIFICATION` - классификация ребер
- `EDGE_REGRESSION` - регрессия на ребрах


Формат датасетов ``ij``
=======================

В GNN-AID предусмотрен внутренний формат ``ij`` для задания произвольных графовых датасетов. Он основан на списке ребер с добавлением файлов, содержащих атрибуты вершин (и ребер при наличии) и разметки. Общие требования:

- все файлы располагаются внутри папки ``raw`` в корневой директории датасета
- файл ``edges.ij`` содержит пары имен вершин, разделенных пробелом
- файлы с атрибутами и разметкой имеют формат JSON
- обязательные файлы: ``edges.ij``, папки ``node_attibutes`` и ``lables`` содержат хотя бы по 1 файлу Далее приведем примеры для разных случаев: 1 граф, несколько графов, гетерограф (только один).

Single graph
------------

Граф из 7 вершин, 8 ребер, 2 атрибутов вершин и 2 разметками вершин Папки

.. code:: text

   edges.ij
   labels/
    |-binary
    |-threeClasses
   node_attributes/
    |-a
    |-b
   edge_attributes/
    |-weight

Файл ``edges.ij``

.. code:: text

   10 11
   11 12
   11 13
   11 15
   12 13
   12 17
   15 14
   15 16

Файл ``labels/binary``

.. code:: json

   {
    "10": 1,
    "11": 1,
    "12": 1,
    "13": 1,
    "14": 0,
    "15": 0,
    "16": 0,
    "17": 0
   }

Файл ``labels/threeClasses``

.. code:: json

   {
    "10": 0,
    "11": 0,
    "12": 1,
    "13": 1,
    "14": 2,
    "15": 2,
    "16": 2,
    "17": 2
   }

Файл ``node_attributes/a``

.. code:: json

   {
    "10":1,
    "11":1,
    "12":0.6,
    "13":0.7,
    "14":0.5,
    "15":0.5,
    "16":0.5,
    "17":0.7
   }

Файл ``node_attributes/b``

.. code:: json

   {
    "10":"A",
    "11":"A",
    "12":"B",
    "13":"C",
    "14":"B",
    "15":"A",
    "16":"A",
    "17":"C"
   }

Файл ``edge_attributes/weight``

.. code:: json

   {
    "10,11": 0.0,
    "11,12": 0.1,
    "11,13": 0.2,
    "11,15": 0.3,
    "12,13": 0.4,
    "12,17": 0.5,
    "15,14": 0.6,
    "15,16": 0.7
   }

Multiple graphs
---------------

Датасет из 3 графов. Появляется новый файл ``edge_index``, который содержит список индексов, с которых начинается каждый следующий граф в списке ребер. Его длина равна числу графов. Файлы атрибутов теперь содержат список словарей для каждого графа. Файлы разметок содержат метки графов а не вершин. Папки

.. code:: text

   edges.ij
   edge_index
   labels/
    |-binary
    |-threeClasses
   node_attributes/
    |-type

Файл ``edges.ij``

.. code:: text

   0 1
   1 2
   0 1
   1 2
   2 3
   3 0
   0 1
   0 2
   0 3
   0 4

Файл ``edge_index``

.. code:: json

   [2, 6, 10]

Файл ``labels/binary``

.. code:: json

   {"0":1,"1":0,"2":0}

Файл ``labels/threeClasses``

.. code:: json

   {"0":0,"1":1,"2":2}

Файл ``node_attributes/type``

.. code:: json

   [
    {"0":"alpha","1":"beta","2":"alpha"},
    {"0":"gamma","1":"beta","2":"gamma","3":"gamma"},
    {"0":"beta","1":"gamma","2":"gamma","3":"alpha","4":"beta"}
   ]

Hetero graph
------------

.. todo::

    Пока не реализовано полностью

Расширение только для датасетов из 1 графа. Вершины и ребра могут иметь различные типы, т.е. свои наборы атрибутов. Тип вершины задается именем (стока), тип ребра тройкой (тип исходящей вершины, тип ребра, тип входящей вершины). Метки вершин также разделяются по типам вершин. Пример - датасет цитирования. Вершины 3 типов: ``author``, ``paper``, ``institution``. Ребра 3 типов: ``author`` -> ``institution``, ``author`` -> ``paper``, ``paper`` -> ``paper``. Метки есть у вершин ``author`` и ``paper`` (2 варианта). Папки

.. code:: text

   edge_attributes  
    |-'author','affiliated_with','institution'  
    |- |-since_year  
    |-'author','writes','paper'  
    |- |-order  
    |-'paper','cites','paper'  
    |- |-context  
   edges  
    |-'author','affiliated_with','institution'  
    |- |-ij  
    |-'author','writes','paper'  
    |- |-ij  
    |-'paper','cites','paper'  
    |- |-ij  
   labels  
    |-author  
    |- |-binary  
    |-paper  
    |- |-score  
    |- |-topic  
   node_attributes  
    |-author  
    |- |-name  
    |-institution  
    |- |-name  
    |- |-rating  
    |-paper  
    |- |-title  
    |- |-year

Содержание файлов аналогично примеру с single graph, меняется только структура и название папок.

Построение признаков из атрибутов
=================================

Исходно датасет в сырых данных содержит атрибуты вершин, которые не всегда числовые. Например, в графе цитирования ``Cora`` для каждой вершины задан мешок слов из абстракта соответствующей статьи. А признаком вершины является бинарный вектор встречаемости слов по всему словарю. В PyG в датасетах доступны только тензоры признаков, а информация об атрибутах утеряна, в то время как это может быть важно с точки зрения интерпретации. В GNN-AID датасет имеет исходные атрибуты, из которых строятся тензоры признаков по определенным правилам. Атрибуты могут быть у вершин, ребер и графов. На данный момент поддерживаются только атрибуты вершин. Правила конвертации в признаки одинаковые для всех.

В GNN-AID выделяется 4 типа для атрибутов

- ``"continuous"`` - непрерывная величина, кодируется как есть
- ``"categorical"`` - категориальный, используется 1-hot кодирование по всем возможным значениям
- ``"vector"`` - вектор непрерывных значений (как в PyG), кодируется как есть
- ``"other"`` - другой тип, например строка, по умолчанию не кодируется

Тип атрибута задается в metainfo. Также там для каждого типа задаются границы возможных значений. Это удобно для нормировки и цветовой отрисовки на фронтенде.

- ``continuous`` - мин и макс значения, список
- ``categorical`` - перечисление возможных значений, список
- ``vector`` - мин и макс значения, список
- ``other`` - пустой список или какие-то указания на то, как обрабатывать значения

При построении датасета в ``DatasetVarConfig`` поле ``features`` (типа ``FeatureConfig``) перечисляет из каких компонентов собирать вектор признаков. Для вершин это структурные компоненты (``node_struct``) и атрибутные (``node_attr``). Каждая преобразуется по своим правилам, итоговый вектор признаков является их конкатенацией. Например

.. code:: python

   features=FeatureConfig(node_struct=["degree"],node_attr=['a', 'b'])

и при этом атрибут ``a`` типа ``continuous``, атрибут ``b`` 1-hot с 3 вариантами значения, то итоговый вектор признака будет содержать ``1+1+3=5`` элементов и может выглядеть так ``[13, 0.728, 0, 0, 1]``. При анализе важно понимать, какой атрибут соответствует какой части вектора признаков. Для этого в классе ``KnownFormatDataset`` есть поле ``node_attr_slices``. В данном случае оно будет содержать ``{'a': (1, 2), 'b': (2, 5)}``.

Создание нового датасета
========================

В общем случае для создания нового датасета нужно:

- набор сырых файлов, в папке ``raw`` (ее путь можно узнать как ``Declare.dataset_root_dir(dataset_config)[0]``)
- файл метаинфо (можно создать и сохранить объект DatasetInfo), где заполнены обязательные поля
- функция построения графа на основе параметров конфига - ``_compute_dataset_data(DatasetConfig)``
- функция построения признаков и разметок на основе параметров вар конфига - ``_compute_dataset_var_data(DatasetVarConfig)``

Для упрощенных вариантов есть подклассы GeneralDataset, которые можно использовать напрямую или расширять (см далее).

5 способов задать датасет
--------------------------

.. raw:: html

   <figure class="table op-uc-figure_align-center op-uc-figure">

.. raw:: html

   <table class="op-uc-table">

.. raw:: html

   <thead class="op-uc-table--head">

.. raw:: html

   <tr class="op-uc-table--row">

.. raw:: html

   <th class="op-uc-p op-uc-table--cell op-uc-table--cell_head">

случай

.. raw:: html

   </th>

.. raw:: html

   <th class="op-uc-p op-uc-table--cell op-uc-table--cell_head">

класс

.. raw:: html

   </th>

.. raw:: html

   <th class="op-uc-p op-uc-table--cell op-uc-table--cell_head">

что задать

.. raw:: html

   </th>

.. raw:: html

   <th class="op-uc-p op-uc-table--cell op-uc-table--cell_head">

особенности

.. raw:: html

   </th>

.. raw:: html

   </tr>

.. raw:: html

   </thead>

.. raw:: html

   <tbody>

.. raw:: html

   <tr class="op-uc-table--row">

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

готовый датасет из PyG

.. raw:: html

   </td>

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

LibPTGDataset

.. raw:: html

   </td>

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

доп параметры для PyG класса

.. raw:: html

   </td>

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

доступен из коробки. Пока не все датасеты поддерживаются

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr class="op-uc-table--row">

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

PyG датасет, созданный в коде

.. raw:: html

   </td>

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

LocalPTGDataset

.. raw:: html

   </td>

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

тензоры

.. raw:: html

   </td>

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

по умолчанию не сохраняет файлы

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr class="op-uc-table--row">

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

файлы в формате ij,gml, g6 и тп

.. raw:: html

   </td>

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

KnownFormatDataset или наследоваться от него

.. raw:: html

   </td>

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

сырые файлы,метаинфо

.. raw:: html

   </td>

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

конвертируется в ij, вар-ты вар конфига управляются с фронта

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr class="op-uc-table--row">

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

пользовательский датасет,наследованный от PyG Dataset

.. raw:: html

   </td>

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

наследоваться от PTGDataset

.. raw:: html

   </td>

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

определить \_define_ptg_dataset()

.. raw:: html

   </td>

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

дергает функции пользовательского PyG Dataset. Фиксированный вариант тензоров

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr class="op-uc-table--row">

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

полностью кастомный датасет

.. raw:: html

   </td>

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

наследоваться от GeneralDataset

.. raw:: html

   </td>

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

raw файлы,метаинфо,определить- node_attributes()- edge_attributes()- \_compute_dataset_data()- \_compute_dataset_var_data()

.. raw:: html

   </td>

.. raw:: html

   <td class="op-uc-p op-uc-table--cell">

вар-ты вар конфига пока не управляются с фронта

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </tbody>

.. raw:: html

   </table>

.. raw:: html

   </figure>

Пример 1 - датасет из библиотеки PyG
------------------------------------

Датасет Cora из серии `Planetoid <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid>`__

.. code:: python

   from gnn_aid.data_structures.configs import DatasetConfig
   from gnn_aid.datasets.ptg_datasets import LibPTGDataset
   from gnn_aid.datasets.datasets_manager import DatasetManager

   dc = DatasetConfig((LibPTGDataset.data_folder, 'Homogeneous', 'Planetoid', 'Cora'))
   dataset = DatasetManager.get_by_config(dc)
   print(dataset.info.nodes)

   >>> [2708]

Пример 2 - PyG датасет, созданный на лету в коде
------------------------------------------------

Датасет из двух одинаковых графов с разными метками.

.. code:: python

   from torch import tensor
   from torch_geometric.data import Data
   from gnn_aid.datasets.ptg_datasets import LocalPTGDataset

   x = tensor([[0, 0], [1, 0], [1, 0]])
   edge_index = tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
   y = tensor([0, 1, 1])
   data_list = [Data(x=x, edge_index=edge_index, y=tensor([0])),
                Data(x=x, edge_index=edge_index, y=tensor([1]))]
   dataset = LocalPTGDataset(data_list)
   print(dataset.data)

   >>> Data(x=[6, 2], edge_index=[2, 8], y=[2])

Пример 3 - файлы в формате ij,gml, g6 и тп
------------------------------------------

Граф из 8 вершин в формате ``ij``.

.. code:: python

   import json

   from gnn_aid.auxil.declaration import Declare
   from gnn_aid.data_structures.configs import DatasetConfig, DatasetVarConfig, FeatureConfig
   from gnn_aid.data_structures import Task
   from gnn_aid.datasets import KnownFormatDataset

   dc = DatasetConfig(('single', 'custom', 'test'))
   root, files_paths = Declare.dataset_root_dir(dc)
   raw = root / 'raw'
   raw.mkdir(parents=True, exist_ok=True)

``raw`` - директория где хранятся все сырые данные датасета.
Создайте файлы вручную или в коде:

.. code:: python

   # Edge list
   with open(raw / 'edges.ij', 'w') as f:
       f.write("10 11\n")
       f.write("11 12\n")
       f.write("11 13\n")
       f.write("11 15\n")
       f.write("12 13\n")
       f.write("12 17\n")
       f.write("15 14\n")
       f.write("15 16\n")

   # 'metainfo' file
   with open(root / 'metainfo', 'w') as f:
       json.dump({
           "name": "example",
           "count": 1,
           "directed": False,
           "nodes": [8],
           "remap": True,
           "node_attributes": {
               "names": ["a", "b"],
               "types": ["continuous", "categorical"],
               "values": [[0, 1], ["A", "B", "C"]]
           },
           "edge_attributes": {
               "names": ["weight"],
               "types": ["continuous"],
               "values": [[0, 1]]
           },
           "labelings": {
               "node-classification": {
                   "binary": 2,
               }
           }
       }, f)
   # Labels for node-classification task
   (raw / 'labels' / 'node-classification').mkdir(parents=True, exist_ok=True)
   with open(raw / 'labels' / 'node-classification' / 'binary', 'w') as f:
       json.dump({"10": 1, "11": 1, "12": 1, "13": 1, "14": 0, "15": 0, "16": 0, "17": 0}, f)

   # Node attributes
   (raw / 'node_attributes').mkdir(exist_ok=True)
   with open(raw / 'node_attributes' / 'a', 'w') as f:
       json.dump(
           {"10": 1, "11": 1, "12": 0.6, "13": 0.7, "14": 0.5, "15": 0.5, "16": 0.5, "17": 0.7},
           f)
   with open(raw / 'node_attributes' / 'b', 'w') as f:
       json.dump(
           {"10": "A", "11": "A", "12": "B", "13": "C", "14": "B", "15": "A", "16": "A",
            "17": "C"}, f)

   # Edge attributes
   (raw / 'edge_attributes').mkdir(exist_ok=True)
   with open(raw / 'edge_attributes' / 'weight', 'w') as f:
       json.dump(
           {"10,11": 0.0, "11,12": 0.1, "11,13": 0.2, "11,15": 0.3, "12,13": 0.4,
            "12,14": 0.3, "12,17": 0.5, "14,15": 0.6, "15,16": 0.7, "16,17": 0.3},
           f)

Теперь можно инициализировать датасет. Для известного формата используем класс ``KnownFormatDataset``

.. code:: python

   # Create a KnownFormatDataset object and register dataset
   dataset = KnownFormatDataset(dc)

   # Create a variable part config
   dvc = DatasetVarConfig(
       task=Task.NODE_CLASSIFICATION,
       features=FeatureConfig(node_attr=['a']),
       labeling='binary',
       dataset_ver_ind=0)

   # Build dataset - create features and labels as PyG tensors
   dataset.build(dvc)
   print(dataset.data)

   >>> Data(x=[8, 1], edge_index=[2, 16], y=[8])

При последующем обращении можно брать этот датасет по его конфигурации:

.. code:: python

   dataset = DatasetManager.get_by_config(dc)


Пример 4 - пользовательский датасет, наследованный от PyG Dataset
-----------------------------------------------------------------

See example at

- ``user_datasets/power_graph/powergrid.py`` - user defined class based on ptg Dataset
- ``user_datasets/power_graph/dataset.py`` - example of wrapper for it based on ``PTGDataset``

Пример 5 - полностью кастомный датасет
--------------------------------------

See example at ``user_datasets/fully_custom_example.py``

Использование существующего датасета
====================================

Датасет внутри фреймворка GNN-AID
---------------------------------

Если датасет уже есть в GNN-AID (создан одним из способов выше), получить его можно следующими способами

1. Вызвать создание объекта класса через конструктор.

.. code:: python

   dc = DatasetConfig(('example', 'example'))
   dataset = KnownFormatDataset(dc)

2. Задаем конфиги DatasetConfig и DatasetVarConfig и используем функцию ``DatasetManager.get_by_config``. После этого необходимо вызвать ``dataset.build(dataset_var_config)`` для определения тензоров.

.. code:: python

   dc = DatasetConfig(('example', 'example'))
   dataset = DatasetManager.get_by_config(dc)
   dvc = DatasetVarConfig(
       task=Task.NODE_CLASSIFICATION,
       features=FeatureConfig(node_attr=['a', 'b']),
       labeling='binary',
       dataset_ver_ind=0)
   dataset.build(dvc)

Или же можно сразу вызвать ``DatasetManager.get_by_config`` с обоими конфигами

.. code:: python

   dc = DatasetConfig(('example', 'example'))
   dvc = DatasetVarConfig(
       task=Task.NODE_CLASSIFICATION,
       features=FeatureConfig(node_attr=['a', 'b']),
       labeling='binary',
       dataset_ver_ind=0)
   dataset = DatasetManager.get_by_config(dc, dvc)

Датасет создан в другом месте или сгенерирован
----------------------------------------------

Допустим, необходимо адаптировать существующий датасет, который хранится в другом месте на диске и имеет один из поддерживаемых форматов ('ij', 'gml' и т.д.).
Тогда нужно следовать примеру 3.
Все подходящие сырые данные нужно скопировать в папку ``raw``. Например ``edges.ij``.
Если есть гарантия, что файл не будет изменяться, можно создать символьную ссылку вместо копирования.
Затем нужно создать минимальный файл ``metainfo``, как показано далее.
Вам понадобится знать число графов в датасете и число вершин в каждом из них.

.. code:: python

   import shutil
   shutil.copy(path_to_edges, raw / 'edges.ij')

   # 'metainfo' file
   with open(root / 'metainfo', 'w') as f:
       json.dump({
           "name": "example",
           "count": 1,
           "directed": False,
           "nodes": [123],
           "remap": True,
           "node_attributes": {
               "names": [],
               "types": [],
               "values": []
           },
           "labelings": {
           }
       }, f)

   dataset = KnownFormatDataset(dc)

Если в вашем датасете нет атрибутов и меток, то можно указать задачу предсказания ребер ``Task.EDGE_PREDICTION``, поскольку она не требует меток, а признаки строить на основе 1-hot представления.

.. code:: python

   dvc = DatasetVarConfig(
       task=Task.EDGE_PREDICTION,
       features=FeatureConfig(node_struct=[FeatureConfig.one_hot]),
       dataset_ver_ind=0)

