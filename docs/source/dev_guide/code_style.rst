Принципы разработки
*******************

.. contents::
    :local:
    :depth: 2



Стиль кода
===============

За основу взят стандарт PEP-8

Линтер
===============

Для автоматической проверки стиля мы используем linter, конфигурация находится в файле `.github/workflows/linter.yaml`.

Работа с репозиторием
=====================

Разработка ведется в открытом репозитории https://github.com/ispras/GNN-AID
В ветку main попадают стабильные версии системы.
В ветке develop и остальных ведется разработка.
Порядок такой: разработчик ответвляется от develop и вносит свои изменения в код.
По окончании он создает пулл-реквест на слияние с веткой develop.

Чтобы обеспечить надежность при изменении кода, отправляемых в гит-репозиторий, используются принципы CI/CD.
В папке `.github/workflows` находятся yaml файлы конфигураций:

- сборка документации (`docs.yaml`)
- запуск линтера (`linter.yaml`)
- запус юнит-тестов (`test_on_push.yaml`)

Эти три проверки запускаются каждый раз при создании запроса на добавление кода в ветку develop.

Комментирование
===============

Предпочтительный стиль комментирования классов и функций в коде.
Документирующие строки - предпочитаем Google Style

1) короткий вариант (если функция getter, setter или делает что-то простое):

.. code-block:: python

        def simple_function(
                param: int
        ) -> bool:
            """ Short description.
            """
            ...

            return True


2) длинный вариант (для сложных функций и всех классов)

.. code-block:: python

    def complex_function(
            param1: int,
            param2: str="default"
    ) -> bool:
        """
        A detailed description with examples if needed in google format.

        Args:
            param1 (int): First parameter description.
            param2 (str): Second parameter description. Default value: `default`.

        Returns:
            bool: Description of the result.
        """
        ...

        return True

Тестирование
===============

Все тесты складываем в папку `tests`.
Для юнит-тестов используем класс ``unittest.TestCase``.
При работе система использует рабочие директории `data`, `datasets`, `models`, `explanations` которые заданы константами в utils.
При тестировании мы подменяем реальные пути временными с помощью функции ``monkey_patch_dirs()`` из ``tests/utils``.


