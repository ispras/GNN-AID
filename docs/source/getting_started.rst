Getting started
***************

.. contents::
    :local:
    :depth: 2

..
    гайд по установке и запуску (на не более 30мин), с ощущением достижения для читателя

    установка
    запуск на бэке
        взять датасет, обучить модель, добавить 2 атаки и посмотреть числа
    запуск фронта
        запуск main и как открыть в браузере
        запуск из докера?


Installation
============

| GNN-AID was developed and tested under Ubuntu 20.04 and 22, so they
  suit best for it.
| For another OS consider docker.

You need python of version ``3.11`` or higher. We advice to create
virtual environment with ``pip``.

::

   python -m pip install --upgrade pip

First, get auxiliary libraries

::

   sudo apt-get install -y build-essential python3-dev libfreetype6-dev pkg-config

Then install all project dependencies

::

   pip install -r requirements1.txt
   pip install -r requirements2.txt
   pip install -r requirements3.txt

The 3rd pack of requirements will take around 20 minutes.

Problems
--------

If you see 139 or 134 error code when run the script, it is likely a
compatibility issue. Try the following:

1. Update video card drivers (if you have decided to use cuda).
2. Update gcc to the most recent version.
3. Remove all torch modules that use С++ code.
4. Install all torch packages again.

Run in backend
==============

.. todo::

    TODO


Create dataset
--------------

.. todo::

    TODO


Build and train model
---------------------

.. todo::

    TODO

Run in frontend
===============

Suppose you are at the project root folder. Activate virtual environment

.. code:: text

   source venv/bin/activate

Go to ``src`` folder and add it to python dependencies

.. code:: text

   cd src
   export PYTHONPATH=.

Run ``main.py`` script

.. code:: text

   python web_interface/main.py

You will see something like this

.. code:: text

   ======== Running on http://0.0.0.0:5000 ========
   (Press CTRL+C to quit)

Then go to `127.0.0.1:5000 <http://127.0.0.1:5000>`__ in your browser.
You should see web-interface is loaded.

Docker version
--------------

.. todo::

    TODO