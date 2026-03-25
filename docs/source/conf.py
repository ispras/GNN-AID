# Configuration file for the Sphinx documentation builder.

# # Извлечь тексты для перевода
# sphinx-build -b gettext . _build/gettext
#
# # Создать/обновить англ переводы
# sphinx-intl update -p _build/gettext -l en
#
# # Собрать документацию на англ и русском
# sphinx-build -b html -D language=en . _build/html/en
# sphinx-build -b html -D language=ru . _build/html/ru


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../gnn_aid'))
sys.path.insert(0, os.path.abspath('../../web_interface'))
sys.path.insert(0, os.path.abspath('../../'))
sys.setrecursionlimit(1500)

# Use fake imports sto avoid readthedocs fails because of installation timeout
sys.path.insert(0, os.path.abspath("../"))    # path to docs/
import docs.mock_imports
import sphinx_rtd_theme

# -- Project information

project = 'GNN-AID'
copyright = '2024, GNN-AID team'
author = 'GNN-AID team'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    # 'sphinx.ext.coverage',
    # 'sphinx.ext.doctest',
    # 'sphinx.ext.duration',
    # 'sphinx.ext.ifconfig',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',  # support Google-style docstrings
#    'sphinxcontrib.fulltoc',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    # 'torch': ("https://pytorch.org/docs/stable/", None),
    'torch': ("https://docs.pytorch.org/docs/stable/", None),
    'torch_geometric': ('https://pytorch-geometric.readthedocs.io/en/latest/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# ---------------------
autodoc_member_order = 'bysource'
autodoc_special_members = '__init__'

# The default options for autodoc directives. They are applied to all autodoc directives automatically. It must be a dictionary which maps option names to the values.
autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': '__init__',
    'exclude-members': '__weakref__',
    'ignore-module-all': True,
}

autosummary_generate = True
add_module_names = False  # убирает 'datasets.ptg_datasets.' перед именем

todo_include_todos = True

# -- locale

language = 'ru'  # русский как основной
locale_dirs = ['locale/']
gettext_compact = False

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': True,
    'titles_only': False,
}

html_static_path = ["_static"]
html_logo = "_static/logo.png"

# -- Options for EPUB output
epub_show_urls = 'footnote'

