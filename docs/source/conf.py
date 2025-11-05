# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../..'))
sys.setrecursionlimit(1500)

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
add_module_names = False  # убирает 'datasets_block.ptg_datasets.' перед именем

todo_include_todos = True