# Configuration file for the Sphinx documentation builder.

# -- Project information
import sys, os

# TODO: needed?
sys.path.insert(0, os.path.abspath('../..'))
print(sys.path)

project = 'nosnoc'
copyright = '2023, Jonathan Frey, Anton Pozharskiy, Armin Nurkanovic'
author = 'Jonathan Frey, Anton Pozharskiy, Armin Nurkanovic'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'enum_tools.autoenum',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
