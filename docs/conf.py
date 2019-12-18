# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'PyPDE'
copyright = '2019, Haran Jackson'
author = 'Haran Jackson'
master_doc = 'index'

# -- General configuration ---------------------------------------------------

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'scipy'
html_theme_path = ['_theme']
html_static_path = ['_static']
html_theme_options = {
    "edit_link": "false",
    "rootlinks": [("https://github.com/haranjackson/PyPDE", "View on GitHub")]
}
html_show_sphinx = False
html_show_sourcelink = False
