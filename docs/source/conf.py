# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
project = 'TurboGuard'
copyright = '2024, Boutrid Mourad & Kassimi Achraf'
author = 'Boutrid Mourad & Kassimi Achraf'
version = '1.0.0'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'myst_parser',
    'sphinxcontrib.mermaid',
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
    'nbsphinx',
    'sphinxcontrib.mermaid'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Source file suffixes
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# -- Options for HTML output ------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_logo = '_static/logo.png'
html_favicon = '_static/favicon.ico'

html_theme_options = {
    'analytics_id': 'your-analytics-id',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

html_context = {
    "display_github": True,
    "github_user": "mouradboutrid",
    "github_repo": "TurboGuard",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

# -- Extension configuration -------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_mock_imports = [
    'tensorflow',
    'keras',
    'numpy',
    'pandas',
    'matplotlib',
    'seaborn',
    'plotly',
    'streamlit',
    'sklearn'
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'tensorflow': ('https://www.tensorflow.org/api_docs/python', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# Mermaid configuration
mermaid_version = "10.3.0"
mermaid_init_js = "mermaid.initialize({startOnLoad:true});"

# MyST Parser configuration
myst_enable_extensions = [
    "deflist",
    "tasklist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "linkify",
    "substitution",
]

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# NBSphinx configuration
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
