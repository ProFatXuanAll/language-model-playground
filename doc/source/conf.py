"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'Language Model Playground'
copyright = '2022, ProFatXuanAll'
author = 'ProFatXuanAll'

# The full version, including alpha/beta/rc tags
version = '1.0.0'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  'sphinx.ext.coverage',
  'sphinx.ext.intersphinx',
  'sphinx.ext.mathjax',
  'sphinx.ext.napoleon',
  'sphinx.ext.todo',
  'sphinx.ext.viewcode',
  'sphinx_copybutton',
  'sphinxcontrib.bibtex',
]

# Generating bibtex automatically.
bibtex_bibfiles = ['refs.bib']

# Customize latex commands.
mathjax3_config = {
  'tex':
    {
      'macros':
        {
          'ElmanNet': r'\operatorname{ElmanNet}',
          'ElmanNetLayer': r'\operatorname{ElmanNetLayer}',
          'LayerNorm': r'\operatorname{LayerNorm}',
          'LSTMNineSeven': r'\operatorname{LSTM1997}',
          'LSTMNineSevenLayer': r'\operatorname{LSTM1997Layer}',
          'LSTMZeroZero': r'\operatorname{LSTM2000}',
          'LSTMZeroZeroLayer': r'\operatorname{LSTM2000Layer}',
          'LSTMZeroTwo': r'\operatorname{LSTM2002}',
          'LSTMZeroTwoLayer': r'\operatorname{LSTM2002Layer}',
          'MultiHeadAttnLayer': r'\operatorname{MultiHeadAttnLayer}',
          'PE': r'\operatorname{PE}',
          'PosEncLayer': r'\operatorname{PosEncLayer}',
          'Sim': r'\operatorname{sim}',
          'TransEnc': r'\operatorname{TransEnc}',
          'TransEncLayer': r'\operatorname{TransEncLayer}',
          'algoCmt': [r'\text{// #1}', 1],
          'algoElse': r'\textbf{else do}',
          'algoElseIf': [r'\textbf{else if } #1 \textbf{ do}', 1],
          'algoEndFor': r'\textbf{end for}',
          'algoEndIf': r'\textbf{end if}',
          'algoEndProc': r'\textbf{end procedure}',
          'algoEq': r'\leftarrow',
          'algoFalse': r'\textbf{ False }',
          'algoFor': [r'\textbf{for } #1 \textbf{ do}', 1],
          'algoIf': [r'\textbf{if } #1 \textbf{ do}', 1],
          'algoIs': r'\textbf{ is }',
          'algoProc': [r'\textbf{procedure} #1', 1],
          'algoReturn': r'\textbf{return }',
          'algoTrue': r'\textbf{ True }',
          'attn': r'\operatorname{attn}',
          'br': [r'{\left[ #1 \right]}', 1],
          'cat': [r'\operatorname{concate}\pa{#1}', 1],
          'drop': [r'\operatorname{dropout}\pa{#1, #2}', 2],
          'dBlk': r'd_{\operatorname{blk}}',
          'dEmb': r'd_{\operatorname{emb}}',
          'dFf': r'd_{\operatorname{ff}}',
          'dHid': r'd_{\operatorname{hid}}',
          'dMdl': r'd_{\operatorname{model}}',
          'fla': [r'\operatorname{flatten}\pa{#1}', 1],
          'hIn': r'H_{\operatorname{in}}',
          'hOut': r'H_{\operatorname{out}}',
          'indent': [r'\hspace{#1em}', 1],
          'init': r'\operatorname{init}',
          'loss': r'\operatorname{Loss}',
          'msk': r'\operatorname{mask}',
          'nBlk': r'n_{\operatorname{blk}}',
          'nLyr': r'n_{\operatorname{lyr}}',
          'nHd': r'n_{\operatorname{head}}',
          'pEmb': r'p_{\operatorname{emb}}',
          'pHid': r'p_{\operatorname{hid}}',
          'pa': [r'{\left( #1 \right)}', 1],
          'pos': r'\operatorname{pos}',
          'sof': [r'\operatorname{softmax}\pa{#1}', 1],
          'sz': [r'\operatorname{size}\pa{#1}', 1],
        }
    }
}

# Generate automatic links to following projects.
intersphinx_mapping = {
  'numpy': ('http://docs.scipy.org/doc/numpy/', None),
  'python': ('https://docs.python.org/3', None),
  'torch': ('https://pytorch.org/docs/master/', None),
}

# Show todo items in output documents.
todo_include_todos = True

# Parse NumPy style docstrings but not google style docstrings.
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add source suffix.
source_suffix = ['.rst', '.md']

# Do not append parentheses to functions.
add_function_parentheses = False
