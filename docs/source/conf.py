# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the ``image_matchmaker`` package importable for autodoc (repo root).
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'bio_matchmaker'
copyright = '2026, kreshuklab'
author = 'kreshuklab'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Autodoc -----------------------------------------------------------------
# The package's heavy / compiled / GPU dependencies are not installed in the docs
# build environment, so mock them out; only ``image_matchmaker`` itself is imported.
autodoc_mock_imports = [
    "numpy", "scipy", "pandas", "matplotlib", "seaborn", "skimage",
    "tifffile", "z5py", "itk", "cvxpy", "transforms3d",
    "open3d", "probreg", "mobie", "elf", "cupy", "click", "yaml",
]
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = True



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
