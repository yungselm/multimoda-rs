# docs/conf.py
import os
import sys
import subprocess

# Critical for PyO3 modules
def setup(app):
    # Build Rust extension in-place
    subprocess.run(["maturin", "develop", "--release"], cwd="..", check=True)
    
    # Workaround for autodoc
    from unittest.mock import MagicMock
    sys.modules["multimodars"] = MagicMock()

# Project info
project = 'multimodars'
copyright = '2024, Anselm W. Stark'
author = 'Anselm W. Stark'
release = '0.0.2'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'myst_parser'
]

# Autodoc config
autodoc_mock_imports = ["multimodars"]
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}
autodoc_typehints = "description"

# Path setup
sys.path.insert(0, os.path.abspath('../src'))

# Theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': False,
}