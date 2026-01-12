import os
import sys
import re

try:
    import multimodars
except ImportError:
    from unittest.mock import MagicMock

    sys.modules["multimodars"] = MagicMock()

# Get the version from Cargo.toml
def get_version():
    with open(os.path.join("..", "Cargo.toml"), "r") as f:
        content = f.read()
    # Find the version string in the [package] section
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if match:
        return match.group(1)
    return "0.0.0"

# Add project to path
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))

# Project info
project = "multimodars"
copyright = "2025, Anselm W. Stark"
author = "Anselm W. Stark"
release = get_version()
version = ".".join(release.split(".")[:2])

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# Templates and static files
_here = os.path.abspath(os.path.dirname(__file__))

templates_path = [os.path.join(_here, "_templates")]
html_static_path = [os.path.join(_here, "_static")]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Theme
# html_theme = "furo"
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}

# Autodoc config
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_typehints = "description"

# Workaround for PyO3 modules
try:
    import multimodars
except ImportError:
    from unittest.mock import MagicMock

    sys.modules["multimodars"] = MagicMock()
    print("Warning: multimodars module not found, using mocks for documentation")
