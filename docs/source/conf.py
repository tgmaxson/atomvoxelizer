from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

project = "AtomVoxelizer"
author = "AtomVoxelizer contributors"
release = "0.2.12"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autodoc_mock_imports = ["cupy"]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "alabaster"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "page_width": "875px",
    "sidebar_width": "260px",
}

autodoc_typehints = "description"
