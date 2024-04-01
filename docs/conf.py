import datetime

# -- Project information -----------------------------------------------------

current_year = datetime.datetime.now().year
project = "scikit-fingerprints"
project_copyright = f"2019 - {current_year} (MIT License)"
author = "scikit-fingerprints developers"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.duration",
    "sphinx.ext.napoleon",
]

autodoc_default_options = {
    "inherited-members": True,
    "members": "fit,fit_transform,transform",
}

templates_path = ["_templates"]
exclude_patterns = [
    ".ipynb_checkpoints",
    ".DS_Store",
    "_build",
    "Thumbs.db",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
