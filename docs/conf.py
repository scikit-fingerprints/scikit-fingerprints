import datetime
import os
import shutil

# -- Project information -----------------------------------------------------

current_year = datetime.datetime.now().year
project = "scikit-fingerprints"
project_copyright = f"2019 - {current_year} (MIT License)"
author = "scikit-fingerprints developers"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "nbsphinx",
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

intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"

copybutton_exclude = ".linenos, .gp, .go"

html_theme_options = {
    "header_links_before_dropdown": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/scikit-fingerprints/scikit-fingerprints",
            "icon": "fa-brands fa-github",
        },
    ],
}

# Sphinx cannot reach outside to parallel dirs, so we copy "examples" directory
# we copy "examples" directory from project root to docs/examples


def all_but_ipynb(dir: str, contents: list[str]) -> list[str]:
    return [
        c
        for c in contents
        if os.path.isfile(os.path.join(dir, c)) and not c.endswith(".ipynb")
    ]


shutil.rmtree("./examples", ignore_errors=True)
shutil.copytree(
    src="../examples",
    dst="./examples",
    ignore=all_but_ipynb,
)
