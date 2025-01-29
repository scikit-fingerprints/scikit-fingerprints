import inspect
import os
from collections import defaultdict
from pathlib import Path

import skfp.descriptors
import skfp.distances
import skfp.filters
import skfp.fingerprints
import skfp.metrics
import skfp.model_selection
import skfp.preprocessing
import skfp.utils

"""
Testing of documentation pages, ensures that all classes are mentioned in proper
.rst files.
"""


def test_docs():
    curr_dir = os.getcwd()
    curr_dir_files = os.listdir(curr_dir)
    if curr_dir.endswith("scikit-fingerprints") or (
        "skfp" in curr_dir_files and "tests" in curr_dir_files
    ):
        root_dir = Path(curr_dir)
    elif curr_dir.endswith("tests"):
        root_dir = Path(curr_dir).parent
    else:
        raise ValueError(f"Directory {curr_dir} not recognized")

    docs_modules_dir = root_dir / "docs" / "modules"

    undocumented = defaultdict(list)
    for docs_file, code_file in [
        ("descriptors.rst", skfp.descriptors),
        ("distances.rst", skfp.distances),
        ("filters.rst", skfp.filters),
        ("fingerprints.rst", skfp.fingerprints),
        ("metrics.rst", skfp.metrics),
        ("model_selection.rst", skfp.model_selection),
        ("preprocessing.rst", skfp.preprocessing),
        ("utils.rst", skfp.utils),
    ]:
        with open(docs_modules_dir / docs_file) as file:
            docs = file.read()

        objects_missing_docs = [
            name
            for name, obj in inspect.getmembers(code_file)
            if (inspect.isclass(obj) or inspect.isfunction(obj)) and name not in docs
        ]
        if objects_missing_docs:
            undocumented[docs_file] = objects_missing_docs

    if undocumented:
        error_msg = "Found objects missing documentation:\n"
        for docs_file, objects_missing_docs in undocumented.items():
            error_msg += f"{docs_file}, missing: {', '.join(objects_missing_docs)}\n"
        raise ValueError(error_msg)


def get_root_dir() -> Path:
    curr_dir = os.getcwd()
    if curr_dir.endswith("scikit-fingerprints"):
        return Path(curr_dir)
    if curr_dir.endswith("tests"):
        return Path(curr_dir).parent

    raise ValueError(f"Current directory {curr_dir} not recognized")
