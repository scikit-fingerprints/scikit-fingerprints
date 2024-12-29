import inspect
import os
from collections import defaultdict
from pathlib import Path

import skfp.distances
import skfp.filters
import skfp.fingerprints
import skfp.metrics

"""
Testing of documentation pages, ensures that all classes are mentioned in proper
.rst files.
"""


def test_docs():
    curr_dir = os.getcwd()
    if curr_dir.endswith("scikit-fingerprints"):
        root_dir = Path(curr_dir)
    elif curr_dir.endswith("tests"):
        root_dir = Path(curr_dir).parent
    else:
        raise ValueError(f"Directory {curr_dir} not recognized")
    docs_modules_dir = root_dir / "docs" / "modules"

    undocumented = defaultdict(list)
    for docs_file, code_file in [
        ("distances.rst", skfp.distances),
        ("filters.rst", skfp.filters),
        ("fingerprints.rst", skfp.fingerprints),
        ("metrics.rst", skfp.metrics),
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
    elif curr_dir.endswith("tests"):
        return Path(curr_dir).parent
    else:
        raise ValueError(f"Current directory {curr_dir} not recognized")
