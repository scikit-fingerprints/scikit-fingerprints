# scikit-fingerprints

[![PyPI version](https://badge.fury.io/py/scikit-fingerprints.svg)](https://badge.fury.io/py/scikit-fingerprints)
[![](https://img.shields.io/pypi/dm/scikit-fingerprints)](https://pypi.org/project/scikit-fingerprints/)
[![Downloads](https://static.pepy.tech/badge/scikit-fingerprints)](https://pepy.tech/project/scikit-fingerprints)
![Libraries.io dependency status for latest release](https://img.shields.io/librariesio/release/pypi/scikit-fingerprints)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-41B5BE?style=flat)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE.md)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/scikit-fingerprints.svg)](https://pypi.org/project/scikit-fingerprints/)
[![Contributors](https://img.shields.io/github/contributors/scikit-fingerprints/scikit-fingerprints)](https://github.com/scikit-fingerprints/scikit-fingerprints/graphs/contributors)
[![check](https://github.com/scikit-fingerprints/scikit-fingerprints/actions/workflows/python-test.yml/badge.svg)](https://github.com/scikit-fingerprints/scikit-fingerprints/actions/workflows/python-test.yml)

[scikit-fingerprints](https://scikit-fingerprints.github.io/scikit-fingerprints/) is a Python library for efficient
computation of molecular fingerprints.

## Table of Contents

- [Description](#description)
- [Supported platforms](#supported-platforms)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Project overview](#project-overview)
- [Contributing](#contributing)
- [License](#license)

---

## Description

Molecular fingerprints are crucial in various scientific fields, including drug discovery, materials science, and
chemical analysis. However, existing Python libraries for computing molecular fingerprints often lack performance,
user-friendliness, and support for modern programming standards. This project aims to address these shortcomings by
creating an efficient and accessible Python library for molecular fingerprint computation.

See [the documentation and API reference](https://scikit-fingerprints.github.io/scikit-fingerprints/) for details.

Main features:
- scikit-learn compatible
- feature-rich, with >30 fingerprints
- parallelization
- sparse matrix support
- commercial-friendly MIT license

## Supported platforms

|                      | `python3.9`            | `python3.10` | `python3.11` | `python3.12` |
|----------------------|------------------------|--------------|--------------|--------------|
| **Ubuntu - latest**  | ✅                      | ✅            | ✅            | ✅            |
| **Windows - latest** | ✅                      | ✅            | ✅            | ✅            |
| **macOS - latest**   | only macOS 13 or newer | ✅            | ✅            | ✅            |

## Installation

You can install the library using pip:

```bash
pip install scikit-fingerprints
```

If you need bleeding-edge features and don't mind potentially unstable or undocumented functionalities,
you can also install directly from GitHub:
```bash
pip install git+https://github.com/scikit-fingerprints/scikit-fingerprints.git
```

## Quickstart

Most fingerprints are based on molecular graphs (topological, 2D-based), and you can use SMILES
input directly:
```python
from skfp.fingerprints import AtomPairFingerprint

smiles_list = ["O=S(=O)(O)CCS(=O)(=O)O", "O=C(O)c1ccccc1O"]

atom_pair_fingerprint = AtomPairFingerprint()

X = atom_pair_fingerprint.transform(smiles_list)
print(X)
```

For fingerprints using conformers (conformational, 3D-based), you need to create molecules first
and compute conformers. Those fingerprints have `requires_conformers` attribute set
to `True`.
```python
from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer
from skfp.fingerprints import WHIMFingerprint

smiles_list = ["O=S(=O)(O)CCS(=O)(=O)O", "O=C(O)c1ccccc1O"]

mol_from_smiles = MolFromSmilesTransformer()
conf_gen = ConformerGenerator()
fp = WHIMFingerprint()
print(fp.requires_conformers)  # True

mols_list = mol_from_smiles.transform(smiles_list)
mols_list = conf_gen.transform(mols_list)

X = fp.transform(mols_list)
print(X)
```

You can also use scikit-learn functionalities like pipelines, feature unions
etc. to build complex workflows. Popular datasets, e.g. from MoleculeNet benchmark,
can be loaded directly.
```python
from skfp.datasets.moleculenet import load_clintox
from skfp.metrics import multioutput_auroc_score, extract_pos_proba
from skfp.model_selection import scaffold_train_test_split
from skfp.fingerprints import ECFPFingerprint, MACCSFingerprint
from skfp.preprocessing import MolFromSmilesTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, make_union


smiles, y = load_clintox()
smiles_train, smiles_test, y_train, y_test = scaffold_train_test_split(
    smiles, y, test_size=0.2
)

pipeline = make_pipeline(
    MolFromSmilesTransformer(),
    make_union(ECFPFingerprint(count=True), MACCSFingerprint()),
    RandomForestClassifier(random_state=0),
)
pipeline.fit(smiles_train, y_train)

y_pred_proba = pipeline.predict_proba(smiles_test)
y_pred_proba = extract_pos_proba(y_pred_proba)
auroc = multioutput_auroc_score(y_test, y_pred_proba)
print(f"AUROC: {auroc:.2%}")
```

## Examples

You can find Jupyter Notebooks with examples and tutorials [in documentation](https://scikit-fingerprints.github.io/scikit-fingerprints/examples.html),
as well as in the ["examples" directory](https://github.com/scikit-fingerprints/scikit-fingerprints/tree/master/examples).

Examples and tutorials:
1. [Introduction to scikit-fingerprints](examples/01_skfp_introduction.ipynb)
2. [Fingerprint types](examples/02_fingerprint_types.ipynb)

## Project overview

`scikit-fingerprint` brings molecular fingerprints and related functionalities into
the scikit-learn ecosystem. With familiar class-based design and `.transform()` method,
fingerprints can be computed from SMILES strings or RDKit `Mol` objects. Resulting NumPy
arrays or SciPy sparse arrays can be directly used in ML pipelines.

Main features:

1. **Scikit-learn compatible:** `scikit-fingerprints` uses familiar scikit-learn
   interface  and conforms to its API requirements. You can include molecular
   fingerprints in pipelines, concatenate them with feature unions, and process with
   ML algorithms.

2. **Performance optimization:** both speed and memory usage are optimized, by
   utilizing parallelism (with Joblib) and sparse CSR matrices (with SciPy). Heavy
   computation is typically relegated to C++ code of RDKit.

3. **Feature-rich:** in addition to computing fingerprints, you can load popular
   benchmark  datasets (e.g. from MoleculeNet), perform splitting (e.g. scaffold
   split), generate conformers, and optimize hyperparameters with optimized cross-validation.

4. **Well-documented:** each public function and class has extensive documentation,
   including relevant implementation details, caveats, and literature references.

5. **Extensibility:** any functionality can be easily modified or extended by
   inheriting from existing classes.

6. **High code quality:** pre-commit hooks scan each commit for code quality (e.g. `black`,
   `flake8`), typing (`mypy`), and security (e.g. `bandit`, `safety`). CI/CD process with
   GitHub Actions also includes over 250 unit and integration tests.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details on our code of
conduct, and the process for submitting pull requests to us.

## Citing

If you use scikit-fingerprints in your work, please cite our main publication, 
[available on SoftwareX (open access)](https://www.sciencedirect.com/science/article/pii/S2352711024003145):
```
@article{scikit_fingerprints,
   title = {Scikit-fingerprints: Easy and efficient computation of molecular fingerprints in Python},
   author = {Jakub Adamczyk and Piotr Ludynia},
   journal = {SoftwareX},
   volume = {28},
   pages = {101944},
   year = {2024},
   issn = {2352-7110},
   doi = {https://doi.org/10.1016/j.softx.2024.101944},
   url = {https://www.sciencedirect.com/science/article/pii/S2352711024003145},
   keywords = {Molecular fingerprints, Chemoinformatics, Molecular property prediction, Python, Machine learning, Scikit-learn},
}
```

Its preprint is also [available on ArXiv](https://arxiv.org/abs/2407.13291).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
