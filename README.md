# scikit-fingerprints

[![PyPI version](https://badge.fury.io/py/scikit-fingerprints.svg)](https://badge.fury.io/py/scikit-fingerprints)
[![](https://img.shields.io/pypi/dm/scikit-fingerprints)](https://pypi.org/project/scikit-fingerprints/)
[![Downloads](https://static.pepy.tech/badge/scikit-fingerprints)](https://pepy.tech/project/scikit-fingerprints)
![Libraries.io dependency status for latest release](https://img.shields.io/librariesio/release/pypi/scikit-fingerprints)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
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
- [Basic Usage](#basic-usage)
- [General Project Vision](#general-project-vision)
- [Contributing](#contributing)
- [License](#license)

---

## Description

Molecular fingerprints are crucial in various scientific fields, including drug discovery, materials science, and
chemical analysis. However, existing Python libraries for computing molecular fingerprints often lack performance,
user-friendliness, and support for modern programming standards. This project aims to address these shortcomings by
creating an efficient and accessible Python library for molecular fingerprint computation.

You can find the documentation [HERE](https://scikit-fingerprints.github.io/scikit-fingerprints/)

- The library offers various functions that accept molecule descriptors (e.g., SMILES) and fingerprint parameters,
  returning the specified fingerprints.
- It's open-source and available for installation via pip.
- The library has been designed for ease of use, minimizing the need for extensive training.
- Compatibility with the standard Python ML stack, based on scikit-learn interfaces, has been a top priority.

## Supported platforms

|                      | `python3.9`   | `python3.10` | `python3.11` | `python3.12` |
|----------------------|---------------|--------------|--------------|--------------|
| **Ubuntu - latest**  | ✅             | ✅            | ✅            | ✅            |
| **Windows - latest** | ✅             | ✅            | ✅            | ✅            |
| **macOS - latest**   | only macOS 13 | ✅            | ✅            | ✅            |

## Installation

You can install the library using pip:

```bash
pip install scikit-fingerprints
```

## Basic Usage

Most fingerprints are based on molecular graphs (2D-based), and you can use SMILES
input directly:
```python
from skfp.fingerprints import AtomPairFingerprint

smiles_list = ["O=S(=O)(O)CCS(=O)(=O)O", "O=C(O)c1ccccc1O"]

atom_pair_fingerprint = AtomPairFingerprint()

X = atom_pair_fingerprint.transform(smiles_list)
print(X)
```

For fingerprints using conformers (3D-based), you need to create molecules first
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
from skfp.metrics import multioutput_auroc_score
from skfp.model_selection.scaffold_split import scaffold_train_test_split
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
auroc = multioutput_auroc_score(y_test, y_pred_proba)
print(f"AUROC: {auroc:.2%}")
```

## General Project Vision

The primary goal of this project was to develop a Python library that simplifies the computation of widely-used
molecular fingerprints, such as Morgan's fingerprint, MACCS fingerprint, and others. This library has the following key
features:

- **User-Friendly Interface:** The library was designed to provide an intuitive interface, making it easy to integrate
  into machine learning workflows.

- **Performance Optimization:** We implemented molecular fingerprint computation algorithms using concurrent programming
  techniques to maximize performance. Large datasets of molecules are processed in parallel for improved efficiency.

- **Compatibility:** The library's interface was inspired by popular data science libraries like Scikit-Learn, ensuring
  compatibility and familiarity for users familiar with these tools.

- **Extensibility:** Users should be able to customize and extend the library to suit their specific needs.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details on our code of
conduct, and the process for submitting pull requests to us.

## Citing

If you use scikit-fingerprints in your work, please cite [our paper, available on ArXiv](https://arxiv.org/abs/2407.13291):
```
@misc{scikit-fingeprints,
      title={Scikit-fingerprints: easy and efficient computation of molecular fingerprints in Python}, 
      author={Jakub Adamczyk and Piotr Ludynia},
      year={2024},
      eprint={2407.13291},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2407.13291}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
