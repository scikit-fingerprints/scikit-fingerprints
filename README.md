# scikit-fingerprints

A Python library for efficient computation of molecular fingerprints

Click [HERE](https://scikit-fingerprints.github.io/scikit-fingerprints/) to see the Documentation.

## Table of Contents

- [Description](#description)
- [General Project Vision](#general-project-vision)
- [Library Description](#library-description)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## Description

Molecular fingerprints are crucial in various scientific fields, including drug discovery, materials science, and
chemical analysis. However, existing Python libraries for computing molecular fingerprints often lack performance,
user-friendliness, and support for modern programming standards. This project aims to address these shortcomings by
creating an efficient and accessible Python library for molecular fingerprint computation.

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

## Library Description

- The library offers various functions that accept molecule descriptors (e.g., SMILES) and fingerprint parameters,
  returning the specified fingerprints.
- It's open-source and available for installation via pip.
- The library has been designed for ease of use, minimizing the need for extensive training.
- Compatibility with the standard Python ML stack, based on Scikit-Learn interfaces, has been a top priority.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details on our code of
conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

