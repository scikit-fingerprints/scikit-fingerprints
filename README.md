# Effective Molecule Fingerprints library


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

Molecular fingerprints are crucial in various scientific fields, including drug discovery, materials science, and chemical analysis. However, existing Python libraries for computing molecular fingerprints often lack performance, user-friendliness, and support for modern programming standards. This project aims to address these shortcomings by creating an efficient and accessible Python library for molecular fingerprint computation.

## General Project Vision
 
The primary goal of this project is to develop a Python library that simplifies the computation of widely-used molecular fingerprints, such as Morgan's fingerprint, MACCS fingerprint, and others. This library will have the following key features:

- **User-Friendly Interface:** The library will provide an intuitive interface, making it easy to integrate into machine learning workflows.

- **Performance Optimization:** We will implement molecular fingerprint computation algorithms using concurrent programming techniques to maximize performance. Large datasets of molecules will be processed in parallel for improved efficiency.

- **Compatibility:** The library's interface will be inspired by popular data science libraries like Scikit-Learn, ensuring compatibility and familiarity for users familiar with these tools.

- **Extensibility:** Users will be able to customize and extend the library to suit their specific needs.

## Library Description

- The library will offer various functions that accept molecule descriptors (e.g., SMILES) and fingerprint parameters, returning the specified fingerprints.
- It will be open-source and available for installation via pip.
- Automated testing will be implemented to support continuous development and integration.
- The library will be designed for ease of use, minimizing the need for extensive training.
- Compatibility with the standard Python ML stack, based on Scikit-Learn interfaces, will be a top priority.

## Installation

You can install the library using pip:

```bash
pip install skfp
```


## Technologies Used

Our project leverages the following technologies:

- [Scikit-learn](https://scikit-learn.org): A Python library for machine learning and data analysis.

- [NumPy](https://numpy.org): A library for numerical and scientific computing.

- [Joblib](https://joblib.readthedocs.io): Used for parallel computing.

- [Rdkit](https://www.rdkit.org/): Open-Source Cheminformatics Software.

---

By contributing to this project, you can help advance the fields of chemistry and cheminformatics by providing scientists with a powerful tool for molecular structure analysis. We welcome your collaboration and feedback.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

