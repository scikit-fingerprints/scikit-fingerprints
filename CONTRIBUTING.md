# Contributing to Effective Molecule Fingerprints library

Thank you for your interest in contributing to Effective Molecule Fingerprints library! We welcome contributions from the community to make our project even better.

## Getting Started

Before you start contributing, please take a moment to review the following guidelines to ensure a smooth and productive collaboration.


### Reporting Issues

If you find a bug, have a question, or want to propose a new feature, please open an issue on our [GitHub repository](https://github.com/Arch4ngel21/scikit-fingerprints.git). Be sure to include as much detail as possible, such as a clear description, steps to reproduce the issue, and any relevant screenshots or error messages.

### Pull Requests

We encourage you to submit pull requests (PRs) to improve our project. To do so, follow these steps:

1. Fork the repository on GitHub.

2. Clone your forked repository to your local machine:
   ```bash
   git clone https://github.com/Arch4ngel21/scikit-fingerprints.git

3. If you don't have it, install [Poetry](https://python-poetry.org/):

4. Install dev dependencies in the main directory of the repository:
   ```bash
   poetry install --with dev
   
5. Install pre-commit hooks
   ```bash
   poetry run pre-commit install
   ```

6. Create a new branch for your feature or bug fix:

    ```bash
    git checkout -b my-feature
   
7. Make your changes and commit them with clear and concise commit messages.

8. Push your changes to your forked repository:

    ```bash
    git push origin my-feature
9. Open a pull request on our GitHub repository and provide a detailed description of your changes.

10. Ensure that your PR follows our coding style and conventions.

11. Participate in the review process by addressing any feedback or comments on your PR.

### Development Setup
To set up a development environment for Effective Molecule Fingerprints library, follow the installation instructions in the README.

### Testing
Before submitting a pull request, please make sure to run the tests to ensure that your changes do not introduce regressions. If necessary, add new tests to cover your code. Also please be sure that you do not violate any code style requirements (you can check it simply by running "flake8" in the CLI).

### Documentation
If you are contributing new features or changes, please update the documentation to reflect your changes. You can find the documentation in the `docs` directory.
To build the documentation, run the following command in the main directory of the repository:
```bash
poetry run make -C docs html
```

To view the documentation, open the `docs/_build/html/index.html` file in your browser.

### Code of Conduct
By participating in this project, you agree to abide by the Code of Conduct. We take violations seriously and may take action if necessary to maintain a welcoming and respectful community.

## Licensing
By contributing to this project, you agree that your contributions will be licensed under the MIT License.

Thank you for your contribution! Your efforts help make Effective Molecule Fingerprints library better for everyone.