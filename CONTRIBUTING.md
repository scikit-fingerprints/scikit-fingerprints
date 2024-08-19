# Contributing to scikit-fingerprints library

Thank you for your interest in contributing to scikit-fingerprints library! We welcome contributions from
the community to make our project even better.

## Getting Started

Before you start contributing, please take a moment to review the following guidelines to ensure a smooth and productive
collaboration.

### Reporting Issues

If you find a bug, have a question, or want to propose a new feature, please open an issue on
our [GitHub repository](https://github.com/scikit-fingerprints/scikit-fingerprints.git). Be sure to include as much detail as
possible, such as a clear description, steps to reproduce the issue, and any relevant screenshots or error messages.

### Pull Requests

We encourage you to submit pull requests (PRs) to improve our project. To do so, follow these steps:

1. Setup your development environment by following the [instructions](#development-setup-linux)

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

### Development Setup (Linux)

Prerequisites:

- Python 3.9 or higher
- Poetry
- Git

To set up the development environment, follow these steps:

1. Fork the repository on GitHub.
2. Clone your forked repository to your local machine:
   ```bash
   git clone <repository-url>
   ```
3. Run make to set up the development environment:
   ```bash
   make setup
   ```
   This command will set up a virtual environment and install all the necessary dependencies.
4. That's it! You're ready to start developing. For more convenience, you can run the following command to activate the
   virtual environment:
   ```bash
   poetry shell
   ```

---
**NOTE**

For other operating systems, please refer to Makefile for the commands to run.

---

### Testing

Before submitting a pull request, make sure to run the tests to ensure that your changes do not introduce
regressions.
To run tests, execute `make test` in the main directory of the repository.
If necessary, add new tests to cover your code. Also, please be sure that you do not violate any code style
requirements (you can check it by running pre-commit on staged files).

### Documentation

If you are contributing new features or changes, please update the documentation to reflect your changes. You can find
the documentation in the `docs` directory.

Run `make docs` in the main directory of the repository to build the documentation. This command will generate HTML
files in the `docs/_build/html` directory.

To view the documentation, open the `docs/_build/html/index.html` file in your browser.

### Releasing
To release scikit-fingerprints open a GitHub release with tag named 'vA.B.C'
where each letter stands for version number. Fill the release notes and submit the release. Then, the version will be automatically sourced from tag by GH action and released to PyPI.

### Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). We take violations seriously and may take
action if necessary to maintain a welcoming and respectful community.

## Licensing

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

Thank you for your contribution! Your efforts help make scikit-fingerprints library better for everyone.