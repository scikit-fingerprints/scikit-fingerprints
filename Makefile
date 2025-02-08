.PHONY: setup docs doctest test test-coverage help
.DEFAULT_GOAL := help

setup: ## Install development dependencies
	# check if poetry is installed
	poetry --version >/dev/null 2>&1 || (echo "Poetry is not installed. Please install it from https://python-poetry.org/docs/#installation" && exit 1)
	# check if pandoc (for docs) is installed
	pandoc --version >/dev/null 2>&1 || (echo "Pandoc is not installed. Please install it from https://pandoc.org/" && exit 1)
	# install dependencies
	poetry sync --with dev,docs
	poetry run pre-commit install

docs: ## Re-generate documentation
	-rm -r docs/modules/generated
	poetry run $(MAKE) -C docs clean html

doctest: docs ## Run documentation tests
	poetry run $(MAKE) -C docs doctest

test: ## Run tests
	poetry run ruff check
	# PyTest has memory access problems on Windows, faulthandler settings fix that
	# https://github.com/pytest-dev/pytest/issues/7634
	poetry run python -X faulthandler -m pytest tests -p no:faulthandler

test-coverage: ## Run tests and calculate test coverage
	-mkdir .tmp_coverage_files
	poetry run pytest --cov=skfp tests
	-rm -rf .tmp_coverage_files

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
