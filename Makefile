.PHONY: setup docs doctest test test-coverage help
.DEFAULT_GOAL := help

setup: ## Install development dependencies
	@# check if uv is installed
	@uv --version >/dev/null 2>&1 || (echo "uv is not installed, please install it" && exit 1)

	@# check if pandoc (for docs) is installed
	@pandoc --version >/dev/null 2>&1 || (echo "Pandoc is not installed. Please install it from https://pandoc.org/" && exit 1)

	@# install dependencies
	uv sync --group dev --group docs
	uv run pre-commit install

docs: ## Re-generate documentation
	uv run $(MAKE) -C docs html

doctest: docs ## Run documentation tests
	uv run $(MAKE) -C docs doctest

test: ## Run tests
	uv run ruff check
	uv run pytest tests

test-coverage: ## Run tests and calculate test coverage
	-mkdir .tmp_coverage_files
	uv run pytest --cov=skfp tests
	-rm -rf .tmp_coverage_files

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
