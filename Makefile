.PHONY: setup docs test test-coverage help
.DEFAULT_GOAL := help

setup: ## Install development dependencies, pre-commit hooks and poetry plugin
	# check if poetry is installed
	poetry --version || (echo "Poetry is not installed. Please install it from https://python-poetry.org/docs/#installation" && exit 1)
	poetry install --with dev,doc --sync
	poetry self add poetry-plugin-export
	poetry self add poetry-plugin-sort
	poetry run pre-commit install

docs: ## Re-generate documentation
	-rm -r docs/modules/generated
	poetry run $(MAKE) -C docs clean html doctest

test: ## Run tests
	poetry run black . --check --diff
	poetry run isort . --check-only --profile black
	poetry sort --check
	poetry run pytest tests

test-coverage: ## Run tests and calculate test coverage
	-mkdir .tmp_coverage_files
	poetry run pytest --cov=skfp tests
	-rm -rf .tmp_coverage_files

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
