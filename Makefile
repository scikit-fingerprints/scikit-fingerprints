.PHONY: install-dev docs help
.DEFAULT_GOAL := help

install-dev: ## Install development dependencies, pre-commit hooks and poetry plugin
	# check if poetry is installed
	poetry --version || (echo "Poetry is not installed. Please install it from https://python-poetry.org/docs/#installation" && exit 1)
	poetry install --with dev,doc --sync --no-root
	poetry self add poetry-plugin-sort
	pre-commit install

docs: ## Re-generate documentation
	$(MAKE) -C docs clean html

test-coverage: ## Run tests and calculate test coverage
	-mkdir .tmp_coverage_files
	pytest --cov=skfp tests
	-rm -rf .tmp_coverage_files

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
