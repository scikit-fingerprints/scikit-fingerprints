.PHONY: install-dev docs

install-dev:
	poetry install --with dev --sync --no-root
	poetry self add poetry-plugin-sort
	pre-commit install

docs:
	$(MAKE) -C docs clean html
