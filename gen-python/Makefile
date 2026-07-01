PYENV_PYTHON := $(shell pyenv which python)
PYENV_PREFIX := $(shell pyenv prefix)
VENV := $(abspath ../.venv)
VENV_PYTHON := $(VENV)/bin/python
VENV_PYTEST := $(VENV)/bin/pytest
VENV_JUPYTER := $(VENV)/bin/jupyter
VENV_PIP := $(VENV)/bin/pip
VENV_PLAYWRIGHT := $(VENV)/bin/playwright

.PHONY: help pyenv-test notebook-clear-output notebook-test notebook-freeze test

help:
	@echo "Available targets:"
	@echo "  pyenv-test            Run cargo tests using the pyenv Python and its library path"
	@echo "  notebook-clear-output Clear outputs from example notebooks in place"
	@echo "  notebook-test         Build via the root Makefile's 'python' target, then run notebook tests"
	@echo "  notebook-freeze       Execute example notebooks in a real browser and freeze widgets to static PNGs"
	@echo "  test                  Run all tests (pyenv-test + notebook-test)"

pyenv-test:
	PYO3_PYTHON="$(PYENV_PYTHON)" \
	LD_LIBRARY_PATH="$(PYENV_PREFIX)/lib:$(LD_LIBRARY_PATH)" \
	DYLD_LIBRARY_PATH="$(PYENV_PREFIX)/lib:$(DYLD_LIBRARY_PATH)" \
	cargo t

notebook-clear-output:
	$(VENV_JUPYTER) nbconvert --clear-output --inplace examples/*.ipynb

notebook-test:
	$(MAKE) -C .. python
	$(VENV_PYTEST) --nbmake --nbmake-timeout=120 examples/

# Renders widget outputs as static PNGs by driving a real Jupyter server through a
# headless browser, so GitHub can display the notebooks without live JS. Run this
# (and commit the result) only when preparing example notebooks for viewing; use
# notebook-clear-output beforehand for routine commits so diffs stay small.
notebook-freeze:
	$(VENV_PIP) show playwright >/dev/null 2>&1 || $(VENV_PIP) install playwright
	$(VENV_PLAYWRIGHT) install --with-deps chromium
	$(MAKE) -C .. jupyter
	$(VENV_PYTHON) scripts/freeze_notebooks.py

test: notebook-clear-output pyenv-test notebook-test
