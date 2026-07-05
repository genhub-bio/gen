PYENV_PYTHON := $(shell pyenv which python)
PYENV_PREFIX := $(shell pyenv prefix)
VENV := $(abspath ../.venv)
VENV_PYTHON := $(VENV)/bin/python
VENV_PYTEST := $(VENV)/bin/pytest
VENV_JUPYTER := $(VENV)/bin/jupyter

.PHONY: help pyenv-test notebook-clear-output notebook-test test

help:
	@echo "Available targets:"
	@echo "  pyenv-test            Run cargo tests using the pyenv Python and its library path"
	@echo "  notebook-clear-output Clear outputs from example notebooks in place"
	@echo "  notebook-test         Build via the root Makefile's 'python' target, then run notebook tests"
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

test: notebook-clear-output pyenv-test notebook-test
