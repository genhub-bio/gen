PYENV_PYTHON := $(shell pyenv which python)
PYENV_PREFIX := $(shell pyenv prefix)
VENV := $(abspath ../.venv)
VENV_PYTHON := $(VENV)/bin/python
VENV_PYTEST := $(VENV)/bin/pytest
VENV_MATURIN := $(VENV)/bin/maturin

.PHONY: help pyenv-test notebook-test test

help:
	@echo "Available targets:"
	@echo "  pyenv-test     Run cargo tests using the pyenv Python and its library path"
	@echo "  notebook-test  Run notebook tests using the project root venv"
	@echo "  test           Run all tests (pyenv-test + notebook-test)"

pyenv-test:
	PYO3_PYTHON="$(PYENV_PYTHON)" \
	LD_LIBRARY_PATH="$(PYENV_PREFIX)/lib:$(LD_LIBRARY_PATH)" \
	DYLD_LIBRARY_PATH="$(PYENV_PREFIX)/lib:$(DYLD_LIBRARY_PATH)" \
	cargo t

notebook-test:
	$(VENV_MATURIN) develop
	$(VENV_PYTEST) --nbmake --nbmake-timeout=120 examples/

test: pyenv-test notebook-test
