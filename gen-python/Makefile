PYENV_PYTHON := $(shell pyenv which python)
PYENV_PREFIX := $(shell pyenv prefix)

.PHONY: help pyenv-test

help:
	@echo "Available targets:"
	@echo "  pyenv-test  Run cargo tests using the pyenv Python and its library path"

pyenv-test:
	PYO3_PYTHON="$(PYENV_PYTHON)" \
	LD_LIBRARY_PATH="$(PYENV_PREFIX)/lib:$(LD_LIBRARY_PATH)" \
	DYLD_LIBRARY_PATH="$(PYENV_PREFIX)/lib:$(DYLD_LIBRARY_PATH)" \
	cargo t
