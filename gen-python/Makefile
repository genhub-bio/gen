PYENV_PYTHON := $(shell pyenv which python)
PYENV_PREFIX := $(shell pyenv prefix)

pyenv-test:
	$(PYENV_PYTHON) -m pip install networkx --quiet
	PYO3_PYTHON="$(PYENV_PYTHON)" \
	LD_LIBRARY_PATH="$(PYENV_PREFIX)/lib:$(LD_LIBRARY_PATH)" \
	cargo t

