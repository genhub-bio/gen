PYENV_PYTHON := $(shell pyenv which python)
PYENV_PREFIX := $(shell pyenv prefix)
VENV := $(abspath ../.venv)
VENV_PYTHON := $(VENV)/bin/python
VENV_PYTEST := $(VENV)/bin/pytest
VENV_JUPYTER := $(VENV)/bin/jupyter
VENV_PIP := $(VENV)/bin/pip
VENV_PLAYWRIGHT := $(VENV)/bin/playwright
STUB_FILE := python/gen/gen/__init__.pyi

.PHONY: help bindings-test api-test notebook-clear-output notebook-test notebook-freeze test stubs stubs-check

help:
	@echo "Available targets:"
	@echo "  bindings-test         Run Rust binding tests using the pyenv Python and its library path"
	@echo "  api-test              Build via the root Makefile's 'python' target, then exercise the"
	@echo "                        installed extension's public API (tests/*.py)"
	@echo "  notebook-clear-output Clear outputs from example notebooks in place"
	@echo "  notebook-test         Build via the root Makefile's 'python' target, then run notebook tests"
	@echo "  notebook-freeze       Execute example notebooks in a real browser and freeze widgets to static PNGs"
	@echo "  stubs                 Regenerate the Python type stubs from the Rust annotations"
	@echo "  stubs-check           Fail if the committed stubs are stale (leaves files untouched)"
	@echo "  test                  Run all tests (bindings-test + api-test + notebook-test)"

bindings-test:
	PYO3_PYTHON="$(PYENV_PYTHON)" \
	LD_LIBRARY_PATH="$(PYENV_PREFIX)/lib:$(LD_LIBRARY_PATH)" \
	DYLD_LIBRARY_PATH="$(PYENV_PREFIX)/lib:$(DYLD_LIBRARY_PATH)" \
	cargo t

# Regenerates $(STUB_FILE) from the #[gen_stub_*] annotations and doc comments in src/. Fix a wrong
# stub in the Rust source (override_type, skip, docs), not in the .pyi, which is overwritten.
stubs:
	cargo run --features abi3 --bin stub_gen
	ruff format $(STUB_FILE)

# Fails when the committed stub is stale, without leaving the regenerated file behind.
stubs-check:
	cp $(STUB_FILE) $(STUB_FILE).committed
	$(MAKE) stubs
	@if cmp -s $(STUB_FILE) $(STUB_FILE).committed; then \
		rm $(STUB_FILE).committed; echo "stubs are up to date"; \
	else \
		diff -u $(STUB_FILE).committed $(STUB_FILE) | head -40; \
		mv $(STUB_FILE).committed $(STUB_FILE); \
		echo "stubs are stale: run 'make -C gen-python stubs' and commit the result"; exit 1; \
	fi

api-test:
	$(MAKE) -C .. python
	$(VENV_PYTHON) -m unittest discover -s tests -v

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

test: notebook-clear-output bindings-test api-test notebook-test
