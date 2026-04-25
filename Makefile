.PHONY: python jupyter release-check-js clean build clippy-fix docker-build
python:
	@[ -d .venv ] || python -m venv .venv
	@.venv/bin/pip show maturin >/dev/null 2>&1 || .venv/bin/pip install maturin
	.venv/bin/maturin develop --release --manifest-path gen-python/Cargo.toml --features extension-module
# The jupyter widget requires a bundled JS file compiled from the TypeScript sources in gen-python/js/.
# We check in the compiled jupyter_widget.js alongside the TS so npm is not required to build the widget.
jupyter: python
	@if command -v npm >/dev/null 2>&1; then \
		cd gen-python && npm install && npm run check && npm run build; \
	else \
		echo "npm not found; using committed gen-python/python/gen/static/jupyter_widget.js"; \
		test -f gen-python/python/gen/static/jupyter_widget.js || \
			(echo "Error: gen-python/python/gen/static/jupyter_widget.js missing. Install npm and run 'make jupyter'." && exit 1); \
	fi
	.venv/bin/maturin develop --release --manifest-path gen-python/Cargo.toml --features extension-module --extras jupyter
release-check-js:
	cd gen-python && npm ci && npm run check && npm run build
	@echo "Verifying committed jupyter_widget.js/.map match build output..."
	git diff --exit-code gen-python/python/gen/static/jupyter_widget.js gen-python/python/gen/static/jupyter_widget.js.map
clean:
	cargo clean
build:
	cargo build --all-features
clippy-fix:
	cargo clippy --all-targets --all-features --allow-dirty --allow-staged --fix -- -D clippy::all
docker-build:
	docker build -t gen .
	docker run -v .:/data --rm --name gen gen cp target/release/gen /data/gen
