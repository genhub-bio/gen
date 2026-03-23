python: venv wasm-pack
	cd gen-python/jupyter-widget && wasm-pack build --target web --out-dir pkg
	./.venv/bin/python3 gen-python/build_bundle.py
	VIRTUAL_ENV=.venv maturin develop --release --manifest-path gen-python/Cargo.toml --features extension-module
clean:
	cargo clean
build:
	cargo build --all-features
clippy-fix:
	cargo clippy --all-targets --all-features --allow-dirty --allow-staged --fix -- -D clippy::all
venv:
	if [ ! -d "./.venv" ]; then python -m venv .venv; fi
	./.venv/bin/python3 -m pip install --quiet maturin anywidget
wasm-pack:
	command -v wasm-pack || cargo install wasm-pack
docker-build:
	docker build -t gen .
	docker run -v .:/data --rm --name gen gen cp target/release/gen /data/gen
