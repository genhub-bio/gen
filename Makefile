python: venv wasm-bindgen-cli
	cargo build --manifest-path gen-python/jupyter-widget/Cargo.toml --target wasm32-unknown-unknown --release
	wasm-bindgen --target web --out-dir gen-python/jupyter-widget/pkg \
		gen-python/jupyter-widget/target/wasm32-unknown-unknown/release/jupyter_widget.wasm
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
wasm-bindgen-cli:
	command -v wasm-bindgen || cargo install wasm-bindgen-cli
docker-build:
	docker build -t gen .
	docker run -v .:/data --rm --name gen gen cp target/release/gen /data/gen
