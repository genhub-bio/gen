python: venv
	VIRTUAL_ENV=.venv maturin develop --release --manifest-path gen-python/Cargo.toml --features extension-module
clean:
	cargo clean
build:
	cargo build --all-features
clippy-fix:
	cargo clippy --all-targets --all-features --allow-dirty --allow-staged --fix -- -D clippy::all
venv:
	if [ ! -d "./.venv" ]; then python -m venv .venv;./.venv/bin/python3 -m pip install maturin; fi
docker-build:
	docker build -t gen .
	docker run -v .:/data --rm --name gen gen cp target/release/gen /data/gen
