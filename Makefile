python: venv
	VIRTUAL_ENV=.venv maturin develop --release --no-default-features --features "models python-bindings extension-module"
clean:
	cargo clean
build:
	cargo build --all-features
venv:
	if [ ! -d "./.venv" ]; then python -m venv .venv;./.venv/bin/python3 -m pip install maturin; fi
docker-build:
	docker build -t gen .
	docker run -v .:/data --rm --name gen gen cp target/release/gen /data/gen
