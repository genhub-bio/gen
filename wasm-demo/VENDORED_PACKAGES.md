# Vendored cockle wasm packages

`cockle_fs-wasm/`, `coreutils-wasm/`, `grep-wasm/`, `less-wasm/`, `sed-wasm/` are prebuilt
`emscripten-wasm32` assets fetched once from the `emscripten-forge` conda channel and committed
here, declared as `local_directory` entries in `cockle-config-in.json`. This avoids requiring
`micromamba` on `PATH` for every `make wasm-jupyter` run — it's only needed for the one-time (or
occasional) re-vendoring below.

These correspond to `@jupyterlite/cockle` **1.7.0** (see `package.json`), with these exact
package versions:

| package     | version | build_string |
|-------------|---------|--------------|
| cockle_fs   | 0.3.0   | h8b79025_1   |
| coreutils   | 9.10    | h072c4ef_2   |
| grep        | 3.12    | h8b79025_0   |
| less        | 693     | hf259948_0   |
| sed         | 4.9     | h072c4ef_0   |

They are pinned manually, not auto-updated — if `@jupyterlite/cockle` is ever bumped in
`package.json`, re-vendor them to match:

1. Temporarily redeclare the four packages without `local_directory` in `cockle-config-in.json`
   (or just delete those entries — `cockle-config-base.json` already lists them as wasm
   packages) so `prepare_wasm.js` fetches them via micromamba again.
2. Ensure `micromamba` is on `PATH` (e.g. it ships inside a miniforge install).
3. From `wasm-demo/`, run:
   ```
   mkdir -p /tmp/cockle-vendor-fetch
   node node_modules/@jupyterlite/cockle/lib/tools/prepare_wasm.js --copy /tmp/cockle-vendor-fetch
   ```
4. Copy the refreshed files over the vendored directories:
   ```
   for pkg in cockle_fs coreutils grep less sed; do
     cp /tmp/cockle-vendor-fetch/$pkg/* ${pkg}-wasm/
   done
   ```
5. Restore the `local_directory` entries in `cockle-config-in.json` and update the version table
   above from the generated `/tmp/cockle-vendor-fetch/cockle-config.json`.

Note: `patches/@jupyterlite+cockle+1.7.0.patch` (applied automatically via the `postinstall`
script and `patch-package`) makes `prepare_wasm.js` skip its micromamba search entirely when
every configured package is `local_directory` — required for `make wasm-jupyter` to run without
micromamba installed at all. If the cockle version bumps, re-generate this patch too
(`npx patch-package @jupyterlite/cockle` after re-applying the same edit to the new
`node_modules` copy of `prepare_wasm.js`).
