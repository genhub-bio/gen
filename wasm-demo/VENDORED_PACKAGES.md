# Vendored cockle wasm packages

`cockle_fs-wasm/`, `coreutils-wasm/`, `grep-wasm/`, `less-wasm/`, `sed-wasm/` (plus `gen-wasm/`,
built locally and gitignored) are prebuilt `emscripten-wasm32` assets committed here as
`local_directory`-style packages. `cockle-config.json` is a static, hand-maintained file recording
the fully-resolved package config for all six — it's copied straight into `dist/` at build time
(`postbuild:copy-assets` in `package.json`), so `@jupyterlite/cockle`'s own build-time tool
(`node_modules/@jupyterlite/cockle/lib/tools/prepare_wasm.js`, which merges config and can fetch
wasm packages via micromamba) is never invoked. `cockle-config-in.json` is kept only as an input
for the one-time/occasional re-vendoring below, not used by the normal build.

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

1. Ensure `micromamba` is on `PATH` (e.g. it ships inside a miniforge install) — only needed for
   this one-off step, not for normal builds.
2. From `wasm-demo/`, run `prepare_wasm.js` against a scratch directory so it fetches the five
   cockle packages fresh via micromamba (it only skips the fetch when every package in
   `cockle-config-in.json` is `local_directory` — temporarily remove the `local_directory` entries
   for `cockle_fs`/`coreutils`/`grep`/`less`/`sed` from `cockle-config-in.json`, keeping `gen`'s,
   to force the fetch path):
   ```
   mkdir -p /tmp/cockle-vendor-fetch
   node node_modules/@jupyterlite/cockle/lib/tools/prepare_wasm.js --copy /tmp/cockle-vendor-fetch
   ```
3. Copy the refreshed files over the vendored directories:
   ```
   for pkg in cockle_fs coreutils grep less sed; do
     cp /tmp/cockle-vendor-fetch/$pkg/* ${pkg}-wasm/
   done
   ```
4. Update the committed `cockle-config.json` and the version table above from the generated
   `/tmp/cockle-vendor-fetch/cockle-config.json` (same shape, just with real `build_string`/
   `version`/`channel` values from the fetch instead of the blank placeholders `local_directory`
   packages get).
