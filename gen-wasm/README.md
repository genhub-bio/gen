# GenHub browser repository actions

`gen-browser-actions` builds a small Gen Rust executable as a JavaScript module and WebAssembly. Its typed API initializes an empty repository, creates a branch from a selected source, and merges a source branch into a target. Each action calls Gen history and remote operations directly and does not execute notebook Python.

Build the module using the same Pyodide Emscripten environment as the browser Gen wheel:

```sh
gen-python/build-browser-actions.sh /path/to/output
```

The output directory contains `gen_actions.js` and `gen_actions.wasm`. The default export creates an action client:

```js
import createGenActions from "./gen_actions.js";

const actions = await createGenActions({
  serverUrl: self.location.origin,
  locateFile: (path) => new URL(`./${path}`, import.meta.url).href,
});

await actions.createRepositoryBranch({
  namespace: "alice",
  slug: "sample",
  sourceBranch: "main",
  branch: "analysis",
});
```

Browser builds must resolve `rusqdoltlite` from a `browser-cloud` source that includes the `bcvLocalClientCount` pointer-subtraction fix. GenHub's notebook runtime build injects its selected `rusqdoltlite` context through Cargo configuration; keep that source context current with the fix, since the older pinned revision traps while opening a graph for push.

The client exposes `initializeRepository({ namespace, slug, branch? })`, `createRepositoryBranch({ namespace, slug, sourceBranch, branch })`, and `mergeRepositoryBranches({ namespace, slug, sourceBranch, targetBranch })`. Results contain `branch` and `commitHash`. Initialization reads GenHub's configured default branch and checks that the repository is uninitialized and has no listed branches.

The factory and action methods reject with `GenRepositoryActionError`. Gen sanitizes diagnostic messages before rejecting and uses a generic message for unexpected startup or runtime failures. Each error includes the safe `message` and a JavaScript stack from the façade.

The JavaScript layer reads the signed-in user and repository state using same-origin session-cookie requests. Gen's existing browser remote acquires scoped CABs for graph reads and pushes, keeping signed capabilities inside Gen's remote operations.
