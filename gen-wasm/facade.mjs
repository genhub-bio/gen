const genActionsWasmFactory = createGenActionsWasm;

const requireAction = (action) => {
  if (typeof action !== "object" || action === null || Array.isArray(action)) {
    throw new TypeError("A repository action requires an options object");
  }
  return action;
};

const repositoryPath = (namespace, slug) =>
  `/api/repos/${encodeURIComponent(namespace)}/${encodeURIComponent(slug)}`;

const readJson = async (url, errorMessage) => {
  const response = await fetch(url, {
    credentials: "same-origin",
    cache: "no-store",
    redirect: "error",
  });
  if (!response.ok) throw new Error(errorMessage);
  return response.json();
};

const currentUser = async (serverUrl) => {
  const user = await readJson(
    new URL("/api/user", serverUrl),
    "Sign in to GenHub before changing this repository",
  );
  if (
    typeof user !== "object" ||
    user === null ||
    typeof user.name !== "string" ||
    typeof user.username !== "string"
  ) {
    throw new Error("GenHub did not return the signed-in user identity");
  }
  return {
    committer: user.name,
    email: `${user.username}@users.noreply.genhub.bio`,
  };
};

const repositoryDetails = async (serverUrl, namespace, slug) => {
  const path = repositoryPath(namespace, slug);
  const repository = await readJson(
    new URL(path, serverUrl),
    "Could not read the repository details",
  );
  if (typeof repository !== "object" || repository === null) {
    throw new Error("GenHub returned invalid repository details");
  }
  const branches = await readJson(
    new URL(`${path}/branches`, serverUrl),
    "Could not check the repository branches",
  );
  if (!Array.isArray(branches) || !branches.every((branch) => typeof branch === "string")) {
    throw new Error("GenHub returned an invalid branch list");
  }
  return {
    defaultBranch: typeof repository.default_branch === "string" ? repository.default_branch : "main",
    initialized: repository.initialized === true,
    branches,
  };
};

export async function createGenActions(options = {}) {
  if (typeof options.serverUrl !== "string" || options.serverUrl.length === 0) {
    throw new TypeError("GenHub's application URL is required");
  }
  const appUrl = new URL(options.serverUrl);
  const serverUrl = appUrl.origin;
  if (
    !new Set(["http:", "https:"]).has(appUrl.protocol) ||
    options.serverUrl !== serverUrl ||
    serverUrl !== self.location.origin
  ) {
    throw new TypeError("GenHub's application URL must match the browser origin");
  }
  const user = await currentUser(serverUrl);
  const wasm = await genActionsWasmFactory({
    ...options,
    locateFile: (path, prefix) => {
      const wasmPath = path.endsWith("gen_browser_actions.wasm")
        ? "gen_actions.wasm"
        : path;
      return options.locateFile
        ? options.locateFile(wasmPath, prefix)
        : new URL(wasmPath, prefix || import.meta.url).href;
    },
  });

  const run = (command, action) => {
    const request = JSON.stringify({
      ...requireAction(action),
      command,
      serverUrl,
      ...user,
    });
    const requestPointer = wasm.stringToNewUTF8(request);
    let responsePointer = 0;
    try {
      responsePointer = wasm._genhub_repository_command(requestPointer);
      if (responsePointer === 0) {
        throw new Error("Gen could not complete the repository action");
      }
      const response = JSON.parse(wasm.UTF8ToString(responsePointer));
      if (response.ok !== true || !response.result) {
        throw new Error(
          typeof response.error === "string"
            ? response.error
            : "Gen could not complete the repository action",
        );
      }
      return response.result;
    } finally {
      if (responsePointer !== 0) wasm._genhub_free_string(responsePointer);
      wasm._free(requestPointer);
    }
  };

  return {
    initializeRepository: async (action) => {
      const actionOptions = requireAction(action);
      const details = await repositoryDetails(
        serverUrl,
        actionOptions.namespace,
        actionOptions.slug,
      );
      if (details.initialized || details.branches.length > 0) {
        throw new Error("Only an empty repository can be initialized");
      }
      if (actionOptions.branch && actionOptions.branch !== details.defaultBranch) {
        throw new Error("Repository initialization must use its configured default branch");
      }
      return run("initializeRepository", {
        ...actionOptions,
        branch: details.defaultBranch,
      });
    },
    createRepositoryBranch: async (action) => run("createRepositoryBranch", action),
    mergeRepositoryBranches: async (action) => run("mergeRepositoryBranches", action),
  };
}

export default createGenActions;
