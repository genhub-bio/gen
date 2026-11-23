const genActionsWasmFactory = createGenActionsWasm;
const genericActionError = "Gen repository action failed";
const safeActionErrors = new WeakSet();

const sanitizeRepositoryActionMessage = (message) =>
  message
    .replace(/[\u0000-\u001f\u007f]/gu, " ")
    .replace(/\bBearer\s+[^\s]+/giu, "Bearer [redacted]")
    .replace(/\b(?:https?|gcs):\/\/[^\s"'<>]+/giu, (value) => {
      try {
        const url = new URL(value);
        const pathname = url.pathname.replace(
          /(\/dolt\/)[^/]+(?=\/(?:publish|abort)(?:\/|$))/giu,
          "$1[redacted]",
        );
        const hasPrivateUrlData =
          url.search || url.hash || url.username || url.password;
        if (!hasPrivateUrlData && pathname === url.pathname) {
          return value;
        }

        const origin =
          url.origin === "null" ? `${url.protocol}//${url.host}` : url.origin;
        return origin + pathname + (hasPrivateUrlData ? "?[redacted]" : "");
      } catch {
        return "[remote resource]";
      }
    })
    .replace(
      /((?:capability|access[_-]?token|token|credential|signature|sig|api[_-]?key|password)=)[^&\s]+/giu,
      "$1[redacted]",
    )
    .trim() || genericActionError;

export class GenRepositoryActionError extends Error {
  constructor(message) {
    super(
      typeof message === "string"
        ? sanitizeRepositoryActionMessage(message)
        : genericActionError,
    );
    this.name = "GenRepositoryActionError";
    safeActionErrors.add(this);
  }
}

const exposeSafeErrors = (operation) => async (...args) => {
  try {
    return await operation(...args);
  } catch (error) {
    if (safeActionErrors.has(error)) {
      throw error;
    }
    throw new GenRepositoryActionError(genericActionError);
  }
};

const requireAction = (action) => {
  if (typeof action !== "object" || action === null || Array.isArray(action)) {
    throw new GenRepositoryActionError(
      "A repository action requires an options object",
    );
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
  if (!response.ok) throw new GenRepositoryActionError(errorMessage);
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
    throw new GenRepositoryActionError(
      "GenHub did not return the signed-in user identity",
    );
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
    throw new GenRepositoryActionError("GenHub returned invalid repository details");
  }
  const branches = await readJson(
    new URL(`${path}/branches`, serverUrl),
    "Could not check the repository branches",
  );
  if (
    !Array.isArray(branches) ||
    !branches.every((branch) => typeof branch === "string")
  ) {
    throw new GenRepositoryActionError("GenHub returned an invalid branch list");
  }
  return {
    defaultBranch:
      typeof repository.default_branch === "string"
        ? repository.default_branch
        : "main",
    initialized: repository.initialized === true,
    branches,
  };
};

export const createGenActions = exposeSafeErrors(async (options = {}) => {
  if (
    !options ||
    typeof options.serverUrl !== "string" ||
    options.serverUrl.length === 0
  ) {
    throw new GenRepositoryActionError("GenHub's application URL is required");
  }

  let appUrl;
  try {
    appUrl = new URL(options.serverUrl);
  } catch {
    throw new GenRepositoryActionError(
      "GenHub's application URL must match the browser origin",
    );
  }
  const serverUrl = appUrl.origin;
  if (
    !new Set(["http:", "https:"]).has(appUrl.protocol) ||
    options.serverUrl !== serverUrl ||
    serverUrl !== self.location.origin
  ) {
    throw new GenRepositoryActionError(
      "GenHub's application URL must match the browser origin",
    );
  }

  const user = await currentUser(serverUrl);
  const wasm = await genActionsWasmFactory({
    serverUrl,
    locateFile: (path, prefix) => {
      const wasmPath = path.endsWith("gen_browser_actions.wasm")
        ? "gen_actions.wasm"
        : path;
      return options.locateFile
        ? options.locateFile(wasmPath, prefix)
        : new URL(wasmPath, prefix || import.meta.url).href;
    },
  });

  const run = async (command, action) => {
    const actionOptions = requireAction(action);
    const request = JSON.stringify({
      ...actionOptions,
      command,
      serverUrl,
      ...user,
    });
    const requestPointer = wasm.stringToNewUTF8(request);
    let responsePointer = 0;
    try {
      responsePointer = wasm._genhub_repository_command(requestPointer);
      if (responsePointer === 0) {
        throw new GenRepositoryActionError(
          "Gen could not complete the repository action",
        );
      }
      const response = JSON.parse(wasm.UTF8ToString(responsePointer));
      if (response.ok !== true || !response.result) {
        throw new GenRepositoryActionError(
          typeof response.error === "string"
            ? response.error
            : genericActionError,
        );
      }
      return response.result;
    } finally {
      if (responsePointer !== 0) wasm._genhub_free_string(responsePointer);
      wasm._free(requestPointer);
    }
  };

  return {
    initializeRepository: exposeSafeErrors(async (action) => {
      const actionOptions = requireAction(action);
      const details = await repositoryDetails(
        serverUrl,
        actionOptions.namespace,
        actionOptions.slug,
      );
      if (details.initialized || details.branches.length > 0) {
        throw new GenRepositoryActionError(
          "Only an empty repository can be initialized",
        );
      }
      if (
        actionOptions.branch &&
        actionOptions.branch !== details.defaultBranch
      ) {
        throw new GenRepositoryActionError(
          "Repository initialization must use its configured default branch",
        );
      }
      return run("initializeRepository", {
        ...actionOptions,
        branch: details.defaultBranch,
      });
    }),
    createRepositoryBranch: exposeSafeErrors((action) =>
      run("createRepositoryBranch", action),
    ),
    mergeRepositoryBranches: exposeSafeErrors((action) =>
      run("mergeRepositoryBranches", action),
    ),
  };
});

export default createGenActions;
