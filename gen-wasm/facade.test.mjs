import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { readFile } from "node:fs/promises";

const SERVER_URL = "https://genhub.example";
const originalGlobals = new Map(
  ["fetch", "self", "__genActionsWasmFactory"].map((name) => [
    name,
    Object.getOwnPropertyDescriptor(globalThis, name),
  ]),
);

const facadeSource = await readFile(new URL("./facade.mjs", import.meta.url), "utf8");
const facadeModuleSource = `const createGenActionsWasm = (...args) =>
  globalThis.__genActionsWasmFactory(...args);\n${facadeSource}`;
const facadeModuleUrl = `data:text/javascript;base64,${Buffer.from(
  facadeModuleSource,
).toString("base64")}`;
const facade = await import(facadeModuleUrl);

afterEach(() => {
  for (const [name, descriptor] of originalGlobals) {
    if (descriptor === undefined) {
      delete globalThis[name];
    } else {
      Object.defineProperty(globalThis, name, descriptor);
    }
  }
});

const jsonResponse = (body) => ({
  ok: true,
  json: async () => body,
});

const browserFetch = ({ initialized = false, branches = [] } = {}) =>
  async (input) => {
    const path = new URL(input).pathname;
    if (path === "/api/user") {
      return jsonResponse({ name: "Alice", username: "alice" });
    }
    if (path.endsWith("/branches")) {
      return jsonResponse(branches);
    }
    return jsonResponse({ initialized, default_branch: "main" });
  };

const wasmResponse = (response) => ({
  stringToNewUTF8: () => 1,
  _genhub_repository_command: () => 2,
  UTF8ToString: () => JSON.stringify(response),
  _genhub_free_string: () => {},
  _free: () => {},
});

const installBrowser = ({
  fetch = browserFetch(),
  wasmFactory = async () => wasmResponse({ ok: true, result: { branch: "main" } }),
} = {}) => {
  globalThis.self = { location: { origin: SERVER_URL } };
  globalThis.fetch = fetch;
  globalThis.__genActionsWasmFactory = wasmFactory;
};

const createClient = (options = {}) =>
  facade.createGenActions({
    serverUrl: SERVER_URL,
    locateFile: (path) => `https://genhub.example/notebook/${path}`,
    ...options,
  });

const assertSafeRejection = async (promise, expectedMessage, secretValues = []) => {
  await assert.rejects(promise, (error) => {
    assert.ok(error instanceof facade.GenRepositoryActionError);
    assert.equal(error.name, "GenRepositoryActionError");
    assert.equal(error.message, expectedMessage);
    assert.equal(Object.hasOwn(error, "cause"), false);

    const serializedError = JSON.stringify({
      name: error.name,
      message: error.message,
      stack: error.stack,
    });
    for (const secretValue of secretValues) {
      assert.equal(error.message.includes(secretValue), false);
      assert.equal(error.stack.includes(secretValue), false);
      assert.equal(serializedError.includes(secretValue), false);
    }
    return true;
  });
};

test("test_create_gen_actions_rejects_invalid_origins_safely", async () => {
  installBrowser();

  await assertSafeRejection(
    facade.createGenActions({
      serverUrl:
        "https://private-user:private-password@invalid.example?sig=private-signature",
    }),
    "GenHub's application URL must match the browser origin",
    ["private-user", "private-password", "private-signature"],
  );
});

test("test_initialization_preflight_preserves_its_validation_message", async () => {
  installBrowser({ fetch: browserFetch({ initialized: true, branches: ["main"] }) });
  const actions = await createClient();

  await assertSafeRejection(
    actions.initializeRepository({ namespace: "alice", slug: "sample" }),
    "Only an empty repository can be initialized",
  );
  await assertSafeRejection(
    actions.createRepositoryBranch(null),
    "A repository action requires an options object",
  );
});

test("test_all_action_methods_sanitize_wasm_response_errors", async () => {
  const rawError =
    "Push failed at gcs://private-bucket/graph?X-Goog-Signature=remote-secret and https://storage.example/api/dolt/cab-secret/publish?access_token=access-secret with Bearer private-token";
  const safeError =
    "Push failed at gcs://private-bucket/graph?[redacted] and https://storage.example/api/dolt/[redacted]/publish?[redacted] with Bearer [redacted]";
  installBrowser({
    wasmFactory: async () => wasmResponse({ ok: false, error: rawError }),
  });
  const actions = await createClient();

  const actionRequests = [
    ["initializeRepository", { namespace: "alice", slug: "sample" }],
    [
      "createRepositoryBranch",
      {
        namespace: "alice",
        slug: "sample",
        sourceBranch: "main",
        branch: "analysis",
      },
    ],
    [
      "mergeRepositoryBranches",
      {
        namespace: "alice",
        slug: "sample",
        sourceBranch: "analysis",
        targetBranch: "main",
      },
    ],
  ];

  for (const [method, action] of actionRequests) {
    await assertSafeRejection(actions[method](action), safeError, [
      "remote-secret",
      "cab-secret",
      "access-secret",
      "private-token",
    ]);
  }
});

test("test_startup_failures_are_generic_and_runtime_callbacks_are_not_forwarded", async () => {
  const secretValues = ["runtime-secret", "cause-secret"];
  const rawError = new Error(`WASM startup failed with ${secretValues[0]}`, {
    cause: new Error(secretValues[1]),
  });
  let wasmOptions;
  installBrowser({
    wasmFactory: async (options) => {
      wasmOptions = options;
      throw rawError;
    },
  });
  const onAbort = () => {
    throw new Error("The raw Emscripten callback should not be forwarded");
  };

  await assertSafeRejection(
    createClient({ onAbort, preRun: [onAbort] }),
    "Gen repository action failed",
    secretValues,
  );
  assert.deepEqual(Object.keys(wasmOptions).sort(), ["locateFile", "serverUrl"]);
});

test("test_unexpected_wasm_runtime_exceptions_are_generic", async () => {
  const secretValues = ["runtime-secret", "resource-secret"];
  installBrowser({
    wasmFactory: async () => ({
      stringToNewUTF8: () => 1,
      _genhub_repository_command: () => {
        throw new Error(
          `runtime failed with ${secretValues[0]} at https://storage.example/file?sig=${secretValues[1]}`,
        );
      },
      _genhub_free_string: () => {},
      _free: () => {},
    }),
  });
  const actions = await createClient();

  await assertSafeRejection(
    actions.createRepositoryBranch({
      namespace: "alice",
      slug: "sample",
      sourceBranch: "main",
      branch: "analysis",
    }),
    "Gen repository action failed",
    secretValues,
  );
});

test("test_unexpected_initialization_preflight_errors_are_generic", async () => {
  const secretValues = ["request-secret", "signed-secret"];
  installBrowser({
    fetch: async (input) => {
      if (new URL(input).pathname === "/api/user") {
        return jsonResponse({ name: "Alice", username: "alice" });
      }
      throw new Error(
        `request failed with ${secretValues[0]} at https://storage.example/file?sig=${secretValues[1]}`,
      );
    },
  });
  const actions = await createClient();

  await assertSafeRejection(
    actions.initializeRepository({ namespace: "alice", slug: "sample" }),
    "Gen repository action failed",
    secretValues,
  );
});
