export function render({ model, el }) {
  el.hidden = true;
  let announcedCommit = "";
  let capturedRequest = "";

  const announce = () => {
    const branch = model.get("branch");
    const commitHash = model.get("commit_hash");
    if (!branch || !commitHash || commitHash === announcedCommit) {
      return;
    }

    announcedCommit = commitHash;
    if (window.parent !== window) {
      window.parent.postMessage(
        {
          type: "genhub:notebook-committed",
          branch,
          commit_hash: commitHash,
        },
        window.location.origin,
      );
    }
  };

  const captureNotebook = async () => {
    const requestId = model.get("capture_request_id");
    if (!requestId || requestId === capturedRequest) {
      return;
    }
    capturedRequest = requestId;

    try {
      const panel = window.jupyterapp?.shell?.currentWidget;
      const notebookModel = panel?.model;
      if (typeof notebookModel?.toJSON !== "function") {
        throw new Error("Select an open notebook before saving it.");
      }
      const notebook = notebookModel.toJSON();
      model.set({
        capture_notebook_json: JSON.stringify(notebook),
        capture_error: "",
        capture_response_id: requestId,
      });
    } catch (error) {
      model.set({
        capture_notebook_json: "",
        capture_error: String(error),
        capture_response_id: requestId,
      });
    }
    model.save_changes();
  };

  model.on("change:branch", announce);
  model.on("change:commit_hash", announce);
  model.on("change:capture_request_id", captureNotebook);
  captureNotebook();
  return () => {
    model.off("change:branch", announce);
    model.off("change:commit_hash", announce);
    model.off("change:capture_request_id", captureNotebook);
  };
}
