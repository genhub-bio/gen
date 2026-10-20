"""Browser-side sessions for editing a snapshot of a Gen repository."""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
import secrets
import sys
import tempfile
import unicodedata
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlsplit, urlunsplit


class NotebookSessionError(RuntimeError):
    """Raised when a notebook snapshot or publication request is invalid."""


Fetch = Callable[..., Awaitable[Any]]
_MAX_NOTEBOOK_BYTES = 8 * 1024 * 1024
_NOTEBOOK_CAPTURE_TIMEOUT = 30
_DEFAULT_NOTEBOOK_PATH = "notebooks/genhub-notebook.ipynb"


class _MultipartForm:
    def __init__(
        self,
        database: bytes,
        fields: dict[str, str],
        notebook: bytes | None = None,
        notebook_filename: str | None = None,
    ) -> None:
        self.database = database
        self.fields = fields
        self.notebook = notebook
        self.notebook_filename = notebook_filename

    def to_browser_form(self) -> Any:
        from js import Blob, FormData, Uint8Array

        form = FormData.new()
        binary = Uint8Array.new(self.database)
        form.append(
            "database",
            Blob.new([binary]),
            "notebook.db",
        )
        if self.notebook is not None:
            notebook = Uint8Array.new(self.notebook)
            form.append(
                "notebook",
                Blob.new([notebook]),
                self.notebook_filename,
            )
        for name, value in self.fields.items():
            form.append(name, value)
        return form


def _validate_notebook_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise NotebookSessionError(
            "Notebook path must be a repository-relative .ipynb path"
        )
    try:
        path_bytes = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise NotebookSessionError("Notebook path must be valid UTF-8") from error
    if len(path_bytes) > 512:
        raise NotebookSessionError("Notebook path must be at most 512 UTF-8 bytes")
    if (
        value.startswith("/")
        or "\\" in value
        or ":" in value
        or any(unicodedata.category(character) == "Cc" for character in value)
        or not value.endswith(".ipynb")
    ):
        raise NotebookSessionError("Notebook path must be a safe relative .ipynb path")
    components = value.split("/")
    if any(component in {"", ".", "..", ".gen"} for component in components):
        raise NotebookSessionError("Notebook path contains a reserved path component")
    return value


def _validate_notebook_cell(cell: Any) -> None:
    if not isinstance(cell, dict) or cell.get("cell_type") not in {
        "code",
        "markdown",
        "raw",
    }:
        raise NotebookSessionError("Notebook contains an invalid cell")
    if not isinstance(cell.get("metadata"), dict):
        raise NotebookSessionError("Notebook cell metadata must be an object")
    source = cell.get("source")
    if not isinstance(source, str) and not (
        isinstance(source, list) and all(isinstance(line, str) for line in source)
    ):
        raise NotebookSessionError(
            "Notebook cell source must be a string or string list"
        )
    if cell["cell_type"] == "code":
        execution_count = cell.get("execution_count")
        if execution_count is not None and type(execution_count) is not int:
            raise NotebookSessionError(
                "Notebook execution_count must be an integer or null"
            )
        if "outputs" in cell and not isinstance(cell["outputs"], list):
            raise NotebookSessionError("Notebook code cell outputs must be an array")


def _notebook_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        try:
            contents = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError) as error:
            raise NotebookSessionError(
                "Notebook data cannot be encoded as JSON"
            ) from error
    elif isinstance(value, (bytes, bytearray)):
        contents = bytes(value)
    else:
        raise TypeError("notebook must be a dictionary or UTF-8 JSON bytes")
    if len(contents) > _MAX_NOTEBOOK_BYTES:
        raise NotebookSessionError("Notebook file exceeds the 8 MiB limit")
    try:
        document = json.loads(
            contents.decode("utf-8"),
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise NotebookSessionError("Notebook file must contain UTF-8 JSON") from error
    if (
        not isinstance(document, dict)
        or type(document.get("nbformat")) is not int
        or document.get("nbformat") != 4
        or type(document.get("nbformat_minor")) is not int
        or document["nbformat_minor"] < 0
        or not isinstance(document.get("metadata"), dict)
        or not isinstance(document.get("cells"), list)
    ):
        raise NotebookSessionError("Notebook file must be valid nbformat 4 JSON")
    for cell in document["cells"]:
        _validate_notebook_cell(cell)
    return contents


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"Nonstandard JSON constant {value}")


def _same_origin(left: str, right: str) -> bool:
    left_parts = urlsplit(left)
    right_parts = urlsplit(right)
    try:
        left_origin = (left_parts.scheme.lower(), left_parts.hostname, left_parts.port)
        right_origin = (
            right_parts.scheme.lower(),
            right_parts.hostname,
            right_parts.port,
        )
    except ValueError:
        return False
    return left_origin == right_origin


def _browser_origin() -> str | None:
    try:
        import js

        return str(js.location.origin)
    except (AttributeError, ImportError):
        return None


def _absolute_snapshot_url(snapshot_url: str) -> str:
    parts = urlsplit(snapshot_url)
    if parts.scheme:
        if (
            parts.scheme not in {"http", "https"}
            or not parts.netloc
            or parts.username is not None
            or parts.password is not None
            or parts.fragment
        ):
            raise NotebookSessionError("The snapshot URL must use HTTP or HTTPS")
        current_origin = _browser_origin()
        if current_origin is not None and not _same_origin(
            snapshot_url, current_origin
        ):
            raise NotebookSessionError(
                "The snapshot URL must use the notebook page origin"
            )
        return snapshot_url

    if parts.fragment:
        raise NotebookSessionError("The snapshot URL must not contain a fragment")
    origin = _browser_origin()
    if origin is None:
        raise NotebookSessionError(
            "A relative snapshot URL can only be opened in the browser"
        )

    if not origin.startswith(("http://", "https://")):
        raise NotebookSessionError("The notebook page has no valid HTTP origin")
    absolute_url = urljoin(origin, snapshot_url)
    if not _same_origin(absolute_url, origin):
        raise NotebookSessionError("The snapshot URL must use the notebook page origin")
    return absolute_url


def _assets_url(snapshot_url: str, branch: str, base_commit: str) -> str:
    parts = urlsplit(snapshot_url)
    parent, separator, leaf = parts.path.rpartition("/")
    if not separator or leaf != "snapshot":
        raise NotebookSessionError("The snapshot URL must end in /notebook/snapshot")
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            f"{parent}/assets",
            urlencode({"branch": branch, "commit": base_commit}),
            "",
        )
    )


def _commit_url(snapshot_url: str) -> str:
    parts = urlsplit(snapshot_url)
    parent, separator, leaf = parts.path.rpartition("/")
    if not separator or leaf != "snapshot":
        raise NotebookSessionError("The snapshot URL must end in /notebook/snapshot")
    return urlunsplit((parts.scheme, parts.netloc, f"{parent}/commit", "", ""))


def _manifest_asset_url(
    raw_url: str,
    manifest_url: str,
    asset_id: str,
    branch: str,
    base_commit: str,
) -> str:
    absolute_url = urljoin(manifest_url, raw_url)
    parts = urlsplit(absolute_url)
    origin = urlsplit(manifest_url)
    asset_root = origin.path.rpartition("/")[0] + "/assets"
    expected_path = f"{asset_root}/{quote(asset_id, safe='')}"
    query = parse_qs(parts.query, keep_blank_values=True)
    if (
        not _same_origin(absolute_url, manifest_url)
        or parts.path != expected_path
        or parts.fragment
        or query.get("branch") != [branch]
        or query.get("commit") != [base_commit]
    ):
        raise NotebookSessionError("The asset manifest contains an invalid asset URL")
    return absolute_url


def _header(headers: Any, name: str) -> str | None:
    if hasattr(headers, "get"):
        value = headers.get(name)
        if value is None:
            value = headers.get(name.lower())
        return None if value is None else str(value)
    return None


async def _response_error(response: Any) -> str:
    status = getattr(response, "status", "unknown")
    message = f"HTTP {status}"
    text_method = getattr(response, "text", None)
    if callable(text_method):
        try:
            body = await text_method()
        except Exception:
            body = ""
        if body:
            message = f"{message}: {body[:500]}"
    return message


async def _require_success(response: Any, label: str) -> None:
    status = getattr(response, "status", 200)
    if not getattr(response, "ok", status < 400):
        raise NotebookSessionError(f"{label} failed: {await _response_error(response)}")


def _response_url(response: Any, requested_url: str) -> None:
    final_url = getattr(response, "url", "")
    if final_url and not _same_origin(str(final_url), requested_url):
        raise NotebookSessionError(
            "The server redirected a notebook request off origin"
        )


def _safe_asset_filename(value: Any) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise NotebookSessionError("The asset manifest contains an invalid filename")
    if "/" in value or "\\" in value or Path(value).name != value:
        raise NotebookSessionError("The asset manifest contains an invalid filename")
    return value


class NotebookSession:
    """An isolated Gen repository snapshot with explicit branch publication.

    Open a session from a GenHub snapshot URL with ``await NotebookSession.open(url)``.
    Sequence edits are committed inside the browser's temporary workspace. Call
    ``await session.commit(target_branch, create_branch=True)`` to publish those
    edits through the authenticated GenHub notebook endpoint.
    """

    def __init__(
        self,
        repository: Any,
        snapshot_url: str,
        source_branch: str,
        base_commit: str,
        temporary_directory: tempfile.TemporaryDirectory[str],
        fetcher: Fetch,
        notebook_path: str | None = None,
    ) -> None:
        self.repository = repository
        self.snapshot_url = snapshot_url
        self.source_branch = source_branch
        self.base_commit = base_commit
        self._temporary_directory = temporary_directory
        self._fetcher = fetcher
        self.notebook_path = notebook_path

    def close(self) -> None:
        """Remove this session's temporary repository workspace."""
        self._temporary_directory.cleanup()

    @classmethod
    async def open(
        cls,
        snapshot_url: str,
        *,
        notebook_path: str | None = None,
    ) -> NotebookSession:
        """Download and validate the snapshot and its assets, then open Gen."""
        if notebook_path is not None:
            notebook_path = _validate_notebook_path(notebook_path)
        return await cls._open(snapshot_url, _pyfetch, notebook_path=notebook_path)

    @classmethod
    async def _open(
        cls,
        snapshot_url: str,
        fetcher: Fetch,
        *,
        notebook_path: str | None = None,
    ) -> NotebookSession:
        absolute_url = _absolute_snapshot_url(snapshot_url)
        try:
            snapshot_response = await fetcher(
                absolute_url,
                credentials="same-origin",
            )
        except Exception as error:
            raise NotebookSessionError(
                f"Could not download repository snapshot: {error}"
            ) from error
        await _require_success(snapshot_response, "Repository snapshot download")
        _response_url(snapshot_response, absolute_url)

        headers = getattr(snapshot_response, "headers", {})
        source_branch = _header(headers, "X-Gen-Notebook-Branch")
        base_commit = _header(headers, "X-Gen-Notebook-Base-Commit")
        if not source_branch or not base_commit:
            raise NotebookSessionError(
                "The repository snapshot is missing its branch metadata"
            )
        if re.fullmatch(r"[0-9a-fA-F]{40}", base_commit) is None:
            raise NotebookSessionError(
                "The repository snapshot has an invalid base commit"
            )

        manifest_url = _assets_url(absolute_url, source_branch, base_commit)
        try:
            manifest_response = await fetcher(
                manifest_url,
                credentials="same-origin",
            )
        except Exception as error:
            raise NotebookSessionError(
                f"Could not download repository assets: {error}"
            ) from error
        await _require_success(manifest_response, "Repository asset manifest download")
        _response_url(manifest_response, manifest_url)
        manifest = await manifest_response.json()
        if not isinstance(manifest, dict) or not isinstance(
            manifest.get("assets"), list
        ):
            raise NotebookSessionError("The repository asset manifest is malformed")

        temporary_directory = tempfile.TemporaryDirectory(prefix="gen-notebook-")
        try:
            workspace = Path(temporary_directory.name)
            gen_directory = workspace / ".gen"
            asset_directory = gen_directory / "assets"
            asset_directory.mkdir(parents=True)
            database = await snapshot_response.bytes()
            if not database:
                raise NotebookSessionError("The repository snapshot is empty")
            (gen_directory / "default.db").write_bytes(database)

            filenames: dict[str, str] = {}
            for asset in manifest["assets"]:
                if not isinstance(asset, dict):
                    raise NotebookSessionError(
                        "The repository asset manifest is malformed"
                    )
                asset_id = asset.get("id")
                checksum = asset.get("checksum")
                if (
                    not isinstance(asset_id, str)
                    or re.fullmatch(r"[0-9a-fA-F]{32}", asset_id) is None
                ):
                    raise NotebookSessionError(
                        "The asset manifest contains an invalid asset ID"
                    )
                if (
                    not isinstance(checksum, str)
                    or re.fullmatch(r"[0-9a-fA-F]{64}", checksum) is None
                ):
                    raise NotebookSessionError(
                        "The asset manifest contains an invalid checksum"
                    )
                filename = _safe_asset_filename(asset.get("filename"))
                previous_checksum = filenames.get(filename)
                if (
                    previous_checksum is not None
                    and previous_checksum.lower() != checksum.lower()
                ):
                    raise NotebookSessionError(
                        "The asset manifest reuses a filename for different bytes"
                    )
                filenames[filename] = checksum
                asset_url = _manifest_asset_url(
                    str(asset.get("url", "")),
                    manifest_url,
                    asset_id,
                    source_branch,
                    base_commit,
                )
                try:
                    asset_response = await fetcher(
                        asset_url,
                        credentials="same-origin",
                    )
                except Exception as error:
                    raise NotebookSessionError(
                        f"Could not download repository asset {filename}: {error}"
                    ) from error
                await _require_success(
                    asset_response, f"Repository asset {filename} download"
                )
                _response_url(asset_response, asset_url)
                contents = await asset_response.bytes()
                actual_checksum = hashlib.sha256(contents).hexdigest()
                if actual_checksum.lower() != checksum.lower():
                    raise NotebookSessionError(
                        f"Checksum verification failed for asset {filename}"
                    )
                destination = asset_directory / filename
                if destination.exists():
                    if destination.read_bytes() != contents:
                        raise NotebookSessionError(
                            "The asset manifest contains conflicting duplicate "
                            "filenames"
                        )
                else:
                    destination.write_bytes(contents)

            from . import Repository

            repository = Repository(str(workspace))
            repository.checkout(source_branch)
            current_branch = repository.current_branch
            if current_branch is None or current_branch.name != source_branch:
                raise NotebookSessionError(
                    "Gen could not check out the requested source branch"
                )
            if str(current_branch.head).lower() != base_commit.lower():
                raise NotebookSessionError(
                    "The repository snapshot does not match its base commit"
                )
            return cls(
                repository,
                absolute_url,
                source_branch,
                base_commit,
                temporary_directory,
                fetcher,
                notebook_path,
            )
        except Exception:
            temporary_directory.cleanup()
            raise

    async def commit(
        self,
        target_branch: str,
        create_branch: bool = False,
        *,
        notebook: dict[str, Any] | bytes | bytearray | None = None,
        notebook_path: str | None = None,
    ) -> dict[str, str]:
        """Publish this edited snapshot to one existing or newly created branch."""
        if not isinstance(target_branch, str) or not target_branch.strip():
            raise ValueError("target_branch must be a non-empty string")
        if not isinstance(create_branch, bool):
            raise TypeError("create_branch must be a bool")

        if (notebook is None) != (notebook_path is None):
            raise ValueError("notebook and notebook_path must be provided together")
        notebook_contents = None
        if notebook is not None:
            notebook_path = _validate_notebook_path(notebook_path)
            notebook_contents = _notebook_bytes(notebook)
            workspace = Path(self._temporary_directory.name).resolve()
            destination = workspace.joinpath(*notebook_path.split("/"))
            if not destination.resolve().is_relative_to(workspace):
                raise NotebookSessionError(
                    "Notebook path escapes the repository workspace"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(notebook_contents)
            try:
                self.repository.add_file(destination, allow_no_changes=True)
            except Exception as error:
                raise NotebookSessionError(
                    f"Could not add notebook {notebook_path} to the local repository: {error}"
                ) from error

        database = self.repository.snapshot_bytes()
        if not isinstance(database, bytes):
            database = bytes(database)
        fields = {
            "source_branch": self.source_branch,
            "base_commit": self.base_commit,
            "target_branch": target_branch,
            "create_branch": "true" if create_branch else "false",
        }
        if notebook_contents is not None and notebook_path is not None:
            fields["notebook_path"] = notebook_path
        body = _MultipartForm(
            database,
            fields,
            notebook_contents,
            PurePosixPath(notebook_path).name
            if notebook_contents is not None and notebook_path is not None
            else None,
        )
        request_body: Any = (
            body.to_browser_form() if sys.platform == "emscripten" else body
        )
        commit_url = _commit_url(self.snapshot_url)
        try:
            if sys.platform == "emscripten":
                response = await _browser_multipart_fetch(commit_url, request_body)
            else:
                response = await self._fetcher(
                    commit_url,
                    method="POST",
                    credentials="same-origin",
                    body=request_body,
                )
        except Exception as error:
            raise NotebookSessionError(
                f"Could not publish notebook changes: {error}"
            ) from error
        await _require_success(response, "Notebook publication")
        _response_url(response, commit_url)
        if sys.platform == "emscripten":
            result = await _browser_response_json(response)
        else:
            result = await response.json()
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("branch"), str)
            or result.get("branch") != target_branch
            or not isinstance(result.get("commit_hash"), str)
            or re.fullmatch(r"[0-9a-fA-F]{40}", result.get("commit_hash", "")) is None
        ):
            raise NotebookSessionError("The notebook publication response is malformed")
        if result["branch"] == self.source_branch:
            self.base_commit = result["commit_hash"]
        if notebook_contents is not None and notebook_path is not None:
            self.notebook_path = notebook_path
        _notify_parent(result["branch"], result["commit_hash"])
        return {"branch": result["branch"], "commit_hash": result["commit_hash"]}

    def commit_widget(self) -> Any:
        """Return controls for explicitly publishing graph edits and a notebook."""
        import ipywidgets as widgets

        branch_names = [branch.name for branch in self.repository.get_branches()]
        branch_options: list[tuple[str, str]] = [(name, name) for name in branch_names]
        if not branch_options:
            branch_options = [("No other branches", "")]
        default_branch = (
            self.source_branch
            if self.source_branch in branch_names
            else branch_options[0][1]
        )
        target = widgets.Dropdown(
            options=branch_options,
            value=default_branch,
            description="Target branch",
            disabled=not branch_names,
        )
        create = widgets.Checkbox(value=False, description="Create a new branch")
        new_branch = widgets.Text(value="notebook-edits", description="New branch name")
        new_branch.layout.display = "none"
        save_notebook = widgets.Checkbox(value=False, description="Save notebook")
        notebook_path = widgets.Text(
            value=self.notebook_path or _DEFAULT_NOTEBOOK_PATH,
            description="Notebook path",
        )
        notebook_path.layout.display = "none"
        message = widgets.HTML(
            value="<span>Notebook edits stay in this browser until you publish.</span>"
        )
        publish = widgets.Button(
            description="Commit",
            button_style="primary",
            disabled=not branch_names,
        )

        def update_target_visibility(change: dict[str, Any]) -> None:
            new_branch.layout.display = "" if change["new"] else "none"
            publish.disabled = not branch_names and not bool(change["new"])

        create.observe(update_target_visibility, names="value")

        def update_notebook_path_visibility(change: dict[str, Any]) -> None:
            notebook_path.layout.display = "" if change["new"] else "none"

        save_notebook.observe(update_notebook_path_visibility, names="value")

        capture = _new_notebook_capture_widget()

        def on_publish_click(button: Any) -> None:
            button.disabled = True
            message.value = "<span>Preparing notebook publication…</span>"

            async def publish_changes() -> None:
                try:
                    branch = new_branch.value.strip() if create.value else target.value
                    if not branch:
                        message.value = (
                            "<span>Enter a branch name before publishing.</span>"
                        )
                        return
                    notebook_options: dict[str, Any] = {}
                    if save_notebook.value:
                        validated_path = _validate_notebook_path(notebook_path.value)
                        message.value = "<span>Capturing the active notebook…</span>"
                        notebook_document = await _capture_active_notebook(capture)
                        notebook_options = {
                            "notebook": notebook_document,
                            "notebook_path": validated_path,
                        }
                    message.value = "<span>Publishing notebook changes…</span>"
                    result = await self.commit(
                        branch,
                        create_branch=create.value,
                        **notebook_options,
                    )
                except Exception as error:
                    message.value = (
                        f"<span>Publication failed: {html.escape(str(error))}</span>"
                    )
                    return
                else:
                    saved_notebook = " and notebook" if save_notebook.value else ""
                    message.value = (
                        "<span>Published graph edits"
                        f"{saved_notebook} to "
                        f"<code>{html.escape(result['branch'])}</code> at "
                        f"<code>{html.escape(result['commit_hash'])}</code>.</span>"
                    )
                    if create.value and result["branch"] not in branch_names:
                        branch_names.append(result["branch"])
                        target.options = [(name, name) for name in branch_names]
                        target.value = result["branch"]
                        create.value = False
                finally:
                    button.disabled = not branch_names and not create.value

            asyncio.create_task(publish_changes())

        publish.on_click(on_publish_click)
        controls = widgets.VBox(
            [
                target,
                create,
                new_branch,
                save_notebook,
                notebook_path,
                publish,
                message,
                capture,
            ]
        )
        return controls


async def _pyfetch(url: str, **kwargs: Any) -> Any:
    try:
        from pyodide.http import pyfetch
    except ImportError as error:
        raise NotebookSessionError(
            "NotebookSession.open() requires the Pyodide browser runtime"
        ) from error
    return await pyfetch(url, **kwargs)


async def _browser_multipart_fetch(url: str, body: Any) -> Any:
    from js import Object, fetch

    options = Object.new()
    options.method = "POST"
    options.credentials = "same-origin"
    options.body = body
    return await fetch(url, options)


async def _browser_response_json(response: Any) -> Any:
    payload = await response.json()
    to_python = getattr(payload, "to_py", None)
    if callable(to_python):
        payload = to_python()
    return payload


def _notify_parent(branch: str, commit_hash: str) -> None:
    try:
        import anywidget
        import traitlets
        from IPython.display import display
    except ImportError:
        return

    class CommitNotification(anywidget.AnyWidget):
        _esm = Path(__file__).parent / "static" / "notebook_commit.js"

        branch = traitlets.Unicode("").tag(sync=True)
        commit_hash = traitlets.Unicode("").tag(sync=True)

    display(CommitNotification(branch=branch, commit_hash=commit_hash))


def _new_notebook_capture_widget() -> Any:
    import anywidget
    import traitlets

    class NotebookCaptureWidget(anywidget.AnyWidget):
        _esm = Path(__file__).parent / "static" / "notebook_commit.js"

        capture_request_id = traitlets.Unicode("").tag(sync=True)
        capture_response_id = traitlets.Unicode("").tag(sync=True)
        capture_notebook_json = traitlets.Unicode("").tag(sync=True)
        capture_error = traitlets.Unicode("").tag(sync=True)

    return NotebookCaptureWidget()


async def _capture_active_notebook(widget: Any) -> dict[str, Any]:
    request_id = secrets.token_urlsafe(24)
    response = asyncio.get_running_loop().create_future()

    def receive_capture(change: dict[str, Any]) -> None:
        if change["new"] != request_id or response.done():
            return
        capture_error = widget.capture_error
        if capture_error:
            response.set_exception(
                NotebookSessionError(
                    f"Could not capture the active notebook: {capture_error}"
                )
            )
        else:
            response.set_result(widget.capture_notebook_json)

    widget.capture_notebook_json = ""
    widget.capture_error = ""
    widget.capture_response_id = ""
    widget.observe(receive_capture, names="capture_response_id")
    try:
        widget.capture_request_id = request_id
        serialized = await asyncio.wait_for(response, _NOTEBOOK_CAPTURE_TIMEOUT)
        if not serialized:
            raise NotebookSessionError("The active notebook capture was empty")
        try:
            document = json.loads(serialized)
        except json.JSONDecodeError as error:
            raise NotebookSessionError(
                "The active notebook capture was not valid JSON"
            ) from error
        if not isinstance(document, dict):
            raise NotebookSessionError(
                "The active notebook capture was not a notebook object"
            )
        _notebook_bytes(document)
        return document
    except asyncio.TimeoutError as error:
        raise NotebookSessionError(
            "Timed out waiting for the active notebook capture"
        ) from error
    finally:
        widget.unobserve(receive_capture, names="capture_response_id")
        widget.capture_request_id = ""
        widget.capture_response_id = ""
        widget.capture_notebook_json = ""
        widget.capture_error = ""


__all__ = ["NotebookSession", "NotebookSessionError"]
