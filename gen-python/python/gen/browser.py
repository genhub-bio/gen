"""Browser-side sessions for editing and publishing Gen repositories."""

from __future__ import annotations

import asyncio
import base64
import html
import json
import mimetypes
import os
import secrets
import stat
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit


class NotebookSessionError(RuntimeError):
    """Raised for failures opening, browsing, or publishing notebook sessions."""


_MAX_NOTEBOOK_BYTES = 8 * 1024 * 1024
_MAX_CONTENTS_FILE_BYTES = _MAX_NOTEBOOK_BYTES
_NOTEBOOK_CAPTURE_TIMEOUT = 30
_DEFAULT_NOTEBOOK_PATH = "notebooks/Untitled.ipynb"
_CONTENTS_COMM_TARGET = "genhub.repository.contents"
_CONTENTS_RESERVED_COMPONENTS = {".gen", ".git"}
_CONTENTS_DATABASE_NAMES = {"default.db", "gen.db"}


def _contents_path(value: Any) -> str:
    """Validate a canonical repository-relative path used by the read-only browser."""
    if not isinstance(value, str):
        raise NotebookSessionError("Repository paths must be strings")
    if value == "":
        return value
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise NotebookSessionError("Repository paths must be valid UTF-8") from error
    if len(encoded) > 4096:
        raise NotebookSessionError("Repository path is too long")
    if (
        value.startswith("/")
        or "\\" in value
        or ":" in value
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise NotebookSessionError("Repository path must be a safe relative path")
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise NotebookSessionError("Repository path contains a reserved component")
    decoded_path = value
    for _ in range(16):
        next_decoded_path = unquote(decoded_path, errors="replace")
        if next_decoded_path == decoded_path:
            break
        if (
            next_decoded_path.startswith("/")
            or next_decoded_path.count("/") != decoded_path.count("/")
            or "\\" in next_decoded_path
            or ":" in next_decoded_path
            or any(
                unicodedata.category(character) == "Cc"
                for character in next_decoded_path
            )
        ):
            raise NotebookSessionError("Repository path contains an encoded separator")
        decoded_path = next_decoded_path
        decoded_components = decoded_path.split("/")
        if any(component in {"", ".", ".."} for component in decoded_components):
            raise NotebookSessionError("Repository path contains an encoded traversal")
        _validate_contents_components(decoded_components)
    else:
        still_encoded = unquote(decoded_path, errors="replace") != decoded_path
        if still_encoded:
            raise NotebookSessionError("Repository path has too many encoded layers")
    _validate_contents_components(components)
    return value


def _validate_contents_components(components: list[str]) -> None:
    if any(
        component.casefold() in _CONTENTS_RESERVED_COMPONENTS
        for component in components
    ):
        raise NotebookSessionError("Repository path contains an internal component")
    for component in components:
        name = component.casefold()
        if name in _CONTENTS_DATABASE_NAMES or any(
            name.startswith(f"{database_name}{suffix}")
            for database_name in _CONTENTS_DATABASE_NAMES
            for suffix in ("-wal", "-shm", "-journal", ".")
        ):
            raise NotebookSessionError("Repository database files are not available")


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
    try:
        left_parts = urlsplit(left)
        right_parts = urlsplit(right)
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


def _absolute_repository_url(repository_url: str) -> str:
    if not isinstance(repository_url, str) or not repository_url:
        raise NotebookSessionError("A GenHub repository URL is required")

    try:
        parts = urlsplit(repository_url)
    except ValueError as error:
        raise NotebookSessionError("The GenHub repository URL is invalid") from error
    if not _same_origin(repository_url, repository_url):
        raise NotebookSessionError("The GenHub repository URL is invalid")
    if parts.query or parts.fragment or "#" in repository_url:
        raise NotebookSessionError(
            "The GenHub repository URL must not contain a query or fragment"
        )
    if parts.username is not None or parts.password is not None:
        raise NotebookSessionError(
            "The GenHub repository URL must not include credentials"
        )

    if parts.scheme:
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise NotebookSessionError(
                "The GenHub repository URL must use HTTP or HTTPS"
            )
        absolute_url = repository_url
    else:
        if parts.netloc:
            raise NotebookSessionError(
                "The GenHub repository URL must use the notebook page origin"
            )
        origin = _browser_origin()
        if origin is None or not origin.startswith(("http://", "https://")):
            raise NotebookSessionError(
                "A relative repository URL can only be opened in the browser"
            )
        absolute_url = urljoin(f"{origin}/", repository_url)

    current_origin = _browser_origin()
    if current_origin is not None and not _same_origin(absolute_url, current_origin):
        raise NotebookSessionError(
            "The GenHub repository URL must use the notebook page origin"
        )

    absolute_parts = urlsplit(absolute_url)
    path = absolute_parts.path.removesuffix("/")
    route = path.split("/")[1:]
    valid_api_path = (len(route) == 4 and route[:2] == ["api", "repos"]) or (
        len(route) == 3 and route[0] == "repos"
    )
    if (
        not valid_api_path
        or any(part in {"", ".", ".."} for part in route)
        or any("%2f" in part.lower() or "%5c" in part.lower() for part in route)
    ):
        raise NotebookSessionError("The URL must identify one GenHub repository")
    return absolute_url


def _validate_branch(value: Any) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise NotebookSessionError("The branch name is invalid")
    return value


class NotebookSession:
    """An isolated Gen clone with explicit branch publication.

    Open a session with a same-origin GenHub repository URL. Repository changes stay
    in its temporary local workspace until the commit widget pushes a branch.
    """

    def __init__(
        self,
        repository: Any,
        repository_url: str,
        source_branch: str,
        base_commit: str,
        temporary_directory: tempfile.TemporaryDirectory[str],
        notebook_path: str | None = None,
    ) -> None:
        self.repository = repository
        self.repository_url = repository_url
        self.source_branch = source_branch
        self.base_commit = base_commit
        self._temporary_directory = temporary_directory
        self.notebook_path = notebook_path
        self._contents_comm: Any | None = None
        self._closed = False

    def close(self) -> None:
        """Detach the read-only browser and remove this session's workspace."""
        if self._closed:
            return
        self._closed = True
        comm = self._contents_comm
        self._contents_comm = None
        if comm is not None:
            try:
                comm.close()
            except Exception:
                pass
        self._temporary_directory.cleanup()

    def expose_contents(self) -> Any:
        """Open a session-bound read-only Jupyter Contents comm for materialized assets.

        The browser receives only current tracked files materialized in this clone's
        workspace. Requests resolve ``Repository.get_assets()`` on demand, so pulls and
        branch checkouts are reflected without copying archived assets into another
        store.
        """
        if self._closed:
            raise NotebookSessionError("This notebook repository session is closed")
        if self._contents_comm is not None:
            return self._contents_comm
        try:
            from comm import create_comm
        except ImportError as error:
            raise NotebookSessionError(
                "Repository file browsing requires Jupyter comm support"
            ) from error

        comm = create_comm(
            target_name=_CONTENTS_COMM_TARGET,
            data={"type": "ready", "protocol": 1},
        )
        self._contents_comm = comm

        def receive(message: Any) -> None:
            if self._closed or self._contents_comm is not comm:
                return
            content = message.get("content", {}) if isinstance(message, dict) else {}
            request = content.get("data", {}) if isinstance(content, dict) else {}
            self._handle_contents_request(comm, request)

        def detached(_message: Any) -> None:
            if self._contents_comm is comm:
                self._contents_comm = None

        comm.on_msg(receive)
        comm.on_close(detached)
        return comm

    def _handle_contents_request(self, comm: Any, request: Any) -> None:
        if not isinstance(request, dict):
            return
        request_id = request.get("id")
        if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
            return
        try:
            if request.get("method") != "get":
                raise NotebookSessionError("Only read-only get requests are supported")
            if type(request.get("content")) is not bool:
                raise NotebookSessionError(
                    "Contents requests must specify a boolean content flag"
                )
            requested_format = request.get("format")
            if requested_format is not None and (
                not isinstance(requested_format, str)
                or requested_format not in {"text", "base64", "json"}
            ):
                raise NotebookSessionError(
                    "Contents format must be text, base64, or json"
                )
            model = self._contents_model(
                request.get("path"),
                content=request["content"],
                format=requested_format,
            )
            response = {"id": request_id, "model": model}
        except NotebookSessionError as error:
            response = {"id": request_id, "error": {"message": str(error)}}
        except Exception:
            response = {
                "id": request_id,
                "error": {"message": "Could not read repository contents"},
            }
        if not self._closed and self._contents_comm is comm:
            try:
                comm.send(response)
            except Exception:
                if self._contents_comm is comm:
                    self._contents_comm = None

    def _contents_workspace(self) -> Path:
        return Path(self._temporary_directory.name).resolve()

    def _contents_database_paths(self, workspace: Path) -> set[Path]:
        database_paths = {
            workspace / "default.db",
            workspace / ".gen" / "default.db",
            Path(self.repository.gen_dir) / "default.db",
        }
        return {
            sibling
            for database_path in database_paths
            for sibling in (
                database_path,
                database_path.with_name(f"{database_path.name}-wal"),
                database_path.with_name(f"{database_path.name}-shm"),
                database_path.with_name(f"{database_path.name}-journal"),
            )
        }

    def _materialized_contents_files(self) -> dict[str, Path]:
        workspace = self._contents_workspace()
        database_paths = self._contents_database_paths(workspace)
        files: dict[str, Path] = {}
        for asset in self.repository.get_assets():
            try:
                relative_path = _contents_path(asset.logical_path)
            except NotebookSessionError:
                continue
            if not relative_path:
                continue
            candidate = workspace
            components = relative_path.split("/")
            try:
                for component in components:
                    candidate = candidate / component
                    file_status = candidate.lstat()
                    if stat.S_ISLNK(file_status.st_mode):
                        raise NotebookSessionError(
                            "Symbolic links are not available in repository browsing"
                        )
                if not stat.S_ISREG(file_status.st_mode):
                    continue
                resolved = candidate.resolve(strict=True)
                if not resolved.is_relative_to(workspace):
                    continue
                if any(
                    os.path.samefile(resolved, database_path)
                    for database_path in database_paths
                    if database_path.exists()
                ):
                    continue
            except (
                FileNotFoundError,
                NotADirectoryError,
                OSError,
                NotebookSessionError,
            ):
                continue
            files[relative_path] = resolved
        return files

    def _contents_model(
        self, path: Any, *, content: bool, format: str | None = None
    ) -> dict[str, Any]:
        relative_path = _contents_path(path)
        files = self._materialized_contents_files()
        file_path = files.get(relative_path)
        if file_path is not None:
            return self._contents_file_model(
                relative_path, file_path, content=content, requested_format=format
            )

        directories = {""}
        for file_name in files:
            components = file_name.split("/")
            directories.update(
                "/".join(components[:index]) for index in range(1, len(components))
            )
        if relative_path not in directories:
            raise NotebookSessionError(
                "Path is not a tracked materialized file or directory"
            )

        name = relative_path.rsplit("/", 1)[-1] if relative_path else ""
        model: dict[str, Any] = {
            "name": name,
            "path": relative_path,
            "type": "directory",
            "content": None,
            "format": None,
            "mimetype": None,
            "writable": False,
            "size": None,
        }
        modified = self._contents_directory_timestamp(relative_path)
        model["created"] = modified
        model["last_modified"] = modified
        if content:
            prefix = f"{relative_path}/" if relative_path else ""
            children: dict[str, dict[str, Any]] = {}
            for file_name, child_path in files.items():
                if not file_name.startswith(prefix):
                    continue
                remainder = file_name[len(prefix) :]
                child_name, separator, _tail = remainder.partition("/")
                child_relative_path = f"{prefix}{child_name}"
                if separator:
                    children[child_name] = self._contents_directory_model(
                        child_name, child_relative_path
                    )
                else:
                    children[child_name] = self._contents_file_model(
                        child_relative_path,
                        child_path,
                        content=False,
                        requested_format=None,
                    )
            model["content"] = [children[key] for key in sorted(children)]
        return model

    def _contents_directory_timestamp(self, path: str) -> str:
        directory = self._contents_workspace()
        if path:
            for component in path.split("/"):
                directory = directory / component
                directory_status = directory.lstat()
                if not stat.S_ISDIR(directory_status.st_mode):
                    raise NotebookSessionError(
                        "Repository directory changed while listing contents"
                    )
        else:
            directory_status = directory.lstat()
        return datetime.fromtimestamp(
            directory_status.st_mtime, timezone.utc
        ).isoformat()

    def _contents_directory_model(self, name: str, path: str) -> dict[str, Any]:
        modified = self._contents_directory_timestamp(path)
        return {
            "name": name,
            "path": path,
            "type": "directory",
            "content": None,
            "format": None,
            "mimetype": None,
            "writable": False,
            "size": None,
            "created": modified,
            "last_modified": modified,
        }

    @staticmethod
    def _contents_file_model(
        path: str,
        file_path: Path,
        *,
        content: bool,
        requested_format: str | None,
    ) -> dict[str, Any]:
        file_status = file_path.stat()
        modified = datetime.fromtimestamp(
            file_status.st_mtime, timezone.utc
        ).isoformat()
        model: dict[str, Any] = {
            "name": path.rsplit("/", 1)[-1],
            "path": path,
            "type": "notebook" if path.lower().endswith(".ipynb") else "file",
            "content": None,
            "format": None,
            "mimetype": mimetypes.guess_type(path)[0],
            "writable": False,
            "size": file_status.st_size,
            "created": modified,
            "last_modified": modified,
        }
        if not content:
            return model

        if file_status.st_size > _MAX_CONTENTS_FILE_BYTES:
            raise NotebookSessionError(
                "File exceeds the 8 MiB browser read limit. Use Gen in a notebook cell "
                "to query large genomic files without opening them in the editor"
            )
        with file_path.open("rb") as stream:
            file_contents = stream.read(_MAX_CONTENTS_FILE_BYTES + 1)
        if len(file_contents) > _MAX_CONTENTS_FILE_BYTES:
            raise NotebookSessionError(
                "File exceeds the 8 MiB browser read limit. Use Gen in a notebook cell "
                "to query large genomic files without opening them in the editor"
            )
        resolved_format = requested_format or (
            "json" if path.lower().endswith(".ipynb") else None
        )
        if resolved_format == "json":
            try:
                model["content"] = json.loads(file_contents.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise NotebookSessionError(
                    "JSON asset does not contain valid UTF-8 JSON"
                ) from error
            model["format"] = "json"
            if path.lower().endswith(".ipynb"):
                model["mimetype"] = "application/x-ipynb+json"
        elif resolved_format == "base64":
            model["content"] = base64.b64encode(file_contents).decode("ascii")
            model["format"] = "base64"
            model["mimetype"] = model["mimetype"] or "application/octet-stream"
        elif resolved_format == "text":
            try:
                model["content"] = file_contents.decode("utf-8")
                model["format"] = "text"
                model["mimetype"] = model["mimetype"] or "text/plain"
            except UnicodeDecodeError as error:
                raise NotebookSessionError(
                    "File is not valid UTF-8 text; request it as base64"
                ) from error
        else:
            try:
                model["content"] = file_contents.decode("utf-8")
                model["format"] = "text"
                model["mimetype"] = model["mimetype"] or "text/plain"
            except UnicodeDecodeError:
                model["content"] = base64.b64encode(file_contents).decode("ascii")
                model["format"] = "base64"
                model["mimetype"] = model["mimetype"] or "application/octet-stream"
        return model

    def _notify_contents_changed(self) -> None:
        comm = self._contents_comm
        if self._closed or comm is None:
            return
        try:
            comm.send({"type": "changed", "path": ""})
        except Exception:
            if self._contents_comm is comm:
                self._contents_comm = None

    @classmethod
    async def open(
        cls,
        repository_url: str,
        *,
        branch: str | None = None,
        committer: str | None = None,
        email: str | None = None,
        notebook_path: str | None = None,
    ) -> NotebookSession:
        """Clone a GenHub repository and open its selected branch locally."""
        repository_url = _absolute_repository_url(repository_url)
        branch = _validate_branch(branch)
        if notebook_path is not None:
            notebook_path = _validate_notebook_path(notebook_path)
        temporary_directory = tempfile.TemporaryDirectory(prefix="gen-notebook-")
        try:
            from . import clone

            repository = clone(
                repository_url,
                path=temporary_directory.name,
                committer=committer,
                email=email,
            )
            current_branch = repository.current_branch
            if current_branch is None:
                raise NotebookSessionError(
                    "The cloned repository has no checked out branch"
                )
            source_branch = branch or current_branch.name
            if source_branch != current_branch.name:
                repository.fetch("origin", source_branch)
                local_branches = {item.name for item in repository.get_branches()}
                if source_branch not in local_branches:
                    repository.create_branch(source_branch, f"origin/{source_branch}")
                repository.checkout(source_branch)
                repository.set_branch_remote("origin")
                repository.pull()
            current_branch = repository.current_branch
            if current_branch is None or current_branch.name != source_branch:
                raise NotebookSessionError(
                    f"Gen could not check out branch '{source_branch}'"
                )
            base_commit = str(current_branch.head)
            return cls(
                repository,
                repository_url,
                source_branch,
                base_commit,
                temporary_directory,
                notebook_path,
            )
        except Exception as error:
            temporary_directory.cleanup()
            if isinstance(error, NotebookSessionError):
                raise
            raise NotebookSessionError(
                f"Could not clone GenHub repository: {error}"
            ) from error

    def refresh(self) -> str:
        """Pull the selected branch from its configured GenHub remote."""
        if self._closed:
            raise NotebookSessionError("This notebook repository session is closed")
        try:
            self.repository.pull()
        except Exception as error:
            raise NotebookSessionError(
                f"Could not pull branch '{self.source_branch}': {error}"
            ) from error
        current_branch = self.repository.current_branch
        if current_branch is None or current_branch.name != self.source_branch:
            raise NotebookSessionError(
                "The repository checkout changed while refreshing the notebook"
            )
        self.base_commit = str(current_branch.head)
        self._notify_contents_changed()
        return self.base_commit

    async def commit(
        self,
        target_branch: str,
        create_branch: bool = False,
        *,
        notebook: dict[str, Any] | bytes | bytearray | None = None,
        notebook_path: str | None = None,
    ) -> dict[str, str]:
        """Save the notebook locally, then push the current or a new branch."""
        target_branch = _validate_branch(target_branch)
        if target_branch is None:
            raise ValueError("target_branch must be a non-empty string")
        if not isinstance(create_branch, bool):
            raise TypeError("create_branch must be a bool")

        if (notebook is None) != (notebook_path is None):
            raise ValueError("notebook and notebook_path must be provided together")
        notebook_contents = None
        if notebook is not None:
            notebook_path = _validate_notebook_path(notebook_path)
            notebook_contents = _notebook_bytes(notebook)
        current_branch = self.repository.current_branch
        if current_branch is None:
            raise NotebookSessionError("The repository has no checked out branch")
        current_name = current_branch.name
        known_branches = {branch.name for branch in self.repository.get_branches()}

        if create_branch:
            if target_branch in known_branches:
                if target_branch != current_name:
                    raise NotebookSessionError(
                        f"Branch '{target_branch}' already exists; open it before editing, "
                        "or choose a new branch name"
                    )
            else:
                try:
                    self.repository.checkout(target_branch, create=True)
                    self.repository.set_branch_remote("origin")
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    raise NotebookSessionError(
                        f"Could not create branch '{target_branch}': {error}"
                    ) from error
        elif target_branch != current_name:
            target_hint = (
                "It is already present in this repository. "
                if target_branch in known_branches
                else ""
            )
            raise NotebookSessionError(
                f"Cannot publish to '{target_branch}' from the current checkout. "
                f"{target_hint}Open that branch before editing, or create a new branch."
            )

        self.source_branch = target_branch
        if notebook_contents is not None and notebook_path is not None:
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
            self.notebook_path = notebook_path

        current_branch = self.repository.current_branch
        if current_branch is None or current_branch.name != target_branch:
            raise NotebookSessionError(
                "The repository checkout changed before the notebook could be pushed"
            )
        commit_hash = str(current_branch.head)
        try:
            self.repository.push(branch=target_branch, force=False)
        except Exception as error:
            raise NotebookSessionError(
                f"Could not push branch '{target_branch}': {error}"
            ) from error
        self.base_commit = commit_hash
        return {"branch": target_branch, "commit_hash": commit_hash}

    def commit_widget(self) -> Any:
        """Return controls for explicitly publishing graph edits and a notebook."""
        import ipywidgets as widgets

        current_branch = self.repository.current_branch
        branch_names = [current_branch.name] if current_branch is not None else []
        branch_options: list[tuple[str, str]] = [(name, name) for name in branch_names]
        if not branch_options:
            branch_options = [("No other branches", "")]
        default_branch = branch_options[0][1]
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
                except (OSError, RuntimeError, TypeError, ValueError) as error:
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
                    with capture.hold_sync():
                        capture.branch = result["branch"]
                        capture.commit_hash = result["commit_hash"]
                    if create.value:
                        create.value = False
                finally:
                    current_branch = self.repository.current_branch
                    if current_branch is not None:
                        branch_names[:] = [current_branch.name]
                        target.options = [(current_branch.name, current_branch.name)]
                        target.value = current_branch.name
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


def _new_notebook_capture_widget() -> Any:
    import anywidget
    import traitlets

    class NotebookCaptureWidget(anywidget.AnyWidget):
        _esm = Path(__file__).parent / "static" / "notebook_commit.js"

        capture_request_id = traitlets.Unicode("").tag(sync=True)
        capture_response_id = traitlets.Unicode("").tag(sync=True)
        capture_notebook_json = traitlets.Unicode("").tag(sync=True)
        capture_error = traitlets.Unicode("").tag(sync=True)
        branch = traitlets.Unicode("").tag(sync=True)
        commit_hash = traitlets.Unicode("").tag(sync=True)

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
    except TimeoutError as error:
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
