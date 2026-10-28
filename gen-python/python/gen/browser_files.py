"""Lazy, session-scoped access to GenHub repository asset URLs."""

from __future__ import annotations

import zlib
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs, unquote, urlencode, urlsplit


class BrowserFileError(RuntimeError):
    """A sanitized failure while reading a signed repository asset."""


class SignedAssetUrl(str):
    """String-compatible expiring URL that redacts its bearer query from reprs."""

    def __new__(cls, value: str, path: str, expires_at: datetime) -> SignedAssetUrl:
        instance = super().__new__(cls, value)
        instance.path = path
        instance.expires_at = expires_at
        return instance

    def __repr__(self) -> str:
        return f"SignedAssetUrl(path={self.path!r}, expires_at={self.expires_at.isoformat()!r})"


async def _fetch_browser(
    url: str,
    *,
    credentials: str,
    mode: str | None = None,
) -> Any:
    """Use Fetch with explicit credential and redirect boundaries in Pyodide."""
    try:
        from js import Object, fetch
        from pyodide.ffi import to_js
    except ImportError:
        raise BrowserFileError("Browser file access requires the Fetch API") from None

    options: dict[str, Any] = {
        "method": "GET",
        "credentials": credentials,
        "redirect": "error",
        "cache": "no-store",
    }
    if mode is not None:
        options["mode"] = mode
    try:
        return await fetch(url, to_js(options, dict_converter=Object.fromEntries))
    except Exception:
        # Fetch exceptions can contain the complete signed URL, so never expose them.
        raise BrowserFileError("Repository file request failed") from None


def _repository_files_url(repository_url: str, branch: str, snapshot_hash: str) -> str:
    query = urlencode({"branch": branch, "snapshot_hash": snapshot_hash})
    return f"{repository_url.rstrip('/')}/files?{query}"


async def fetch_repository_files(
    repository_url: str,
    branch: str,
    snapshot_hash: str,
) -> list[dict[str, Any]]:
    """Fetch metadata and expiring URLs without requesting any asset body."""
    response = await _fetch_browser(
        _repository_files_url(repository_url, branch, snapshot_hash),
        credentials="same-origin",
    )
    if not response.ok:
        raise BrowserFileError(
            f"Could not list repository files (HTTP {response.status})"
        )
    try:
        contents = await response.json()
        if hasattr(contents, "to_py"):
            contents = contents.to_py()
    except Exception:
        raise BrowserFileError("Repository file metadata was not valid JSON") from None
    if not isinstance(contents, list) or not all(
        isinstance(file, dict) for file in contents
    ):
        raise BrowserFileError("Repository file metadata had an invalid shape")
    return contents


async def fetch_asset_response(url: str) -> Any:
    """Fetch a signed GCS asset without cookies or redirect forwarding."""
    return await _fetch_browser(url, credentials="omit", mode="cors")


def _repository_identity(repository_url: str) -> tuple[str, str]:
    parts = urlsplit(repository_url)
    route = [unquote(part) for part in parts.path.strip("/").split("/")]
    if len(route) == 4 and route[:2] == ["api", "repos"]:
        return route[2], route[3]
    if len(route) == 3 and route[0] == "repos":
        return route[1], route[2]
    raise BrowserFileError("The repository URL is invalid")


def signed_url_expiration(url: Any, repository_url: str, logical_path: str) -> datetime:
    """Accept only short-lived GCS URLs for this repository's archived asset path."""
    if not isinstance(url, str):
        raise BrowserFileError("Repository file does not have a signed storage URL")
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if (
            parts.scheme.lower() != "https"
            or parts.username is not None
            or parts.password is not None
            or host != "storage.googleapis.com"
        ):
            raise ValueError

        namespace, slug = _repository_identity(repository_url)
        url_components = [unquote(component) for component in parts.path.split("/")]
        expected_prefix = ["repos", namespace, slug, ".gen", "assets"]
        if not any(
            url_components[index : index + len(expected_prefix)] == expected_prefix
            for index in range(max(0, len(url_components) - len(expected_prefix) + 1))
        ):
            raise ValueError
        if any(
            component in {".", ".."} or "/" in component or "\\" in component
            for component in url_components
        ):
            raise ValueError

        query = {
            key.casefold(): values for key, values in parse_qs(parts.query).items()
        }
        signature = query.get("x-goog-signature", [""])[0]
        signed_date = query.get("x-goog-date", [""])[0]
        expires_in = int(query.get("x-goog-expires", ["0"])[0])
        if not signature or not signed_date or not 0 < expires_in <= 3600:
            raise ValueError
        signed_at = datetime.strptime(signed_date, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
        expires_at = signed_at + timedelta(seconds=expires_in)
        now = datetime.now(timezone.utc)
        if signed_at > now + timedelta(seconds=60):
            raise ValueError
    except (TypeError, ValueError, IndexError):
        raise BrowserFileError(
            f"Repository file {logical_path!r} does not have a safe, expiring storage URL"
        ) from None
    return expires_at


def _bytes_from_javascript(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if hasattr(value, "to_py"):
        value = value.to_py()
    try:
        return bytes(value)
    except (TypeError, ValueError):
        raise BrowserFileError(
            "Repository file stream returned invalid bytes"
        ) from None


class RepositoryAssetFile:
    """Metadata and lazy access to one tracked logical repository file."""

    def __init__(
        self,
        session: Any,
        *,
        path: str,
        name: str,
        size: int | None,
        last_modified: str,
        file_type: str,
        mimetype: str | None,
        archive_needs_decode: bool,
        asset: Any,
        snapshot_hash: str,
        published_snapshot_hash: str | None = None,
        url: str | None = None,
        url_expires_at: datetime | None = None,
    ) -> None:
        self._session = session
        self._set_metadata(
            name=name,
            size=size,
            last_modified=last_modified,
            file_type=file_type,
            mimetype=mimetype,
            archive_needs_decode=archive_needs_decode,
            url=url,
            url_expires_at=url_expires_at,
        )
        self.path = path
        self._asset = asset
        self._snapshot_hash = snapshot_hash
        self._published_snapshot_hash = published_snapshot_hash
        try:
            asset_id = asset.id
            if asset_id is None:
                raise TypeError
            self._asset_id = str(asset_id)
        except (AttributeError, TypeError) as error:
            raise BrowserFileError(
                "Gen asset metadata is missing its content identity"
            ) from error
        if not self._asset_id:
            raise BrowserFileError("Gen asset content identity is invalid")

    def _set_metadata(
        self,
        *,
        name: str,
        size: int | None,
        last_modified: str,
        file_type: str,
        mimetype: str | None,
        archive_needs_decode: bool,
        url: str | None,
        url_expires_at: datetime | None,
    ) -> None:
        self.name = name
        self.size = size
        self.last_modified = last_modified
        self.file_type = file_type
        self.mimetype = mimetype
        self._archive_needs_decode = archive_needs_decode
        self._url = url
        self._url_expires_at = url_expires_at

    def __repr__(self) -> str:
        return f"RepositoryAssetFile(path={self.path!r}, size={self.size!r})"

    async def url(self) -> SignedAssetUrl:
        """Return a current string-compatible URL whose repr omits its bearer query."""
        if self._session._closed:
            raise BrowserFileError("This notebook repository session is closed")
        await self._session._ensure_file_url(self)
        if self._url is None or self._url_expires_at is None:
            raise BrowserFileError(
                "File has not been published to repository storage yet"
            )
        return SignedAssetUrl(self._url, self.path, self._url_expires_at)

    def open(self) -> RepositoryAssetStream:
        """Open a bounded-memory async stream of the logical file bytes."""
        return RepositoryAssetStream(self)

    async def read(self, limit: int) -> bytes:
        """Read at most ``limit`` logical bytes, rejecting a larger file."""
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        chunks = bytearray()
        async with self.open() as stream:
            while len(chunks) <= limit:
                remaining = limit + 1 - len(chunks)
                if remaining == 0:
                    break
                chunk = await stream.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.extend(chunk)
        if len(chunks) > limit:
            raise BrowserFileError(
                f"Repository file exceeds the {limit}-byte browser read limit"
            )
        return bytes(chunks)


class RepositoryAssetStream:
    """Incremental response reader with optional BGZF-to-workspace decoding."""

    def __init__(self, file: RepositoryAssetFile) -> None:
        self._file = file
        self._reader: Any | None = None
        self._local_stream: Any | None = None
        self._buffer = bytearray()
        self._pending = b""
        self._closed = False
        self._at_eof = False
        self._network_eof = False
        self._decompressor = zlib.decompressobj(31)
        self._decoder_has_input = False
        self._decoder_saw_member = False

    async def __aenter__(self) -> RepositoryAssetStream:
        if self._file._session._closed:
            raise BrowserFileError("This notebook repository session is closed")
        self._file._session._ensure_file_current(self._file)
        local_path = self._file._session._local_asset_path(self._file)
        if local_path is not None:
            try:
                self._local_stream = local_path.open("rb")
            except OSError:
                raise BrowserFileError(
                    "Local repository asset could not be opened"
                ) from None
            self._file._session._register_asset_stream(self)
            return self

        response = await self._file._session._open_asset_response(self._file)
        if self._file._session._closed:
            body = getattr(response, "body", None)
            cancel = getattr(body, "cancel", None)
            if cancel is not None:
                try:
                    await cancel()
                except Exception:
                    pass
            raise BrowserFileError("This notebook repository session is closed")
        body = getattr(response, "body", None)
        if body is None:
            raise BrowserFileError("Repository file response has no readable body")
        try:
            self._reader = body.getReader()
        except Exception:
            raise BrowserFileError(
                "Repository file response could not be streamed"
            ) from None
        self._file._session._register_asset_stream(self)
        return self

    async def __aexit__(
        self, _exception_type: Any, _exception: Any, _traceback: Any
    ) -> None:
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        reader = self._reader
        self._reader = None
        local_stream = self._local_stream
        self._local_stream = None
        self._file._session._unregister_asset_stream(self)
        if local_stream is not None:
            try:
                local_stream.close()
            except OSError:
                pass
        if reader is None:
            return
        try:
            cancel = getattr(reader, "cancel", None)
            if cancel is not None and not self._at_eof:
                await cancel()
        except Exception:
            pass
        try:
            reader.releaseLock()
        except Exception:
            pass

    def close_local(self) -> None:
        """Close the local archive handle when the notebook session closes synchronously."""
        self._closed = True
        local_stream = self._local_stream
        self._local_stream = None
        if local_stream is not None:
            try:
                local_stream.close()
            except OSError:
                pass
        reader = self._reader
        self._reader = None
        if reader is not None:
            try:
                reader.cancel()
            except Exception:
                pass
            try:
                reader.releaseLock()
            except Exception:
                pass
        self._file._session._unregister_asset_stream(self)

    async def read(self, size: int) -> bytes:
        if type(size) is not int or size < 0:
            raise ValueError("stream read size must be a non-negative integer")
        if self._file._session._closed:
            await self.close()
            raise BrowserFileError("This notebook repository session is closed")
        if (self._reader is None and self._local_stream is None) or self._closed:
            raise BrowserFileError("Repository file stream is not open")
        if size == 0:
            return b""
        while len(self._buffer) < size and not self._at_eof:
            if self._file._archive_needs_decode:
                await self._read_decoded_chunk(size)
            else:
                await self._read_raw_chunk(size)
            if self._file._session._closed:
                await self.close()
                raise BrowserFileError("This notebook repository session is closed")
        result = bytes(self._buffer[:size])
        del self._buffer[:size]
        return result

    async def _read_transport_chunk(self) -> None:
        if self._at_eof or self._pending:
            return
        if self._local_stream is not None:
            try:
                chunk = self._local_stream.read(64 * 1024)
            except OSError:
                raise BrowserFileError("Local repository asset stream failed") from None
            if chunk:
                self._pending = chunk
            else:
                self._network_eof = True
            return
        try:
            result = await self._reader.read()
        except Exception:
            raise BrowserFileError("Repository file stream failed") from None
        if bool(result.done):
            self._network_eof = True
            return
        self._pending = _bytes_from_javascript(result.value)

    async def _read_raw_chunk(self, limit: int) -> None:
        while not self._pending and not getattr(self, "_network_eof", False):
            await self._read_transport_chunk()
        if not self._pending:
            self._at_eof = True
            return
        remaining = limit - len(self._buffer)
        self._buffer.extend(self._pending[:remaining])
        self._pending = self._pending[remaining:]

    async def _read_decoded_chunk(self, target_size: int) -> None:
        while len(self._buffer) < target_size and not self._at_eof:
            if not self._pending and not getattr(self, "_network_eof", False):
                await self._read_transport_chunk()

            if not self._pending:
                if not getattr(self, "_network_eof", False):
                    continue
                self._at_eof = True
                if self._decoder_has_input and not self._decompressor.eof:
                    raise BrowserFileError("Repository asset archive is truncated")
                if not self._decoder_saw_member:
                    raise BrowserFileError("Repository asset archive is invalid")
                return

            try:
                decoded = self._decompressor.decompress(
                    self._pending, max(1, target_size - len(self._buffer))
                )
            except zlib.error:
                raise BrowserFileError("Repository asset archive is invalid") from None
            self._buffer.extend(decoded)

            if self._decompressor.eof:
                self._decoder_saw_member = True
                self._decoder_has_input = False
                self._pending = self._decompressor.unused_data
                self._decompressor = zlib.decompressobj(31)
                continue

            self._decoder_has_input = True
            self._pending = self._decompressor.unconsumed_tail
            if not self._pending:
                # The current member may continue in the next transport chunk.
                continue
