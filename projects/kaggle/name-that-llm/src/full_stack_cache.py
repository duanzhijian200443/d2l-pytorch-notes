"""Small, fail-closed component caches; COMMITTED means bytes, not model coverage.

Only input IDs and exact UTF-8 text SHA-256 fingerprints are persisted. Arrays
are copied before saving, retain their dtype/shape, and never use object pickle.
The caller owns component coverage, prediction semantics, and metadata content.
"""
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np


LABELS = (
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
)
_MANIFEST_KEYS = {"schema_version", "state", "commit_scope", "component", "identity", "identity_sha256",
                  "metadata", "finish_metadata", "required_arrays", "required_notes", "arrays", "notes"}
_ARRAY_KEYS = {"name", "path", "bytes", "sha256", "dtype", "shape", "metadata"}
_NOTE_KEYS = {"name", "path", "bytes", "sha256"}
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


class CacheError(ValueError):
    """The cache cannot be safely written or accepted."""


def _check(condition, message):
    if not condition:
        raise CacheError(message)


def _name(value):
    _check(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is not None
           and not value.endswith(".") and value.split(".", 1)[0].upper() not in _RESERVED,
           "Unsafe component or entry name")
    return value


def _json_value(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        _check(math.isfinite(value), "Nonfinite JSON metadata")
    elif type(value) is list:
        for item in value:
            _json_value(item)
    elif type(value) is dict:
        _check(all(type(key) is str for key in value), "JSON object keys must be strings")
        for item in value.values():
            _json_value(item)
    else:
        raise CacheError("Metadata must contain only JSON values")


def _encode(value):
    try:
        _json_value(value)
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (ValueError, UnicodeError, RecursionError) as error:
        raise CacheError("Metadata cannot be encoded as JSON") from error


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _check(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def _bad_constant(value):
    raise CacheError("Nonfinite JSON value")


def _decode(content):
    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_bad_constant)
        _json_value(value)
        return value
    except (ValueError, UnicodeError, TypeError, RecursionError) as error:
        raise CacheError("Invalid cache JSON") from error


def _snapshot(value):
    return _decode(_encode(value))


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _hash(value):
    _check(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "Invalid SHA-256")


def _sequence(values, name):
    _check(not isinstance(values, (str, bytes, set, frozenset, dict)), "Expected an ordered " + name + " sequence")
    try:
        return list(values)
    except TypeError as error:
        raise CacheError("Expected an ordered " + name + " sequence") from error


def _identity(ids, texts, labels):
    ids, texts, labels = (_sequence(value, name) for value, name in ((ids, "IDs"), (texts, "texts"), (labels, "labels")))
    _check(all(isinstance(value, str) and bool(value.strip()) for value in ids), "IDs must be nonblank strings")
    _check(len(ids) == len(set(ids)) and len(ids) == len(texts), "Duplicate IDs or ID/text length mismatch")
    _check(all(isinstance(value, str) for value in texts), "Texts must be strings")
    _check(labels == list(LABELS), "The fixed 18-label order changed")
    try:
        hashes = [_digest(value.encode("utf-8")) for value in texts]
    except UnicodeError as error:
        raise CacheError("Text cannot be fingerprinted as UTF-8") from error
    return {"ids": [str(value) for value in ids], "text_sha256": hashes, "labels": list(LABELS), "rows": len(ids)}


def _link(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _root(path, create=False):
    path = Path(path).absolute()
    _check(all(not _link(part) for part in (path, *path.parents)), "Symlink/junction cache path is forbidden")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    _check(path.is_dir(), "Cache root is not a directory")
    return path.resolve(strict=True)


def _required(values):
    values = _sequence(values, "required names")
    for value in values:
        _name(value)
    _check(len({value.casefold() for value in values}) == len(values), "Duplicate required name")
    return values


def _inventory(directory, expected):
    _check(directory.is_dir() and not _link(directory), "Unsafe component directory")
    observed = set()
    for path in directory.iterdir():
        _check(path.is_file() and not _link(path), "Unexpected directory or link in cache")
        observed.add(path.name)
    _check(observed == set(expected), "Missing or extra cache file")


def _array_from_bytes(content, receipt):
    _check(content.startswith(b"\x93NUMPY"), "Expected a single NPY array")
    try:
        stream = io.BytesIO(content)
        result = np.load(stream, allow_pickle=False)
        _check(isinstance(result, np.ndarray) and stream.tell() == len(content), "Array has trailing/unsupported content")
    except (ValueError, TypeError, OSError, EOFError) as error:
        raise CacheError("Array cannot be read without pickle") from error
    _check(result.dtype.kind in "biufc" and result.dtype.str == receipt["dtype"]
           and list(result.shape) == receipt["shape"], "Array dtype or shape changed")
    _check(np.isfinite(result).all(), "Array contains nonfinite values")
    return result


def _validate_manifest(manifest, component, identity):
    _check(type(manifest) is dict and set(manifest) == _MANIFEST_KEYS, "Invalid manifest schema")
    _check(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
           and manifest["state"] == "COMMITTED" and manifest["commit_scope"] == "component_bytes_only",
           "Cache is not byte-committed")
    _check(_name(manifest["component"]) == component, "Component name changed")
    _check(_encode(manifest["identity"]) == _encode(identity), "IDs, text fingerprints, or label order changed")
    _hash(manifest["identity_sha256"])
    _check(manifest["identity_sha256"] == _digest(_encode(identity)), "Identity fingerprint changed")
    names, paths = set(), set()
    for kind, keys, suffix in (("arrays", _ARRAY_KEYS, ".npy"), ("notes", _NOTE_KEYS, ".json")):
        _check(type(manifest[kind]) is list, "Invalid manifest entries")
        for entry in manifest[kind]:
            _check(type(entry) is dict and set(entry) == keys, "Invalid entry schema")
            name = _name(entry["name"])
            _check(name.casefold() not in names, "Duplicate cache entry name")
            names.add(name.casefold())
            expected = ("array-" if kind == "arrays" else "note-") + name + suffix
            _check(entry["path"] == expected and expected.casefold() not in paths, "Unsafe/duplicate cache entry path")
            paths.add(expected.casefold())
            _check(type(entry["bytes"]) is int and entry["bytes"] > 0, "Invalid entry byte length")
            _hash(entry["sha256"])
            if kind == "arrays":
                _check(type(entry["dtype"]) is str, "Invalid array dtype")
                try:
                    dtype = np.dtype(entry["dtype"])
                except (TypeError, ValueError) as error:
                    raise CacheError("Invalid array dtype") from error
                _check(dtype.kind in "biufc" and dtype.str == entry["dtype"], "Unsupported/noncanonical array dtype")
                _check(type(entry["shape"]) is list and all(type(dim) is int and dim >= 0 for dim in entry["shape"]),
                       "Invalid array shape")
    for kind in ("arrays", "notes"):
        required = _required(manifest["required_" + kind])
        _check(set(required).issubset({entry["name"] for entry in manifest[kind]}), "Missing required cache entry")


def _verify_files(directory, manifest, include_manifest, keep_arrays):
    entries = manifest["arrays"] + manifest["notes"]
    expected = [entry["path"] for entry in entries] + (["manifest.json"] if include_manifest else [])
    _inventory(directory, expected)
    arrays, notes = {}, {}
    for kind in ("arrays", "notes"):
        for entry in manifest[kind]:
            content = (directory / entry["path"]).read_bytes()
            _check(len(content) == entry["bytes"] and _digest(content) == entry["sha256"], "Cache entry bytes/hash changed")
            if kind == "arrays":
                value = _array_from_bytes(content, entry)
                if keep_arrays:
                    snapshot = value.copy(order="K")
                    snapshot.setflags(write=False)
                    arrays[entry["name"]] = snapshot
            else:
                notes[entry["name"]] = _decode(content)
    _inventory(directory, expected)
    for entry in entries:
        _check(_digest((directory / entry["path"]).read_bytes()) == entry["sha256"], "Cache changed during verification")
    return arrays, notes


@dataclass(frozen=True)
class CacheSnapshot:
    directory: Path
    manifest: dict
    manifest_sha256: str
    arrays: dict
    notes: dict

    @property
    def metadata(self):
        return self.manifest["metadata"]

    @property
    def finish_metadata(self):
        return self.manifest["finish_metadata"]


class CacheWriter:
    enabled = True

    def __init__(self, root, component, ids, texts, labels=LABELS, metadata=None):
        self.component = _name(component)
        self._identity = _identity(ids, texts, labels)
        self._metadata = _snapshot(metadata)
        self._entries = {"arrays": [], "notes": []}
        self._names = set()
        self._closed = False
        self.manifest_sha256 = None
        try:
            self.directory = _root(root, create=True) / self.component
            self.directory.mkdir(exist_ok=False)
        except OSError as error:
            raise CacheError("Component directory must be new and writable") from error

    def _new_name(self, name):
        _check(not self._closed, "Cache writer is closed")
        _root(self.directory)
        _name(name)
        _check(name.casefold() not in self._names, "Duplicate cache entry name")

    def array(self, name, array, metadata=None):
        self._new_name(name)
        details = _snapshot(metadata)
        try:
            snapshot = np.array(array, copy=True, order="K", subok=False)
        except (ValueError, TypeError) as error:
            raise CacheError("Cannot snapshot array") from error
        _check(snapshot.dtype.kind in "biufc" and np.isfinite(snapshot).all(), "Expected a finite numeric/bool nonobject array")
        filename = "array-" + name + ".npy"
        try:
            with (self.directory / filename).open("xb") as stream:
                np.save(stream, snapshot, allow_pickle=False)
            content = (self.directory / filename).read_bytes()
        except (OSError, ValueError) as error:
            self.close()
            raise CacheError("Array write failed; component remains uncommitted") from error
        receipt = {"name": name, "path": filename, "dtype": snapshot.dtype.str, "shape": list(snapshot.shape),
                   "bytes": len(content), "sha256": _digest(content), "metadata": details}
        self._entries["arrays"].append(receipt)
        self._names.add(name.casefold())
        return _snapshot(receipt)

    def note(self, name, metadata):
        self._new_name(name)
        content = _encode(metadata)
        filename = "note-" + name + ".json"
        try:
            with (self.directory / filename).open("xb") as stream:
                stream.write(content)
        except OSError as error:
            self.close()
            raise CacheError("Note write failed; component remains uncommitted") from error
        receipt = {"name": name, "path": filename, "bytes": len(content), "sha256": _digest(content)}
        self._entries["notes"].append(receipt)
        self._names.add(name.casefold())
        return _snapshot(receipt)

    def finish(self, required_arrays, required_notes=(), metadata=None):
        _check(not self._closed, "Cache writer is closed")
        _root(self.directory)
        manifest = {"schema_version": 1, "state": "COMMITTED", "commit_scope": "component_bytes_only",
                    "component": self.component, "identity": self._identity,
                    "identity_sha256": _digest(_encode(self._identity)), "metadata": self._metadata,
                    "finish_metadata": _snapshot(metadata), "required_arrays": _required(required_arrays),
                    "required_notes": _required(required_notes), **self._entries}
        _validate_manifest(manifest, self.component, self._identity)
        try:
            _verify_files(self.directory, manifest, include_manifest=False, keep_arrays=False)
            content = _encode(manifest)
            with (self.directory / "manifest.json").open("xb") as stream:
                stream.write(content)
            _inventory(self.directory, [entry["path"] for kind in ("arrays", "notes") for entry in manifest[kind]] + ["manifest.json"])
            _check((self.directory / "manifest.json").read_bytes() == content, "Manifest changed during commit")
        except (OSError, ValueError) as error:
            self.close()
            raise CacheError("Component commit failed") from error
        self.manifest_sha256 = _digest(content)
        self.close()
        return {"state": "COMMITTED", "commit_scope": "component_bytes_only", "component": self.component,
                "manifest_sha256": self.manifest_sha256, "manifest": _snapshot(manifest)}

    def close(self):
        self._closed = True

    def __enter__(self):
        _check(not self._closed, "Cache writer is closed")
        return self

    def __exit__(self, *_):
        self.close()


def read_cache(root, component, ids, texts, labels=LABELS, required_arrays=(), required_notes=(), *, manifest_sha256=None):
    """Read copied arrays only after all bytes and caller-supplied identities agree.

    Pass the writer's manifest_sha256 to additionally bind the whole manifest to
    a separately retained receipt. Without that anchor, checks detect internal
    inconsistency, not a coordinated rewrite of the manifest and its contents.
    """
    component = _name(component)
    identity = _identity(ids, texts, labels)
    try:
        directory = _root(root) / component
        _check(directory.is_dir() and not _link(directory), "Missing/unsafe component directory")
        path = directory / "manifest.json"
        _check(path.is_file() and not _link(path), "Uncommitted component: missing safe manifest")
        content = path.read_bytes()
        digest = _digest(content)
        if manifest_sha256 is not None:
            _hash(manifest_sha256)
            _check(digest == manifest_sha256, "Manifest hash differs from external receipt")
        manifest = _decode(content)
        _validate_manifest(manifest, component, identity)
        for kind, required in (("arrays", required_arrays), ("notes", required_notes)):
            _check(set(_required(required)).issubset({entry["name"] for entry in manifest[kind]}), "Required entry absent")
        arrays, notes = _verify_files(directory, manifest, include_manifest=True, keep_arrays=True)
        _check(path.read_bytes() == content, "Manifest changed during verification")
        return CacheSnapshot(directory, _snapshot(manifest), digest, arrays, notes)
    except OSError as error:
        raise CacheError("Cache files cannot be read") from error


class NullWriter:
    """Disabled observation: no allocation, validation, file IO, or commit claim."""
    enabled = False
    directory = None
    manifest_sha256 = None

    def __init__(self, *args, **kwargs):
        pass

    def array(self, name, array, metadata=None):
        return None

    def note(self, name, metadata):
        return None

    def finish(self, required_arrays=(), required_notes=(), metadata=None):
        return {"state": "DISABLED", "commit_scope": "disabled"}

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
