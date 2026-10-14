from __future__ import annotations

import json
from hashlib import sha256
from importlib.resources import files
from pathlib import Path

from .io import library_from_dict, load_library
from .types import Library


def list_libraries() -> list[dict]:
    """Return the bundled library catalog."""
    catalog = files("binary_properties").joinpath("motifs", "catalog.json")
    return json.loads(catalog.read_text(encoding="utf-8"))["libraries"]


def load_builtin_library(name: str) -> Library:
    """Load and verify a bundled motif library."""
    entry = next((row for row in list_libraries() if row["id"] == name), None)
    if entry is None:
        raise ValueError(
            f"Unknown bundled library {name!r}; run 'binary-properties libraries'"
        )
    resource = files("binary_properties").joinpath(
        "motifs", *(name + ".json").split("/")
    )
    data = resource.read_bytes()
    if sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Bundled library {name!r} failed its SHA256 check")
    return library_from_dict(json.loads(data))


def resolve_library(reference: str | Path) -> Library:
    """Load a library file or a catalog ID; an existing file takes precedence."""
    path = Path(reference)
    if path.is_file():
        return load_library(path)
    if str(reference) in {row["id"] for row in list_libraries()}:
        return load_builtin_library(str(reference))
    if isinstance(reference, Path) or path.suffix.lower() == ".json":
        return load_library(path)
    raise ValueError(
        f"Library {str(reference)!r} is neither a file nor a bundled ID; "
        "run 'binary-properties libraries'"
    )
