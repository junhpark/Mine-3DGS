"""E57 inspection errors (Phase 0B.1).

All of these are ``ContractError`` subclasses so the CLI maps them to exit code 2 and the
user sees a sentence explaining what is wrong with *their file*, not a library traceback.

Reading an E57 must fail with one of these or succeed — never with a bare ``Exception``.
"""

from __future__ import annotations

from pathlib import Path

from minegs.core.errors import ContractError


class E57Error(ContractError):
    """Base for every E57 inspection failure."""


class E57FileNotFoundError(E57Error):
    def __init__(self, path: str | Path) -> None:
        super().__init__(f"E57 file not found: {path}")
        self.path = str(path)


class E57NotAFileError(E57Error):
    def __init__(self, path: str | Path) -> None:
        super().__init__(f"Not a file (is it a directory?): {path}")
        self.path = str(path)


class E57ReadError(E57Error):
    """The file exists but libE57 could not open or parse it."""

    def __init__(self, path: str | Path, detail: str) -> None:
        super().__init__(
            f"Could not read {path} as an E57 file: {detail}. The file may be truncated, "
            "corrupt, or not an E57 at all."
        )
        self.path = str(path)
        self.detail = detail


class E57UnsupportedStructureError(E57Error):
    """The file parses, but its structure is not one this reader can interpret."""

    def __init__(self, path: str | Path, detail: str) -> None:
        super().__init__(f"Unsupported E57 structure in {path}: {detail}")
        self.path = str(path)
        self.detail = detail


class E57NoScansError(E57Error):
    def __init__(self, path: str | Path) -> None:
        super().__init__(
            f"E57 file opened successfully but contains no readable Data3D scans: {path}"
        )
        self.path = str(path)


class E57PoseUnusableError(E57Error):
    """A scan's pose was requested as a transform, but the file does not support one.

    Raised by ``ScanPose.se3()``. The inventory deliberately keeps unusable poses in the
    report (that is how a user learns what is wrong with their file); this is the gate that
    stops such a pose from being silently turned into a working SE(3).
    """

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"pose cannot be used as a rigid transform: {reason}. Inspect it with "
            "`minegs ingest e57 inventory` rather than treating it as identity."
        )
        self.reason = reason


class E57ReadScanError(E57Error):
    """A scan's point payload could not be read.

    Distinct from ``E57ReadError``: the file opened and its metadata parsed, so the inventory
    would have succeeded on it. Only the payload failed.
    """

    def __init__(self, index: int, detail: str) -> None:
        super().__init__(
            f"scan index {index}: point data could not be read ({detail}). Inspect the file "
            "with `minegs ingest e57 inventory` — extraction reads the payload, the inventory "
            "does not."
        )
        self.index = index
