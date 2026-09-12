"""Guarded libE57 node access, shared by every E57 reader in this package.

One place that knows how to open an E57 and how to read a node that may not exist. Without
it each reader grows its own guards and they drift — the failure mode that let the inventory
and the splitter disagree about poses in Phase 0B.1.

Every accessor here answers "what does the file declare?" and returns ``None`` when the
answer is "nothing". None of them guess.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from minegs.core.errors import MissingDependencyError
from minegs.ingest.e57.exceptions import (
    E57FileNotFoundError,
    E57NotAFileError,
    E57ReadError,
)


def pye57_module() -> Any:
    try:
        import pye57
    except ImportError as e:
        raise MissingDependencyError("pye57", "e57", "reading E57 files") from e
    return pye57


@contextmanager
def open_e57(path: str | Path) -> Iterator[Any]:
    """Open an E57 for reading, mapping every failure to an explaining error.

    Closes the handle on the way out, including when the body raises.
    """
    p = Path(path)
    if not p.exists():
        raise E57FileNotFoundError(p)
    if not p.is_file():
        raise E57NotAFileError(p)
    pye57 = pye57_module()
    try:
        handle = pye57.E57(str(p))
    except Exception as e:  # libe57 raises its own exception type
        raise E57ReadError(p, str(e)) from e
    try:
        yield handle
    finally:
        handle.close()


def fields(node: Any) -> list[str]:
    """Child element names, or ``[]`` if the node cannot be enumerated."""
    try:
        return [node.get(i).elementName() for i in range(node.childCount())]
    except Exception:
        return []


def value(node: Any, *path: str) -> Any:
    """``node[a][b].value()``, or ``None`` if any step is missing or unreadable."""
    cur = node
    try:
        for key in path:
            cur = cur[key]
        return cur.value()
    except Exception:  # libe57 raises for an undefined node
        return None


def str_value(node: Any, *path: str) -> str | None:
    """Like :func:`value`, as a stripped string. Empty strings become ``None``."""
    v = value(node, *path)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def is_defined(node: Any, key: str) -> bool:
    try:
        return bool(node.isDefined(key))
    except Exception:
        return False
