"""Versioned pydantic config base + migration registry + canonical hashing.

Every on-disk schema (manifest, ingest config, eval config, run.json, profiles) carries
``schema_version`` from day one (§4). ``MigrationRegistry`` chains ``from -> to``
functions over the raw dict *before* pydantic validation, so old files keep loading.
``config_hash`` is the canonical sha256 used everywhere in provenance (§9).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field

from minegs.core.errors import ContractError

Migration = Callable[[dict[str, Any]], dict[str, Any]]
T = TypeVar("T", bound="VersionedModel")


def parse_version(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in str(v).split("."))
    except ValueError as e:  # pragma: no cover
        raise ContractError(f"invalid schema_version {v!r}") from e


class MigrationRegistry:
    """Ordered ``from_version -> (to_version, fn)`` chain for one schema family."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._steps: dict[str, tuple[str, Migration]] = {}

    def register(self, from_version: str, to_version: str) -> Callable[[Migration], Migration]:
        if parse_version(to_version) <= parse_version(from_version):
            raise ValueError("migration must move forward")

        def deco(fn: Migration) -> Migration:
            if from_version in self._steps:
                raise ValueError(f"{self.name}: migration from {from_version} already registered")
            self._steps[from_version] = (to_version, fn)
            return fn

        return deco

    def migrate(self, data: dict[str, Any], target: str) -> dict[str, Any]:
        data = dict(data)
        current = str(data.get("schema_version", "0.0"))
        hops = 0
        while parse_version(current) < parse_version(target):
            if current not in self._steps:
                raise ContractError(
                    f"{self.name}: no migration path from schema_version {current} to {target}"
                )
            to_version, fn = self._steps[current]
            data = fn(dict(data))
            data["schema_version"] = to_version
            current = to_version
            hops += 1
            if hops > 64:  # pragma: no cover
                raise ContractError(f"{self.name}: migration loop")
        if parse_version(current) > parse_version(target):
            raise ContractError(
                f"{self.name}: file schema_version {current} is newer than supported {target}; "
                "upgrade minegs"
            )
        return data

    def versions(self) -> list[str]:
        return sorted(self._steps, key=parse_version)


class VersionedModel(BaseModel):
    """Base for every persisted schema. Subclasses set ``SCHEMA_VERSION`` and ``MIGRATIONS``."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    SCHEMA_VERSION: ClassVar[str] = "1.0"
    MIGRATIONS: ClassVar[MigrationRegistry | None] = None

    schema_version: str = Field(default="1.0")

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        data = dict(data)
        data.setdefault("schema_version", cls.SCHEMA_VERSION)
        if cls.MIGRATIONS is not None:
            data = cls.MIGRATIONS.migrate(data, cls.SCHEMA_VERSION)
        elif parse_version(str(data["schema_version"])) != parse_version(cls.SCHEMA_VERSION):
            raise ContractError(
                f"{cls.__name__}: schema_version {data['schema_version']} != {cls.SCHEMA_VERSION}"
            )
        return cls.model_validate(data)

    @classmethod
    def load(cls: type[T], path: str | Path) -> T:
        path = Path(path)
        raw = load_structured(path)
        if not isinstance(raw, dict):
            raise ContractError(f"{path}: expected a mapping at top level")
        try:
            return cls.from_dict(raw)
        except ContractError:
            raise
        except Exception as e:  # pydantic ValidationError etc.
            raise ContractError(f"{path}: {e}") from e

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        if path.suffix in {".yaml", ".yml"}:
            path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        else:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        return path

    def config_hash(self, exclude: set[str] | None = None) -> str:
        return config_hash(self.model_dump(mode="json", exclude=exclude))


def load_structured(path: str | Path) -> Any:
    path = Path(path)
    text = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def canonical_json(data: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, floats as repr."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(data: Any) -> str:
    """sha256 over canonical JSON. Used for config_hash in provenance (§9)."""
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def short_hash(data: Any, n: int = 6) -> str:
    return config_hash(data)[:n]
