"""Training profiles (§8.3). Profiles request *capabilities*, never backend flags, so a
gsplat version bump cannot silently break the profile contract."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from minegs.core.config import VersionedModel
from minegs.core.errors import ContractError

BUILTIN = ("light", "heavy")


class Profile(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    name: str
    description: str = ""
    backend: str = "gsplat"
    default_runner: str = "local"
    max_images: int | None = None
    data_factor: int = Field(default=1, ge=1)
    max_steps: int = Field(default=7000, ge=1)
    requests: dict[str, bool] = Field(default_factory=dict)  # capability -> required
    backend_args: dict[str, Any] = Field(default_factory=dict)

    def required_capabilities(self) -> list[str]:
        return sorted(k for k, required in self.requests.items() if required)

    def optional_capabilities(self) -> list[str]:
        return sorted(k for k, required in self.requests.items() if not required)


def load_profile(name_or_path: str | Path) -> Profile:
    p = Path(name_or_path)
    if p.exists():
        return Profile.load(p)
    if str(name_or_path) in BUILTIN:
        with resources.as_file(resources.files(__package__) / f"{name_or_path}.yaml") as f:
            return Profile.load(f)
    raise ContractError(f"unknown profile {name_or_path!r} (builtin: {BUILTIN}, or a YAML path)")
