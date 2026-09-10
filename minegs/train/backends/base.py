"""Backend adapter contract (§8.1).

An adapter declares ``BackendCapabilities``; a profile requests capabilities; the runner
refuses a (profile, backend) pair whose *required* capabilities are missing. Adapter
responsibilities: ``(dataset, profile) -> command`` and normalising outputs into the
``runs/<run_id>/`` convention with every ``.ply`` re-expressed in LOCAL_METRIC (§3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path

from minegs.core.errors import ContractError
from minegs.core.frames import Sim3
from minegs.train.profiles import Profile

RUN_LAYOUT = ("point_cloud", "ckpt", "log", "run.json")


@dataclass(frozen=True)
class BackendCapabilities:
    appearance_embedding: bool = False
    bilateral_grid: bool = False
    depth_loss: bool = False
    normal_loss: bool = False
    antialiasing: bool = False
    absgrad: bool = False
    mcmc_strategy: bool = False
    pose_refinement: bool = False
    depth_render: bool = False  # can export depth maps (needed for GS -> surface, §1.7)
    resume: bool = False

    def has(self, name: str) -> bool:
        if name not in {f.name for f in fields(self)}:
            raise ContractError(
                f"unknown capability {name!r}; known: {[f.name for f in fields(self)]}"
            )
        return bool(getattr(self, name))

    def as_dict(self) -> dict[str, bool]:
        return {f.name: bool(getattr(self, f.name)) for f in fields(self)}


@dataclass
class TrainCommand:
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    workdir: str | None = None
    # LOCAL_METRIC <- BACKEND_INTERNAL (similarity: backends normalise scene scale), known
    # *before* training. Identity when the backend is told not to normalise.
    T_local_from_internal: Sim3 = field(default_factory=Sim3.identity)


class TrainBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def capabilities(self) -> BackendCapabilities: ...

    @abstractmethod
    def build_command(
        self,
        dataset_dir: Path,
        out_dir: Path,
        profile: Profile,
        resume: bool = False,
        **kwargs: object,
    ) -> TrainCommand:
        """``dataset_dir`` is the *staged* dataset written by ``minegs.train.staging``."""

    @abstractmethod
    def normalize_outputs(
        self, out_dir: Path, run_dir: Path, T_local_from_internal: Sim3
    ) -> list[Path]:
        """Move/convert outputs into ``run_dir`` with ``point_cloud/*.ply`` in LOCAL_METRIC.
        Returns the list of produced PLY files."""

    def check_profile(self, profile: Profile) -> list[str]:
        """Return capabilities the profile *requires* that this backend lacks."""
        caps = self.capabilities()
        return [c for c in profile.required_capabilities() if not caps.has(c)]

    def resolve_requests(self, profile: Profile) -> dict[str, bool]:
        """Capabilities that will actually be enabled (required ones must exist; optional if present)."""
        missing = self.check_profile(profile)
        if missing:
            raise ContractError(
                f"backend {self.name} lacks required capabilities {missing} for profile {profile.name}"
            )
        caps = self.capabilities()
        return {c: caps.has(c) for c in profile.requests}


def get_backend(name: str) -> TrainBackend:
    if name == "gsplat":
        from minegs.train.backends.gsplat import GsplatBackend

        return GsplatBackend()
    if name in ("splatfacto", "pgsr", "2dgs"):
        from minegs.core.errors import NotYetImplementedError

        raise NotYetImplementedError(f"backend {name}", "3" if name != "splatfacto" else "1+")
    if name == "inria":
        raise ContractError(
            "INRIA 3DGS is non-commercial and not shipped (§8.1); call it externally for baselines"
        )
    raise ContractError(f"unknown backend {name!r}")
