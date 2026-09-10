"""Exception hierarchy. Keep it small; the CLI maps these to exit codes."""

from __future__ import annotations


class MinegsError(Exception):
    """Base class for all minegs errors."""


class ContractError(MinegsError):
    """The dataset / manifest / run contract is violated (§4, §9)."""


class ProtocolViolation(MinegsError):  # noqa: N818 — domain term (§5)
    """An evaluation claim is not supported by the manifest's split/initialization (§5)."""


class FrameError(MinegsError):
    """Coordinate frame misuse (§3): wrong frame, non-rigid transform, float32 overflow."""


class MissingDependencyError(MinegsError):
    """An optional dependency (pye57, pdal, gsplat, viser, ...) is required but absent."""

    def __init__(self, package: str, extra: str, purpose: str = "") -> None:
        msg = f"'{package}' is required{(' for ' + purpose) if purpose else ''}. "
        msg += f"Install with: pip install 'minegs[{extra}]'"
        super().__init__(msg)
        self.package = package
        self.extra = extra


class NotYetImplementedError(MinegsError):
    """Interface exists, implementation is scheduled for a later phase (§13)."""

    def __init__(self, what: str, phase: str) -> None:
        super().__init__(f"{what} is scheduled for Phase {phase} (docs/ARCHITECTURE.md §13).")
        self.phase = phase


class NoGpuError(MinegsError):
    """LocalRunner found no CUDA device; suggest RunPod routing (§8.2)."""
