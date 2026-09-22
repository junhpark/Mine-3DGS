"""SfM backend contract (§6.2): COLMAPIncremental, COLMAPGlobal (COLMAP >= 4.0 global mapper,
former GLOMAP), experimental GLUEMAP (deferred).

A backend composes commands and runs them. It does not decide which reconstruction is *the*
reconstruction, and it does not write the evidence — that is
``minegs/ingest/video/sfm/run.py``, which enumerates what COLMAP produced and records it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

from minegs.core.errors import ContractError

#: The mappers this project drives. Validated rather than pattern-matched: the old selector
#: treated everything that was not ``global`` as ``incremental``, so ``--mapper globl`` ran a
#: different algorithm than the operator asked for and said nothing.
MapperName = Literal["global", "incremental"]
BackendName = Literal["colmap", "gluemap"]

MATCHERS: dict[str, list[str]] = {
    "sequential": ["colmap", "sequential_matcher"],
    "exhaustive": ["colmap", "exhaustive_matcher"],
    "vocab_tree": ["colmap", "vocab_tree_matcher"],
}


@dataclass
class SfMOptions:
    """What a reconstruction is allowed to be told before it starts.

    ``mapper`` used to live here too and was read by nothing: the backend was chosen by the
    argument to :func:`get_sfm_backend`, so the field was a second answer to a question that
    already had one. It is gone.
    """

    fix_intrinsics: bool = False
    camera_model: str = "PINHOLE"
    camera_params: list[float] | None = None
    #: Where ``camera_params`` came from, when they are given. Fixing an "independent"
    #: reconstruction onto intrinsics measured elsewhere is allowed and is never silent: the
    #: SfM record carries this string, and a reconstruction pinned to TLS-derived intrinsics
    #: is not independent of the TLS however it is labelled.
    intrinsics_source: str = "estimated_by_sfm"
    rig_config: Path | None = None  # COLMAP rig_config.json (360 crops)
    masks_dir: Path | None = None
    matcher: Literal["sequential", "exhaustive", "vocab_tree"] = "sequential"
    #: Loop detection and the vocab-tree matcher both need a vocabulary tree; COLMAP rejects
    #: the request without one, so it is only asked for when the file is there.
    vocab_tree: Path | None = None
    use_gpu: bool = True
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class SfMRun:
    """What one execution did. Evidence is assembled from this, not returned by it."""

    sparse_root: Path
    commands: list[list[str]]
    log: Path | None = None


class SfMBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def commands(self, images_dir: Path, work_dir: Path, opts: SfMOptions) -> list[list[str]]:
        """Ordered CLI commands (pure — testable without COLMAP installed)."""

    def run(self, images_dir: Path, work_dir: Path, opts: SfMOptions) -> SfMRun:
        import shutil
        import subprocess

        if shutil.which("colmap") is None:
            from minegs.core.errors import MissingDependencyError

            raise MissingDependencyError(
                "colmap", "video", "SfM (system binary >= 4.0, see docker/Dockerfile.cpu)"
            )
        work_dir.mkdir(parents=True, exist_ok=True)
        # The mapper writes each reconstruction into a numbered subdirectory of --output_path
        # and expects that path to exist. Nothing created it before, so the first real run
        # would have failed on a directory this code could have made itself.
        (work_dir / "sparse").mkdir(parents=True, exist_ok=True)
        log = work_dir / "sfm.log"
        executed: list[list[str]] = []
        with open(log, "ab") as lf:
            lf.write(f"# minegs sfm {self.name} {images_dir}\n".encode())
            for argv in self.commands(images_dir, work_dir, opts):
                lf.write((" ".join(argv) + "\n").encode())
                lf.flush()
                result = subprocess.run(argv, stdout=lf, stderr=subprocess.STDOUT, check=False)
                executed.append(list(argv))
                if result.returncode != 0:
                    raise ContractError(
                        f"{' '.join(argv[:2])} failed with exit code {result.returncode}. "
                        f"The command log is at {log}; a partial reconstruction is not a "
                        "smaller one, so nothing here is usable."
                    )
        return SfMRun(sparse_root=work_dir / "sparse", commands=executed, log=log)


def get_sfm_backend(name: str = "colmap", mapper: str = "global") -> SfMBackend:
    """The backend for a (name, mapper) pair, refusing anything it does not implement."""
    if name not in get_args(BackendName):
        raise ContractError(
            f"unknown SfM backend {name!r}; this project drives {list(get_args(BackendName))}"
        )
    if name == "gluemap":
        from minegs.ingest.video.sfm.gluemap import GLUEMAP

        return GLUEMAP()
    if mapper not in get_args(MapperName):
        raise ContractError(
            f"unknown mapper {mapper!r}; COLMAP is driven with {list(get_args(MapperName))}. "
            "A name that is not one of these used to fall through to the incremental mapper, "
            "which quietly ran a different algorithm than the one that was asked for."
        )
    if mapper == "global":
        from minegs.ingest.video.sfm.colmap_global import COLMAPGlobal

        return COLMAPGlobal()
    from minegs.ingest.video.sfm.colmap_incremental import COLMAPIncremental

    return COLMAPIncremental()
