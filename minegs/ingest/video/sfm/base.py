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

#: The COLMAP these commands are written against. 3.x is not an older version of the same
#: interface: it has neither ``global_mapper`` nor ``rig_configurator``, and it names the
#: feature-extraction options ``SiftExtraction.*`` where 4.0 names them ``FeatureExtraction.*``.
#: So every command below is rejected by it — including by the 3.9.1 that `apt install colmap`
#: puts on an Ubuntu 24.04 LTS machine, which is the COLMAP an operator is most likely to have.
MIN_COLMAP: tuple[int, int] = (4, 0)

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
        require_colmap_version()
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


def colmap_version() -> tuple[int, ...] | None:
    """The installed COLMAP's version, or ``None`` when it cannot be established.

    Read from the banner rather than from ``--version``, which 3.x does not accept at all.
    """
    import re

    from minegs.core.provenance import _cli_version

    banner = _cli_version("colmap")
    if not banner:
        return None
    m = re.search(r"COLMAP\s+v?(\d+)\.(\d+)(?:\.(\d+))?", banner)
    return tuple(int(g) for g in m.groups() if g is not None) if m else None


def require_colmap_version() -> None:
    """Refuse a COLMAP known to be too old, before it fails on an unrecognised option.

    Presence was the only thing checked, and presence is not enough: 3.9.1 is installed by
    ``apt install colmap`` on the current Ubuntu LTS, and against it the very first command
    dies with ``unrecognised option '--FeatureExtraction.use_gpu'`` and a pointer to a log —
    which reads like a bug in this project rather than like the wrong COLMAP.

    A version that cannot be established is *not* refused. Refusing what we failed to parse
    would block a perfectly good 4.x behind a changed banner; what is refused is what we know
    is wrong, and what we do not know is recorded as unknown (:func:`_cli_version`).
    """
    version = colmap_version()
    if version is None or version >= MIN_COLMAP:
        return
    raise ContractError(
        f"COLMAP {'.'.join(str(v) for v in version)} is installed, and these commands are "
        f"written for COLMAP >= {MIN_COLMAP[0]}.{MIN_COLMAP[1]}. 3.x is not an older version "
        "of the same interface: it has no `global_mapper` and no `rig_configurator`, and it "
        "names the feature-extraction options `SiftExtraction.*` rather than "
        "`FeatureExtraction.*`, so the first command would fail on an unrecognised option "
        "rather than here. `apt install colmap` on Ubuntu 24.04 LTS gives 3.9.1; "
        "docker/Dockerfile.cpu has the version this project drives."
    )


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
