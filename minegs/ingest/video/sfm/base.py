"""SfM backend contract (§6.2): COLMAPIncremental, COLMAPGlobal (COLMAP >= 4.0 global mapper,
former GLOMAP), experimental GLUEMAP (Phase 2+)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from minegs.core.errors import ContractError


@dataclass
class SfMOptions:
    mapper: Literal["incremental", "global"] = "global"
    fix_intrinsics: bool = False
    camera_model: str = "PINHOLE"
    camera_params: list[float] | None = None
    rig_config: Path | None = None  # COLMAP rig_config.json (360 crops)
    masks_dir: Path | None = None
    matcher: Literal["sequential", "exhaustive", "vocab_tree"] = "sequential"
    use_gpu: bool = True
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class SfMResult:
    sparse_dir: Path
    n_registered: int
    n_points: int
    backend: str
    log: Path | None = None


class SfMBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def commands(self, images_dir: Path, work_dir: Path, opts: SfMOptions) -> list[list[str]]:
        """Ordered CLI commands (pure — testable without COLMAP installed)."""

    def run(self, images_dir: Path, work_dir: Path, opts: SfMOptions) -> SfMResult:
        import shutil
        import subprocess

        if shutil.which("colmap") is None:
            from minegs.core.errors import MissingDependencyError

            raise MissingDependencyError(
                "colmap", "video", "SfM (system binary >= 4.0, see docker/Dockerfile.cpu)"
            )
        work_dir.mkdir(parents=True, exist_ok=True)
        log = work_dir / "sfm.log"
        with open(log, "ab") as lf:
            for argv in self.commands(images_dir, work_dir, opts):
                lf.write((" ".join(argv) + "\n").encode())
                subprocess.run(argv, check=True, stdout=lf, stderr=subprocess.STDOUT)
        sparse = work_dir / "sparse" / "0"
        from minegs.ingest.common import colmap_io

        if not (sparse / "cameras.txt").exists():
            subprocess.run(
                [
                    "colmap",
                    "model_converter",
                    "--input_path",
                    str(sparse),
                    "--output_path",
                    str(sparse),
                    "--output_type",
                    "TXT",
                ],
                check=True,
            )
        model = colmap_io.read_model(sparse)
        return SfMResult(sparse, len(model.images), len(model.points3D), self.name, log)


def get_sfm_backend(name: str = "colmap", mapper: str = "global") -> SfMBackend:
    if name == "colmap":
        if mapper == "global":
            from minegs.ingest.video.sfm.colmap_global import COLMAPGlobal

            return COLMAPGlobal()
        from minegs.ingest.video.sfm.colmap_incremental import COLMAPIncremental

        return COLMAPIncremental()
    if name == "gluemap":
        from minegs.ingest.video.sfm.gluemap import GLUEMAP

        return GLUEMAP()
    raise ContractError(f"unknown SfM backend {name!r}")
