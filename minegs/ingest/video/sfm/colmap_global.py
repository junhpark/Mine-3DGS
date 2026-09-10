"""COLMAP >= 4.0 ``global_mapper`` (GLOMAP merged upstream; standalone repo archived 2026-03).
Repetitive / low-texture drifts may need the incremental fallback (§6.2)."""

from __future__ import annotations

from pathlib import Path

from minegs.ingest.video.sfm.base import SfMBackend, SfMOptions
from minegs.ingest.video.sfm.colmap_incremental import _common


class COLMAPGlobal(SfMBackend):
    name = "colmap_global"

    def commands(self, images_dir: Path, work_dir: Path, opts: SfMOptions) -> list[list[str]]:
        cmds = _common(images_dir, work_dir, opts)
        sparse = work_dir / "sparse"
        mapper = [
            "colmap",
            "global_mapper",
            "--database_path",
            str(work_dir / "database.db"),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse),
        ]
        if opts.fix_intrinsics:
            mapper += [
                "--GlobalMapper.ba_refine_focal_length",
                "0",
                "--GlobalMapper.ba_refine_principal_point",
                "0",
                "--GlobalMapper.ba_refine_extra_params",
                "0",
            ]
        for k, v in opts.extra.items():
            mapper += [f"--{k}", v]
        cmds.append(mapper)
        cmds.append(
            [
                "colmap",
                "model_converter",
                "--input_path",
                str(sparse / "0"),
                "--output_path",
                str(sparse / "0"),
                "--output_type",
                "TXT",
            ]
        )
        return cmds
