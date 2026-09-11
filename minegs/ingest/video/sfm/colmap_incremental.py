from __future__ import annotations

from pathlib import Path

from minegs.ingest.video.sfm.base import SfMBackend, SfMOptions


def _common(images_dir: Path, work_dir: Path, opts: SfMOptions) -> list[list[str]]:
    db = work_dir / "database.db"
    gpu = "1" if opts.use_gpu else "0"
    fe = [
        "colmap",
        "feature_extractor",
        "--database_path",
        str(db),
        "--image_path",
        str(images_dir),
        "--ImageReader.camera_model",
        opts.camera_model,
        "--ImageReader.single_camera_per_folder",
        "1",
        "--FeatureExtraction.use_gpu",
        gpu,
    ]
    if opts.camera_params:
        fe += ["--ImageReader.camera_params", ",".join(str(v) for v in opts.camera_params)]
    if opts.masks_dir:
        fe += ["--ImageReader.mask_path", str(opts.masks_dir)]
    cmds = [fe]
    if opts.rig_config:
        cmds.append(
            [
                "colmap",
                "rig_configurator",
                "--database_path",
                str(db),
                "--rig_config_path",
                str(opts.rig_config),
            ]
        )
    matcher = {
        "sequential": ["colmap", "sequential_matcher", "--SequentialMatching.loop_detection", "1"],
        "exhaustive": ["colmap", "exhaustive_matcher"],
        "vocab_tree": ["colmap", "vocab_tree_matcher"],
    }[opts.matcher]
    cmds.append([*matcher, "--database_path", str(db), "--FeatureMatching.use_gpu", gpu])
    return cmds


class COLMAPIncremental(SfMBackend):
    name = "colmap_incremental"

    def commands(self, images_dir: Path, work_dir: Path, opts: SfMOptions) -> list[list[str]]:
        cmds = _common(images_dir, work_dir, opts)
        sparse = work_dir / "sparse"
        mapper = [
            "colmap",
            "mapper",
            "--database_path",
            str(work_dir / "database.db"),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse),
        ]
        if opts.fix_intrinsics:
            mapper += [
                "--Mapper.ba_refine_focal_length",
                "0",
                "--Mapper.ba_refine_principal_point",
                "0",
                "--Mapper.ba_refine_extra_params",
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
