from __future__ import annotations

from pathlib import Path

from minegs.ingest.video.sfm.base import MATCHERS, SfMBackend, SfMOptions


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
    matcher = list(MATCHERS[opts.matcher])
    if opts.vocab_tree is not None:
        # Loop detection is what closes a tunnel traverse that comes back on itself, and it is
        # a vocabulary-tree search: COLMAP rejects the request when no tree is given, so it is
        # asked for only when there is one to search.
        matcher += ["--VocabTreeMatching.vocab_tree_path", str(opts.vocab_tree)]
        if opts.matcher == "sequential":
            matcher += ["--SequentialMatching.loop_detection", "1"]
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
