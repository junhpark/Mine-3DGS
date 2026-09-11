from __future__ import annotations

from pathlib import Path

import typer

from minegs.cli._common import console, dump_json, run_guarded

app = typer.Typer(no_args_is_help=True)
e57_app = typer.Typer(no_args_is_help=True, help="E57 (TLS) path (§6.1)")
video_app = typer.Typer(no_args_is_help=True, help="video / 360 path (§6.2)")
app.add_typer(e57_app, name="e57")
app.add_typer(video_app, name="video")


@e57_app.command("inventory")
def e57_inventory(
    file: Path = typer.Argument(...), out: Path | None = typer.Option(None, help="Write JSON")
) -> None:
    """List scans / embedded panoramas / poses (header only, pye57)."""
    from minegs.ingest.e57.inventory import inventory

    run_guarded(lambda: dump_json(inventory(file), out))


@e57_app.command("split")
def e57_split(
    file: Path = typer.Argument(...),
    out_dir: Path = typer.Argument(...),
    voxel_m: float = typer.Option(0.01),
) -> None:
    """Write one scanner-frame PLY + pose JSON per station."""
    from minegs.ingest.e57.scan_split import split_scans

    run_guarded(
        lambda: console.print(
            f"wrote {len(split_scans(file, out_dir, voxel_m))} stations to {out_dir}"
        )
    )


@e57_app.command("tiles")
def e57_tiles(
    src: Path = typer.Argument(...),
    out_dir: Path = typer.Argument(...),
    length_m: float = typer.Option(80.0),
    buffer_m: float = typer.Option(15.0),
    voxel_m: float = typer.Option(0.02),
    dry_run: bool = typer.Option(False),
) -> None:
    """PDAL voxel downsample + XY tiling for large cartesian scans."""
    from minegs.ingest.e57.tiles import run_pipeline, tile_pipeline

    def go() -> None:
        pipe = tile_pipeline(src, out_dir, length_m, buffer_m, voxel_m)
        if dry_run:
            dump_json(pipe, None)
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"points: {run_pipeline(pipe, out_dir)}")

    run_guarded(go)


@e57_app.command("pano-calibrate")
def e57_pano_calibrate(
    pano: Path = typer.Argument(...),
    scan_ply: Path = typer.Argument(..., help="scanner-frame PLY"),
    out: Path | None = typer.Option(None, help="overlay PNG"),
) -> None:
    """Golden gate: find the PanoConvention under which reprojected points match the panorama."""
    import numpy as np
    from PIL import Image

    from minegs.core.pointcloud import read_ply
    from minegs.viz.overlay import calibrate_convention, render_overlay, save_overlay

    def go() -> None:
        img = np.asarray(Image.open(pano).convert("RGB"))
        pc = read_ply(scan_ply)
        conv, scores = calibrate_convention(img, pc.xyz)
        console.print(f"best convention: {conv}  score={scores['best']:.3f}")
        dump_json({"convention": conv.to_manifest(), "scores": scores}, None)
        if out:
            save_overlay(out, render_overlay(img, pc.xyz, conv))

    run_guarded(go)


@video_app.command("frames")
def video_frames(
    video: Path = typer.Argument(...),
    out_dir: Path = typer.Argument(...),
    fps: float = typer.Option(2.0),
    dry_run: bool = typer.Option(False),
) -> None:
    """ffmpeg frame extraction."""
    from minegs.ingest.video.frames import extract_command, extract_frames

    def go() -> None:
        if dry_run:
            console.print(" ".join(extract_command(video, out_dir, fps=fps)))
            return
        console.print(f"extracted {len(extract_frames(video, out_dir, fps=fps))} frames")

    run_guarded(go)


@video_app.command("select")
def video_select(
    frames_dir: Path = typer.Argument(...),
    blur: float = typer.Option(60.0),
    hamming: int = typer.Option(6),
    max_frames: int | None = typer.Option(None),
    out: Path | None = typer.Option(None, help="decisions JSON"),
) -> None:
    """Blur + duplicate filtering; writes decisions, does not delete."""
    from minegs.ingest.video.dedup_blur import select_frames

    def go() -> None:
        paths = sorted(list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg")))
        dec = select_frames(paths, blur, hamming, max_frames)
        console.print(f"kept {sum(d.keep for d in dec)} / {len(dec)}")
        dump_json([d.__dict__ for d in dec], out)

    run_guarded(go)


@video_app.command("rig")
def video_rig(
    out: Path = typer.Argument(..., help="rig_config.json"),
    n_yaw: int = typer.Option(6),
    fov_deg: float = typer.Option(90.0),
    size: int = typer.Option(1200),
) -> None:
    """Write COLMAP rig_config.json for a 360 ring crop."""
    from minegs.ingest.common.equirect import RingCropSpec
    from minegs.ingest.video.rig import write_rig_config

    run_guarded(
        lambda: console.print(
            f"wrote {write_rig_config(RingCropSpec(n_yaw=n_yaw, fov_deg=fov_deg, width=size, height=size), out)}"
        )
    )


@video_app.command("sfm")
def video_sfm(
    images_dir: Path = typer.Argument(...),
    work_dir: Path = typer.Argument(...),
    mapper: str = typer.Option("global"),
    fix_intrinsics: bool = typer.Option(False),
    rig_config: Path | None = typer.Option(None),
    masks: Path | None = typer.Option(None),
    dry_run: bool = typer.Option(False),
) -> None:
    """Run COLMAP (>= 4.0) SfM; --dry-run prints the commands."""
    from minegs.ingest.video.sfm import SfMOptions, get_sfm_backend

    def go() -> None:
        be = get_sfm_backend("colmap", mapper)
        opts = SfMOptions(
            mapper=mapper, fix_intrinsics=fix_intrinsics, rig_config=rig_config, masks_dir=masks
        )  # type: ignore[arg-type]
        if dry_run:
            for c in be.commands(images_dir, work_dir, opts):
                console.print(" ".join(c))
            return
        r = be.run(images_dir, work_dir, opts)
        console.print(
            f"{r.backend}: registered {r.n_registered} images, {r.n_points} points -> {r.sparse_dir}"
        )

    run_guarded(go)
