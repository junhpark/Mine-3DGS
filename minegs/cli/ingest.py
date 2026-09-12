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
    file: Path = typer.Argument(..., help="Path to the .e57 file"),
    json_out: Path | None = typer.Option(
        None, "--json", help="Also write the full machine-readable report here"
    ),
    no_hash: bool = typer.Option(
        False,
        "--no-hash",
        help="Skip the SHA-256 (faster on very large files; recorded as skipped)",
    ),
) -> None:
    """Inspect an E57: how many scans, what each declares, whether poses are usable.

    Reads header metadata only — no point data is loaded, so this is fast and safe on
    multi-gigabyte files. It reports what the file *declares*; it does not interpret the
    coordinates as TLS_GLOBAL and does not map panoramas to stations (Phase 0B.2).
    """
    from minegs.ingest.e57.inventory import inventory

    def go() -> None:
        inv = inventory(file, compute_hash=not no_hash)
        _print_inventory(inv)
        if json_out:
            inv.save(json_out)
            console.print(f"\nwrote {json_out}")

    run_guarded(go)


def _yn(value: bool) -> str:
    return "[green]yes[/]" if value else "[dim]no[/]"


def _pose_label(scan) -> str:
    """Render pose_status. "declared but unreadable" must never read as "none declared"."""
    if scan.pose_status == "absent":
        return "[dim]none declared[/]"
    if scan.pose_status == "unreadable":
        return "[red]DECLARED BUT UNREADABLE[/] (the file has a pose node we could not parse)"
    if scan.pose_status == "invalid":
        issues = "; ".join(scan.pose.validation.issues) if scan.pose else "validation failed"
        return f"[red]INVALID[/] ({issues})"
    if scan.pose_status == "identity":
        return "[yellow]identity[/] (file declares no displacement for this scan)"
    t = scan.pose.translation_m
    return f"[green]yes[/]  translation = ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m"


def _print_inventory(inv) -> None:
    """Block-per-scan report: stays readable at any terminal width and for any scan count."""
    f = inv.file
    console.print(f"[bold]E57 file:[/] {f.file_name}")
    console.print(f"  path:   {f.path}")
    console.print(f"  size:   {f.size_bytes:,} bytes ({f.size_bytes / 1e9:.2f} GB)")
    if f.sha256:
        console.print(f"  sha256: {f.sha256}")
    else:
        console.print(f"  sha256: [yellow]not computed[/] ({f.hash_skipped_reason})")
    if f.e57_library_version:
        console.print(f"  writer: {f.e57_library_version}")
    if f.coordinate_metadata:
        console.print(f"  coordinate metadata: {f.coordinate_metadata}")
    console.print(
        f"[bold]Scans:[/] {inv.scan_count}   "
        f"[dim](usable as declared: {inv.usable_scan_count()})[/]"
    )

    for station, s in zip(inv.station_candidates, inv.scans, strict=True):
        console.print(f"\n[bold]Scan {station.station_id}[/]")
        console.print(f"  scan_id:       {s.scan_id}")
        if s.name:
            console.print(f"  name:          {s.name}")
        if s.guid:
            console.print(f"  guid:          {s.guid}")
        console.print(
            f"  points:        {s.point_count:,}"
            if s.point_count is not None
            else "  points:        [dim]not declared[/]"
        )
        console.print(f"  Cartesian XYZ: {_yn(s.has_cartesian_xyz)}")
        console.print(f"  Spherical:     {_yn(s.has_spherical)}")
        console.print(f"  RGB:           {_yn(s.has_rgb)}")
        console.print(f"  Intensity:     {_yn(s.has_intensity)}")
        console.print(f"  Row/column:    {_yn(s.has_row_column)}")
        console.print(f"  Pose:          {_pose_label(s)}")
        if s.bounds is not None and s.bounds.valid:
            dx, dy, dz = s.bounds.extent()
            console.print(
                f"  Bounds extent: {dx:.2f} x {dy:.2f} x {dz:.2f} m ({s.bounds.source_field})"
            )
        for msg in s.issues:
            console.print(f"  [red]problem:[/] {msg}")
        for msg in s.notes:
            console.print(f"  [yellow]note:[/] {msg}")

    console.print(
        "\n[bold]Images:[/] "
        + (
            f"images2D present with {inv.images.image_count} entries"
            if inv.images.has_images2d and inv.images.image_count is not None
            else "images2D present but the entry count could not be read"
            if inv.images.has_images2d
            else "no images2D structure"
        )
        + " [dim]— detected only; station/panorama mapping is Phase 0B.2[/]"
    )
    console.print(
        f"[bold]Station candidates:[/] {len(inv.station_candidates)}, one per scan, "
        "status [yellow]inferred_from_scan[/] [dim](unconfirmed until Phase 0B.2)[/]"
    )
    console.print(
        "[dim]Poses are in the file's own SOURCE frame. Declaring them TLS_GLOBAL is a later "
        "decision (docs/ARCHITECTURE.md §3).[/]"
    )

    if inv.issues:
        console.print("\n[bold red]Problems[/]")
        for msg in inv.issues:
            console.print(f"  [red]•[/] {msg}")
    if inv.notes:
        console.print("\n[bold]Notes[/]")
        for msg in inv.notes:
            console.print(f"  [yellow]•[/] {msg}")
    if not inv.has_any_issue():
        console.print("\n[green]No problems detected.[/]")


@e57_app.command("pano-map")
def e57_pano_map(
    file: Path = typer.Argument(..., help="Path to the .e57 file"),
    json_out: Path | None = typer.Option(
        None, "--json", help="Write the full PanoMappingReport here"
    ),
    mapping: Path | None = typer.Option(
        None, "--mapping", help="Explicit station/scan <-> image mapping (CSV or JSON)"
    ),
    vendor_manifest: Path | None = typer.Option(
        None, "--vendor-manifest", help="Machine-generated vendor index (JSON)"
    ),
    images_dir: Path | None = typer.Option(
        None, "--images-dir", help="Map external image files in this directory instead of /images2D"
    ),
    no_hash: bool = typer.Option(False, "--no-hash", help="Skip the SHA-256 (recorded as skipped)"),
) -> None:
    """Discover images and map them to scans — from evidence only.

    Reads metadata, never pixels or point data. An image is mapped when the E57 associates it
    with exactly one scan, when a vendor index says so, or when you say so with --mapping.
    Index equality, equal counts, file order and name similarity never produce a mapping
    (docs/ARCHITECTURE.md §3, docs/ROADMAP.md Phase 0B.2): a panorama attributed to the wrong
    station trains and converges just as well as a correct one.
    """
    from minegs.ingest.e57.mapping import build_mapping_report

    def go() -> None:
        rep = build_mapping_report(
            file,
            mapping=mapping,
            vendor_manifest=vendor_manifest,
            images_dir=images_dir,
            compute_hash=not no_hash,
        )
        _print_mapping(rep)
        if json_out:
            rep.save(json_out)
            console.print(f"\nwrote {json_out}")

    run_guarded(go)


_STATUS_STYLE = {
    "confirmed": "green",
    "manual": "cyan",
    "unmapped": "yellow",
    "ambiguous": "red",
    "orphan": "red",
    "conflict": "red",
}


def _print_mapping(rep) -> None:
    """Block-per-image report. Every line says what the evidence was, or that there was none."""
    console.print(f"[bold]Source:[/] {rep.source_file}")
    if rep.image_root:
        console.print(f"  images from: {rep.image_root}")
    console.print(
        f"[bold]Scans:[/] {rep.scan_count}   [bold]Images:[/] {len(rep.images)}   "
        f"[dim](resolved: {len(rep.resolved())})[/]"
    )

    by_id = {im.image_id: im for im in rep.images}
    for m in rep.mappings:
        im = by_id[m.image_id]
        style = _STATUS_STYLE.get(m.status, "white")
        console.print(f"\n[bold]{m.image_id}[/]  [{style}]{m.status}[/]")
        if im.name:
            console.print(f"  name:           {im.name}")
        console.print(
            f"  representation: {im.representation}"
            + ("  [green](panorama candidate)[/]" if im.panorama_candidate else "")
        )
        if im.width and im.height:
            console.print(f"  size:           {im.width} x {im.height}")
        console.print(f"  scan:           {m.scan_id or '[dim]not determined[/]'}")
        console.print(f"  station:        {m.station_id or '[dim]not determined[/]'}")
        console.print(f"  evidence:       {m.evidence_type}", highlight=False)
        if m.evidence_value:
            console.print(f"  evidence value: {m.evidence_value}", highlight=False)
        console.print(f"  why:            {m.reason}")
        if m.candidate_scan_ids:
            console.print(f"  candidates:     {', '.join(m.candidate_scan_ids)}")
        for hint in m.hints:
            console.print(f"  [yellow]hint:[/] {hint} [dim](a hint is not evidence)[/]")

    if rep.issues:
        console.print("\n[bold red]Problems[/]")
        for msg in rep.issues:
            console.print(f"  [red]•[/] {msg}")
    if rep.notes:
        console.print("\n[bold]Notes[/]")
        for msg in rep.notes:
            console.print(f"  [yellow]•[/] {msg}")
    console.print(
        "\n[dim]Mapping is evidence-based: scan/image index equality, equal counts, file order "
        "and name similarity are never used (docs/ROADMAP.md Phase 0B.2).[/]"
    )


@e57_app.command("extract")
def e57_extract(
    file: Path = typer.Argument(..., help="Path to the .e57 file"),
    work_dir: Path = typer.Argument(..., help="Staging directory to write (NOT a dataset/)"),
    voxel: float | None = typer.Option(None, "--voxel", help="Voxel size in metres, e.g. 0.01"),
    scan: list[str] = typer.Option(
        [], "--scan", help="Extract only these scan ids (repeatable), e.g. --scan scan_000"
    ),
    mapping: Path | None = typer.Option(None, "--mapping", help="Explicit image mapping file"),
    vendor_manifest: Path | None = typer.Option(
        None, "--vendor-manifest", help="Machine-generated vendor index (JSON)"
    ),
    images_dir: Path | None = typer.Option(
        None, "--images-dir", help="Map external image files in this directory"
    ),
    no_images: bool = typer.Option(False, "--no-images", help="Extract scans only"),
    raw: bool = typer.Option(
        False,
        "--raw",
        help="Write scanner-frame clouds marked unregistered instead of SOURCE-frame ones",
    ),
    max_scan_points: int | None = typer.Option(
        None, "--max-scan-points", help="Refuse scans larger than this (pye57 reads a scan whole)"
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace existing output"),
    no_hash: bool = typer.Option(
        False,
        "--no-hash",
        help="Skip hashing the inputs (E57, mapping files, external images). Output digests "
        "are still recorded — they are the record of what this run produced.",
    ),
) -> None:
    """Extract scans and supported images into a Phase 0B staging directory.

    Writes inventory.json, pano_mapping.json, scans/<scan_id>.ply + .pose.json,
    images/<image_id>.jpg and extraction_manifest.json. The result is staging, not a dataset:
    points stay in the E57's own SOURCE frame (or, with --raw, each scan's SCANNER frame).
    Declaring TLS_GLOBAL, choosing a LOCAL_METRIC origin and building dataset/ are Phase 0C.

    By default every selected scan must declare a usable pose. --raw writes unregistered
    scanner-frame clouds instead; a broken pose is refused either way, and neither path ever
    substitutes identity for a missing one.
    """
    from minegs.ingest.e57.extract import extract

    def go() -> None:
        manifest = extract(
            file,
            work_dir,
            scan_ids=list(scan) or None,
            voxel_m=voxel,
            registered=not raw,
            with_images=not no_images,
            mapping=mapping,
            vendor_manifest=vendor_manifest,
            images_dir=images_dir,
            compute_hash=not no_hash,
            overwrite=overwrite,
            max_scan_points=max_scan_points,
        )
        _print_extraction(manifest)

    run_guarded(go)


def _print_extraction(m) -> None:
    console.print(f"[bold]Extracted:[/] {m.source_e57}")
    console.print(f"  into:   {m.work_dir}")
    console.print(
        f"  frame:  [bold]{m.output_frame}[/] ({m.registration}) "
        "[dim]— not TLS_GLOBAL, not a dataset[/]"
    )
    for s in m.scan_outputs:
        console.print(f"\n[bold]{s.scan_id}[/]  {Path(s.path).name}")
        console.print(
            f"  points:      {s.point_count_output:,}"
            + (
                f"  [dim](declared {s.point_count_input:,}"
                if s.point_count_input is not None
                else "  [dim]("
            )
            + f", {s.invalid_points_removed:,} invalid removed)[/]"
        )
        console.print(f"  pose:        {s.pose_status}")
        console.print(f"  attributes:  {', '.join(s.attributes) or '[dim]none[/]'}")
        if s.color_conversion:
            console.print(f"  colour:      {s.color_conversion}")
        if s.voxel_m:
            console.print(f"  voxel:       {s.voxel_m} m")
        for msg in s.issues:
            console.print(f"  [red]problem:[/] {msg}")
    for im in m.image_outputs:
        console.print(
            f"\n[bold]{im.image_id}[/]  {Path(im.path).name}  {im.representation}  "
            f"mapping: {im.mapping_status}"
            + ("" if im.extracted else "  [dim](referenced, not copied)[/]")
        )
    for sk in m.skipped_images:
        console.print(f"\n[yellow]{sk.image_id} skipped[/]  {sk.reason}")
    if m.issues:
        console.print("\n[bold red]Problems[/]")
        for msg in m.issues:
            console.print(f"  [red]•[/] {msg}")
    if m.notes:
        console.print("\n[bold]Notes[/]")
        for msg in m.notes:
            console.print(f"  [yellow]•[/] {msg}")


@e57_app.command("split")
def e57_split(
    file: Path = typer.Argument(...),
    out_dir: Path = typer.Argument(...),
    voxel_m: float = typer.Option(0.01),
) -> None:
    """Deprecated: flat scanner-frame PLY + pose JSON per scan. Use `extract` instead.

    Kept for the Phase 0A dataset builder. `extract` writes the same points plus the mapping
    report, the image outputs and an extraction manifest, can place points in the SOURCE
    frame, and publishes its output all-or-nothing. This command writes scan by scan with no
    rollback, so an interrupted run leaves whatever it had written.
    """
    from minegs.ingest.e57.scan_split import split_scans

    def go() -> None:
        console.print(
            "[yellow]`split` is deprecated; use `minegs ingest e57 extract` "
            "(docs/ROADMAP.md Phase 0B.3).[/]"
        )
        console.print(f"wrote {len(split_scans(file, out_dir, voxel_m))} stations to {out_dir}")

    run_guarded(go)


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
