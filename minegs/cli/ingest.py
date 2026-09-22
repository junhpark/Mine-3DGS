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
    scale_width: int | None = typer.Option(None, help="downscale to this width, keeping aspect"),
    start_s: float | None = typer.Option(None, help="skip this many seconds of the video"),
    duration_s: float | None = typer.Option(None, help="extract only this many seconds"),
    pattern: str = typer.Option("v_%06d.png"),
    dry_run: bool = typer.Option(False),
) -> None:
    """ffmpeg frame extraction. For a recorded, selectable set use `video frameset`."""
    from minegs.ingest.video.frames import extract_command, extract_frames

    kw = {
        "fps": fps,
        "pattern": pattern,
        "scale_width": scale_width,
        "start_s": start_s,
        "duration_s": duration_s,
    }
    kw = {k: v for k, v in kw.items() if v is not None}

    def go() -> None:
        if dry_run:
            console.print(" ".join(extract_command(video, out_dir, **kw)))
            return
        console.print(f"extracted {len(extract_frames(video, out_dir, **kw))} frames")

    run_guarded(go)


@video_app.command("frameset")
def video_frameset(
    source: Path = typer.Argument(..., help="video file, or a directory of images"),
    out_dir: Path = typer.Argument(..., help="where the frame set artifact is written"),
    kind: str = typer.Option("video", help="video | video360 | image_set"),
    fps: float | None = typer.Option(None),
    scale_width: int | None = typer.Option(None),
    start_s: float | None = typer.Option(None),
    duration_s: float | None = typer.Option(None),
    blur: float = typer.Option(60.0, help="variance-of-Laplacian floor"),
    hamming: int = typer.Option(6, help="dHash distance below which a frame is a duplicate"),
    max_frames: int | None = typer.Option(None),
    n_yaw: int = typer.Option(8, help="360: perspective views around the ring"),
    fov_deg: float = typer.Option(90.0, help="360: crop field of view"),
    size: int = typer.Option(1200, help="360: crop width and height"),
    pitches_deg: str = typer.Option("0", help="360: comma-separated pitches"),
    yaw_offset_deg: float = typer.Option(0.0),
    pano_source: str | None = typer.Option(
        None, help="360: what the panorama convention was derived from (required)"
    ),
    pano_vendor: str | None = typer.Option(None),
    pano_az_sign: int = typer.Option(1),
    pano_el_flip: bool = typer.Option(False),
    pano_az_offset_deg: float = typer.Option(0.0),
    nadir_el_deg: float | None = typer.Option(
        None, help="mask everything looking below this elevation (tripod, operator)"
    ),
    overwrite: bool = typer.Option(False),
) -> None:
    """Extract, select, crop and mask one input into the set SfM will be given.

    The artifact it writes is the identity of that set: which video, which settings, which
    frames survived selection and why, and for 360 which crop was cut from which panorama at
    which yaw. `ingest video sfm` reads it, so a frame the selector rejected cannot reach the
    reconstruction and an edited image is not the image that was selected.
    """
    from minegs.ingest.common.equirect import RingCropSpec
    from minegs.ingest.common.geometry import PanoConvention
    from minegs.ingest.video.build import build_frameset

    def go() -> None:
        from minegs.core.errors import ContractError

        if kind not in ("video", "video360", "image_set"):
            raise ContractError(f"unknown frame set kind {kind!r} (video, video360, image_set)")
        ring = convention = None
        if kind == "video360":
            pitches = [float(v) for v in pitches_deg.split(",") if v.strip()]
            ring = RingCropSpec(
                n_yaw=n_yaw,
                fov_deg=fov_deg,
                width=size,
                height=size,
                pitches_deg=pitches,
                yaw_offset_deg=yaw_offset_deg,
            )
            if not pano_source:
                raise ContractError(
                    "--pano-source is required for a 360 frame set: the panorama convention "
                    "decides where azimuth zero is and which way elevation runs, and with "
                    "fixed crop intrinsics it becomes a geometric constraint on the "
                    "reconstruction. An unmeasured one must not be presented as measured."
                )
            convention = PanoConvention(
                az_sign=pano_az_sign,
                el_flip=pano_el_flip,
                az_offset_deg=pano_az_offset_deg,
                source=pano_source,
                vendor=pano_vendor,
            )
        # What the source *is* decides how the frames arrive; `kind` says what the pictures
        # are. Reading the directory case off `kind` meant a 360 survey delivered as a folder
        # of panoramas — already extracted, or exported by the camera — could only be ingested
        # as `image_set`, which throws away the ring crops that are the whole 360 path.
        if source.is_dir():
            video, image_dir = None, source
        elif source.is_file():
            video, image_dir = source, None
        else:
            raise ContractError(f"{source}: neither a video file nor a directory of images")
        rec, root = build_frameset(
            out_dir,
            kind=kind,  # type: ignore[arg-type]
            video=video,
            image_dir=image_dir,
            fps=fps,
            scale_width=scale_width,
            start_s=start_s,
            duration_s=duration_s,
            blur_threshold=blur,
            hamming_threshold=hamming,
            max_frames=max_frames,
            ring=ring,
            pano_convention=convention,
            nadir_el_deg=nadir_el_deg,
            overwrite=overwrite,
        )
        sel = rec.selection
        console.print(f"frame set [bold]{rec.frameset_id}[/] in {root}")
        if sel is not None:
            console.print(f"  selected {sel.kept} / {sel.considered} frames")
        if rec.crops:
            console.print(
                f"  {len(rec.crops)} crops from {len(set(c.parent_frame for c in rec.crops))} panoramas"
            )
        if rec.masks:
            console.print(f"  {len(rec.masks)} masks")
        console.print(f"  SfM input: {len(rec.images)} images, digest {rec.images_sha256[:12]}")
        if not rec.extraction.real_execution:
            console.print("[yellow]frame extraction was substituted[/]")

    run_guarded(go)


@video_app.command("rig")
def video_rig(
    out: Path = typer.Argument(..., help="rig_config.json"),
    n_yaw: int = typer.Option(6),
    fov_deg: float = typer.Option(90.0),
    size: int = typer.Option(1200),
) -> None:
    """Write COLMAP rig_config.json for a 360 ring crop.

    For hand-run COLMAP. The Phase 3 path derives the rig from the frame set's crop records
    instead, so that the configuration describes the crops that exist rather than the ones a
    spec says should.
    """
    from minegs.ingest.common.equirect import RingCropSpec
    from minegs.ingest.video.rig import write_rig_config

    run_guarded(
        lambda: console.print(
            f"wrote {write_rig_config(RingCropSpec(n_yaw=n_yaw, fov_deg=fov_deg, width=size, height=size), out)}"
        )
    )


@video_app.command("sfm")
def video_sfm(
    frameset_dir: Path = typer.Argument(..., help="a frame set written by `video frameset`"),
    out_dir: Path = typer.Argument(..., help="where the reconstruction and its record go"),
    backend: str = typer.Option("colmap"),
    mapper: str = typer.Option("global", help="global | incremental"),
    matcher: str = typer.Option("sequential", help="sequential | exhaustive | vocab_tree"),
    fix_intrinsics: bool = typer.Option(False),
    vocab_tree: Path | None = typer.Option(None, help="enables loop detection"),
    use_gpu: bool = typer.Option(True, "--gpu/--no-gpu"),
    component: str | None = typer.Option(
        None, help="which reconstruction to use when COLMAP produced several"
    ),
    dry_run: bool = typer.Option(False),
    overwrite: bool = typer.Option(False),
) -> None:
    """Run COLMAP (>= 4.0) over a frame set and record what it reconstructed.

    The result is in SFM_INTERNAL: its own coordinates, arbitrary in scale. It becomes metric
    only through `minegs eval register`, which measures the transform and records what it
    measured it against. There is no flag here that substitutes the reconstruction.
    """
    from minegs.ingest.video.models import check_frameset, load_frameset
    from minegs.ingest.video.sfm import SfMOptions, get_sfm_backend
    from minegs.ingest.video.sfm.run import run_sfm

    def go() -> None:
        rec_fs, fs_dir = load_frameset(frameset_dir)
        check_frameset(rec_fs, fs_dir)
        opts = SfMOptions(
            fix_intrinsics=fix_intrinsics,
            matcher=matcher,  # type: ignore[arg-type]
            vocab_tree=vocab_tree,
            use_gpu=use_gpu,
            masks_dir=rec_fs.mask_root(fs_dir),
        )
        if dry_run:
            be = get_sfm_backend(backend, mapper)
            for c in be.commands(rec_fs.image_root(fs_dir), out_dir, opts):
                console.print(" ".join(c))
            return
        rec, root = run_sfm(
            fs_dir,
            out_dir,
            backend=backend,
            mapper=mapper,
            options=opts,
            component=component,
            overwrite=overwrite,
        )
        console.print(f"sfm [bold]{rec.sfm_id}[/] in {root}")
        console.print(
            f"  {rec.registered_images} images, {rec.points} points, "
            f"component {rec.selected_component} of {len(rec.components)}"
        )
        console.print(f"  frame: [bold]{rec.frame}[/] ({rec.metric_state})")
        console.print(f"  real SfM execution: {rec.real_sfm_execution}")

    run_guarded(go)
