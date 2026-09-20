from __future__ import annotations

from pathlib import Path

import typer

from minegs.cli._common import console, dump_json, run_guarded

app = typer.Typer(no_args_is_help=True)


def _load_dataset_and_centerline(dataset_dir: Path):
    from minegs.core.centerline import Centerline
    from minegs.core.errors import ContractError
    from minegs.core.manifest import Manifest

    m = Manifest.load_dataset(dataset_dir, strict_layout=False)
    if m.centerline is None:
        raise ContractError("dataset has no centerline (needed for sections/volume/holdout ranges)")
    cl = Centerline.from_csv(
        dataset_dir / m.centerline.file, m.centerline.frame, m.centerline.source
    )
    if m.centerline.frame == "LOCAL_METRIC":
        cl = cl.transformed(m.T_tls_from_local, "TLS_GLOBAL")
    return m, cl


def _to_tls(pc, m):
    if pc.frame == "TLS_GLOBAL":
        return pc
    if pc.frame in ("LOCAL_METRIC", "UNKNOWN"):
        if pc.frame == "UNKNOWN":
            console.print("[yellow]PLY has no frame comment; assuming LOCAL_METRIC[/]")
        return pc.transformed(m.T_tls_from_local, "TLS_GLOBAL")
    from minegs.core.errors import ContractError

    raise ContractError(
        f"cannot evaluate a cloud in frame {pc.frame}; adapters must export LOCAL_METRIC (§3)"
    )


SURFACE_REQUIRED = (
    "Geometry accuracy requires a surface artifact. Gaussian centres are not surfaces. "
    "Build one with `minegs eval surface-depth <depth_dir> <dataset_dir> --run-dir <run_dir> "
    "--out <surface_dir>` and pass <surface_dir> here, or pass --diagnostic for non-claim "
    "numbers from a raw PLY."
)


def _unverified_depth(rec) -> str:
    return (
        f"surface {rec.surface_id} was built from depth maps minegs did not render "
        f"(depth_source={rec.depth_source}), so it cannot carry a geometry_accuracy claim. "
        f"Nothing ties those maps to run {rec.run_id}: the same directory paired with any "
        "succeeded run of this dataset would produce the same artifact. Rendering metric depth "
        "from a trained run is Phase 1B (docs/ROADMAP.md). Pass --diagnostic for non-claim "
        "numbers."
    )


def _resolve_pred(pred: Path, dataset_dir: Path, m, diagnostic: bool):
    """Resolve the geometry input to (points, SurfaceRecord | None).

    A surface artifact is what separates "these points came off a reconstruction" from "these
    points are where the optimiser put its Gaussians" (§16) — and ``check_surface`` verifies
    the points against the record rather than taking the record's word for them, so wrapping a
    Gaussian PLY in a hand-written surface.json does not get past this. Being a surface is
    still not sufficient: a claim also needs depth this project rendered (§1A).
    """
    from minegs.core.errors import ContractError
    from minegs.core.pointcloud import read_ply
    from minegs.core.provenance import sha256_tree
    from minegs.eval.surface.depth import rederive_depth_source
    from minegs.eval.surface.models import check_surface, find_surface, load_surface
    from minegs.train.runner.base import DATASET_HASH_PATTERNS

    if find_surface(pred) is not None:
        rec, points = load_surface(pred)
        pc = check_surface(
            rec, points, m.dataset_id, sha256_tree(dataset_dir, DATASET_HASH_PATTERNS)
        )
        console.print(
            f"surface [bold]{rec.surface_id}[/] ({rec.method}, depth {rec.depth_source}, "
            f"{rec.point_count} points from {rec.depth_map_count} depth maps, run {rec.run_id})"
        )
        if rec.supports_accuracy_claim:
            # `depth_source` is derived from evidence once, when the surface is fused, and then
            # read back forever. `check_surface` re-reads the points and checks their digest; it
            # has never re-checked this field, so a hand-written surface.json over any PLY --
            # `raw/tls_full.ply` itself, in the audit that found this -- claimed geometry
            # accuracy of 0.0 mm against the cloud it had been copied from. So the value is
            # re-derived here rather than read, and what comes back is what is used.
            stale = rederive_depth_source(rec, dataset_dir)
            if stale is not None:
                if not diagnostic:
                    raise ContractError(stale)
                console.print(f"[yellow]warning: {stale}[/]")
                rec.depth_source = "external_unverified"
        if not rec.supports_accuracy_claim:
            if not diagnostic:
                raise ContractError(_unverified_depth(rec))
            console.print(
                "[yellow]warning: this surface's depth maps are unverified external input; "
                "these numbers are diagnostic, not a validated geometry claim[/]"
            )
        return pc, rec
    if not diagnostic:
        raise ContractError(SURFACE_REQUIRED)
    if not pred.is_file():
        raise ContractError(f"{pred}: not a surface artifact (no surface.json) and not a PLY file")
    console.print(
        "[yellow]warning: raw PLY accepted only as diagnostic; this is not a validated "
        "surface artifact[/]"
    )
    return read_ply(pred), None


@app.command("render-depth")
def render_depth(
    run_dir: Path = typer.Argument(..., help="a succeeded run to render depth from"),
    dataset_dir: Path = typer.Argument(...),
    out: Path | None = typer.Option(
        None, "--out", help="depth directory (default: <run_dir>/depth)"
    ),
    min_alpha: float | None = typer.Option(
        None,
        "--min-alpha",
        help="pixels whose ray accumulates less opacity than this have no range and are "
        "written NaN (default 0.5)",
    ),
) -> None:
    """Render metric depth per dataset view from a trained run (§1.7 Phase 1B). GPU only.

    Writes ``<image stem>.npy`` plus a ``depth_manifest.json`` naming the run, the dataset, the
    checkpoint and a digest per map. That manifest is what lets `eval surface-depth` build a
    claim-capable surface; depth from anywhere else stays diagnostic-only.
    """
    from minegs.eval.surface.render import render_depths

    def go() -> None:
        manifest, depth_dir = render_depths(run_dir, dataset_dir, out, min_alpha)
        console.print(
            f"depth [bold]{manifest.manifest_id}[/]: {len(manifest.depths)} views from "
            f"{manifest.renderer['name']} on run {manifest.run_id}"
        )
        cover = sum(d.valid_ratio for d in manifest.depths) / len(manifest.depths)
        near = min(d.min_m for d in manifest.depths if d.min_m is not None)
        far = max(d.max_m for d in manifest.depths if d.max_m is not None)
        console.print(f"  ranges {near:.2f}-{far:.2f} m, mean coverage {cover * 100:.1f}%")
        console.print(f"  wrote {depth_dir}")

    run_guarded(go)


@app.command("surface-depth")
def surface_depth(
    depth_dir: Path = typer.Argument(
        ..., help="<image stem>.npy depth maps, metres along camera z"
    ),
    dataset_dir: Path = typer.Argument(...),
    run_dir: Path = typer.Option(
        ..., "--run-dir", help="the succeeded run these depth maps were rendered from"
    ),
    out: Path | None = typer.Option(
        None, "--out", help="surface directory (default: <run_dir>/surface/depth_v001)"
    ),
    stride: int = typer.Option(2, help="pixel stride when sampling each depth map"),
    max_depth: float | None = typer.Option(
        None, "--max-depth", help="drop samples further than this from the camera (m)"
    ),
) -> None:
    """Fuse depth maps into a LOCAL_METRIC surface artifact (§1.7).

    The artifact records which run and dataset the samples came from; it claims no accuracy.
    Depth maps themselves are rendered elsewhere — `render_depths` is Phase 1B.
    """
    from minegs.eval.surface.depth import build_depth_surface

    def go() -> None:
        target = out or (run_dir / "surface" / "depth_v001")
        rec, surface_dir = build_depth_surface(
            depth_dir, dataset_dir, run_dir, target, stride, max_depth
        )
        span = rec.span_m or []
        console.print(
            f"surface [bold]{rec.surface_id}[/]: {rec.point_count} points from "
            f"{rec.depth_map_count} depth maps (stride {stride})"
        )
        if span:
            console.print(f"  LOCAL_METRIC span {span[0]:.2f} x {span[1]:.2f} x {span[2]:.2f} m")
        console.print(f"  wrote {surface_dir}")

    run_guarded(go)


@app.command()
def protocol(
    dataset_dir: Path = typer.Argument(...), as_json: bool = typer.Option(False, "--json")
) -> None:
    """Which protocol(s) and claims this dataset supports (§5)."""
    from minegs.core.manifest import Manifest
    from minegs.eval.protocol import judge

    def go() -> None:
        j = judge(Manifest.load_dataset(dataset_dir, strict_layout=False))
        if as_json:
            dump_json(j, None)
            return
        console.print(f"protocols: [bold]{[p.value for p in j.protocols]}[/]")
        console.print(f"claims:    {[c.value for c in j.claims]}")
        for r in j.reasons:
            console.print(f"  [dim]{r}[/]")
        for r in j.refusals:
            console.print(f"  [red]refused:[/] {r}")

    run_guarded(go)


@app.command()
def register(
    source_ply: Path = typer.Argument(..., help="SfM cloud (arbitrary frame)"),
    target_ply: Path = typer.Argument(..., help="TLS cloud (TLS_GLOBAL)"),
    targets: Path | None = typer.Option(None, help="id,x,y,z CSV in the source frame"),
    targets_tls: Path | None = typer.Option(None, help="id,x,y,z CSV in TLS_GLOBAL"),
    icp_max_dist_m: float = typer.Option(0.5),
    icp_iters: int = typer.Option(50),
    out: Path | None = typer.Option(
        None, help="registration JSON (paste into manifest.registration)"
    ),
) -> None:
    """Sim(3) initial alignment from targets -> SE(3) ICP -> diagnostics (§7)."""
    from minegs.core.frames import SE3
    from minegs.core.pointcloud import read_ply
    from minegs.eval.register import align_correspondences, diagnose, icp_point_to_point
    from minegs.eval.register.initial_alignment import load_targets_csv, match_by_id

    def go() -> None:
        src, tgt = read_ply(source_ply), read_ply(target_ply)
        if targets and targets_tls:
            a, b, ids = match_by_id(load_targets_csv(targets), load_targets_csv(targets_tls))
            T0, inl = align_correspondences(a, b, with_scale=True)
            console.print(
                f"initial Sim3 from {int(inl.sum())}/{len(ids)} targets: scale={T0.s:.5f}"
            )
        else:
            console.print(
                "[yellow]no targets given: assuming source is already metric and roughly aligned[/]"
            )
            from minegs.core.frames import Sim3

            T0 = Sim3.identity()
        scaled = T0.apply(src.xyz)
        res = icp_point_to_point(
            scaled, tgt.xyz, SE3.identity(), max_dist_m=icp_max_dist_m, max_iters=icp_iters
        )
        T = res.T @ T0
        diag = diagnose(T, src.xyz, tgt.xyz, inlier_m=icp_max_dist_m / 5, method="sim3+icp")
        console.print(
            f"rmse={diag.rmse_m:.4f} m  inliers={diag.inlier_ratio:.3f}  scale={diag.scale:.5f}  icp_iters={res.iterations}"
        )
        dump_json(diag.to_manifest(), out)

    run_guarded(go)


@app.command()
def geometry(
    pred: Path = typer.Argument(
        ...,
        help="surface artifact directory from `eval surface-depth`; a raw PLY only "
        "with --diagnostic",
    ),
    dataset_dir: Path = typer.Argument(...),
    tls_ply: Path = typer.Option(..., help="reference TLS cloud (TLS_GLOBAL, raw/)"),
    holdout_only: bool = typer.Option(
        True, help="restrict to the manifest's holdout chainage ranges"
    ),
    max_dist_m: float = typer.Option(1.0),
    diagnostic: bool = typer.Option(
        False, help="Allow numbers without a geometry claim (reconstruction runs)"
    ),
    out: Path | None = typer.Option(None),
) -> None:
    """Bidirectional accuracy / completeness / Chamfer against TLS (§11). Refuses claims the manifest cannot support (§5)."""
    import numpy as np

    from minegs.core.errors import ContractError, ProtocolViolation
    from minegs.core.pointcloud import read_ply
    from minegs.eval.geometry import compare_clouds
    from minegs.eval.protocol import Claim, judge

    def go() -> None:
        m, cl = _load_dataset_and_centerline(dataset_dir)
        j = judge(m)
        claim = Claim.GEOMETRY_ACCURACY
        if not j.allows(claim):
            if not diagnostic:
                raise ProtocolViolation(
                    f"{m.dataset_id}: protocol {j.primary.value} cannot claim geometry_accuracy ({'; '.join(j.refusals) or 'no holdout'}); pass --diagnostic for non-claim numbers"
                )
            claim = Claim.GEOMETRY_DIAGNOSTIC
        points, surface = _resolve_pred(pred, dataset_dir, m, diagnostic)
        if surface is None or not surface.supports_accuracy_claim:
            # Neither a raw PLY nor a surface fused from external depth carries an accuracy
            # claim, whatever the manifest would allow: in the first case nothing establishes
            # that these points sample the tunnel wall, in the second nothing ties the depth
            # to the run (§18, §1A). _resolve_pred has already refused unless --diagnostic.
            claim = Claim.GEOMETRY_DIAGNOSTIC
        if claim is Claim.GEOMETRY_ACCURACY and not (holdout_only and j.holdout_ranges_m):
            # geometry_accuracy is defined as accuracy *on holdout TLS* (Claim docstring), and
            # the manifest only grants it because those ranges were excluded from
            # initialisation. Measuring over the whole cloud instead measures the model against
            # the geometry it was initialised and trained on, which is the leak the protocol
            # gate exists to prevent -- so --no-holdout-only reports a number, not a claim.
            console.print(
                "[yellow]warning: --no-holdout-only measures the training chainage too, so "
                "these numbers are fit to the data the run saw; reporting as diagnostic[/]"
            )
            claim = Claim.GEOMETRY_DIAGNOSTIC
        pred_pc = _to_tls(points, m)
        ref = read_ply(tls_ply)
        if ref.frame != "TLS_GLOBAL":
            # The predicted side is refused for exactly this in _to_tls; the reference side used
            # to get a console line that never reached the JSON. Comparing a LOCAL_METRIC cloud
            # against a TLS_GLOBAL one measures the distance between two coordinate systems, and
            # `dataset/init_points.ply` sits one directory from `raw/tls_full.ply`, so the
            # mix-up is an ordinary one. UNKNOWN counts as wrong: a cloud that never declared
            # its frame has not been established to be in this one.
            wrong_frame = (
                f"{tls_ply}: reference cloud is in frame {ref.frame}, not TLS_GLOBAL. Geometry "
                "is compared in TLS_GLOBAL, so this would measure the offset between two "
                "coordinate systems rather than between two surfaces. Re-export it with its "
                "frame declared, or pass --diagnostic for non-claim numbers."
            )
            if not diagnostic:
                raise ContractError(wrong_frame)
            console.print(f"[yellow]{wrong_frame}[/]")
        pxyz, rxyz = pred_pc.xyz, ref.xyz
        rng = None
        if holdout_only and j.holdout_ranges_m:
            sp, _ = cl.project(pxyz)
            sr, _ = cl.project(rxyz)
            mp = np.zeros(len(sp), bool)
            mr = np.zeros(len(sr), bool)
            for lo, hi in j.holdout_ranges_m:
                mp |= (sp >= lo) & (sp <= hi)
                mr |= (sr >= lo) & (sr <= hi)
            pxyz, rxyz = pxyz[mp], rxyz[mr]
            rng = (min(r[0] for r in j.holdout_ranges_m), max(r[1] for r in j.holdout_ranges_m))
        # An accuracy/completeness pair over an empty cloud is not a number, it is a missing
        # input: the comparison would come back all-NaN (or, until this check existed, as a
        # bare KeyError from compare_clouds with no message at all). Name which side emptied
        # and what emptied it, because the usual cause is a reference that does not cover the
        # holdout chainage -- a different TLS epoch, or a different chainage origin.
        empty = [n for n, a in (("prediction", pxyz), ("TLS reference", rxyz)) if len(a) == 0]
        if empty:
            where = f" inside the holdout chainage {rng[0]}-{rng[1]} m" if rng else ""
            raise ContractError(
                f"nothing to compare: the {' and the '.join(empty)} has no points{where}. "
                f"Check that {tls_ply} covers this dataset's chainage and shares its origin."
            )
        rep = compare_clouds(pxyz, rxyz, max_dist_m)
        rep.chainage_range_m = rng
        rep.claim = claim.value
        console.print(
            f"\\[{claim.value}] acc median={rep.accuracy.median_m * 1000:.1f}mm p95={rep.accuracy.p95_m * 1000:.1f}mm  comp median={rep.completeness.median_m * 1000:.1f}mm p95={rep.completeness.p95_m * 1000:.1f}mm  chamfer={rep.chamfer_m * 1000:.1f}mm"
        )
        dump_json(rep, out)

    run_guarded(go)


RAW_SECTIONS_WARNING = (
    "these sections were cut from a raw point cloud, not from a verified surface artifact. "
    "They are recorded as such and can never support a volume_accuracy claim, whatever the "
    "dataset protocol allows. For a claim-capable series, render depth with "
    "`minegs eval render-depth`, fuse it with `minegs eval surface-depth`, and section the "
    "surface directory."
)


def _unverified_sections(rec) -> str:
    src = rec.source
    if src.kind != "surface":
        return (
            f"sections {rec.section_id} were cut from a raw point cloud "
            f"({src.point_path}), so they carry no evidence about any reconstruction: the same "
            "cloud paired with any dataset would produce the same series. A volume_accuracy "
            "claim needs sections cut from a surface minegs rendered and fused (§1C). "
            "Pass --diagnostic for non-claim numbers."
        )
    return (
        f"sections {rec.section_id} were cut from surface {src.surface_id}, whose depth maps "
        f"minegs did not render (depth_source={src.depth_source}). Nothing ties those maps to "
        f"run {src.run_id}, so the surface cannot carry a geometry claim and neither can a "
        "volume integrated from it (§1A, §1C). Pass --diagnostic for non-claim numbers."
    )


BARE_SERIES = (
    "this is a bare section series with no provenance: it records no dataset, no surface and "
    "no run, so nothing can be checked about where its areas came from. A volume_accuracy "
    "claim needs a section artifact from `minegs eval sections` over a verified surface (§1C). "
    "Pass --diagnostic for non-claim numbers."
)


@app.command()
def sections(
    pred: Path = typer.Argument(
        ...,
        help="surface artifact directory from `eval surface-depth`; a raw PLY is accepted but "
        "is recorded as diagnostic-only",
    ),
    dataset_dir: Path = typer.Argument(...),
    interval_m: float = typer.Option(1.0),
    thickness_m: float = typer.Option(0.1),
    angle_bins: int = typer.Option(180),
    start_m: float | None = typer.Option(None),
    end_m: float | None = typer.Option(None),
    out: Path | None = typer.Option(None),
) -> None:
    """Cross-section areas A(s) along the centerline, as a section artifact (§11).

    The artifact records which dataset, which surface, which run and which reference axis the
    series was cut from. That provenance is what `eval volume` interrogates before it grants a
    volume_accuracy claim; a raw PLY still sections fine and is recorded as what it is.
    """
    import numpy as np

    from minegs.eval.sections import build_section_record, section_source

    def go() -> None:
        m, cl = _load_dataset_and_centerline(dataset_dir)
        points, surface = _resolve_pred(pred, dataset_dir, m, diagnostic=True)
        pc = _to_tls(points, m)
        src = section_source(surface, pred)
        rec = build_section_record(
            pc.xyz,
            src,
            dataset_dir,
            m,
            cl,
            interval_m=interval_m,
            thickness_m=thickness_m,
            angle_bins=angle_bins,
            start_m=start_m,
            end_m=end_m,
        )
        ser = rec.series
        console.print(
            f"sections [bold]{rec.section_id}[/] ({src.kind}"
            + (f", depth {src.depth_source}, run {src.run_id}" if src.kind == "surface" else "")
            + f"): valid {ser.valid_count()}/{len(ser.sections)}  "
            f"mean A = {float(np.nanmean(ser.areas())):.3f} m²"
        )
        if src.kind == "raw_cloud":
            console.print(f"[yellow]warning: {RAW_SECTIONS_WARNING}[/]")
        elif not src.supports_accuracy_claim:
            console.print(
                f"[yellow]warning: depth_source={src.depth_source}; these sections cannot "
                "support a volume_accuracy claim[/]"
            )
        dump_json(rec, out)

    run_guarded(go)


def _incomplete_coverage(cov, ranges) -> str:
    asked = ", ".join(f"{lo:g}-{hi:g} m" for lo, hi in ranges)
    unobserved = cov.requested_length_m - cov.covered_length_m
    head = (
        f"volume_accuracy is a claim about the whole declared holdout {asked}, and "
        f"{unobserved:.2f} m of {cov.requested_length_m:.2f} m of it "
        f"({(1 - cov.coverage_fraction) * 100:.1f}%) has no observed sections: "
        f"{cov.describe_gaps()}. "
        "The volume over a gap is not measured, and this project does not interpolate it or "
        "accept a coverage threshold it has not validated. Re-section over the holdout "
        "(`--start-m`/`--end-m`) once the reconstruction covers it"
    )
    if cov.covered_length_m > 0:
        return (
            head + ", or pass --diagnostic for the partial volume with this coverage "
            "reported alongside it."
        )
    # No two consecutive observed sections anywhere in the holdout, so there is no partial
    # volume over it to offer -- promising one and then exiting on "nothing to integrate" was
    # a remedy that did not work. The honest alternative is a different span, which is what
    # --no-holdout-only asks for, and it is a diagnostic by construction.
    return (
        head + ". There is no partial volume over the holdout either: it holds "
        f"{cov.valid_section_count} observed station(s) and a volume needs two consecutive "
        "ones, so --diagnostic has nothing to compute there. Use --no-holdout-only for a "
        "diagnostic volume over the span these sections do cover."
    )


@app.command()
def volume(
    sections_json: Path = typer.Argument(
        ..., help="section artifact from `eval sections` (a bare series is diagnostic-only)"
    ),
    dataset_dir: Path = typer.Argument(...),
    design_radius_m: float | None = typer.Option(
        None, help="circular design profile for over/underbreak"
    ),
    holdout_only: bool = typer.Option(
        True, help="restrict the integration to the manifest's holdout chainage ranges"
    ),
    diagnostic: bool = typer.Option(False),
    out: Path | None = typer.Option(None),
) -> None:
    """∫A(s)ds from a section artifact (+ design comparison) (§11).

    A ``volume_accuracy`` claim needs all of: a dataset whose protocol grants it, a section
    artifact belonging to *this* dataset and axis, sections cut from a surface minegs rendered,
    integration restricted to the declared holdout, and complete coverage of it. Anything else
    is refused, or computed as ``geometry_diagnostic`` with --diagnostic.
    """
    from minegs.core.errors import ContractError, ProtocolViolation
    from minegs.core.provenance import sha256_tree
    from minegs.eval.protocol import Claim, judge
    from minegs.eval.sections import (
        check_claim_evidence,
        check_section_record,
        load_section_input,
        reference_axis_of,
    )
    from minegs.eval.volume import compare_to_design, integrate_sections, plan_integration
    from minegs.train.runner.base import DATASET_HASH_PATTERNS

    def go() -> None:
        m, cl = _load_dataset_and_centerline(dataset_dir)
        j = judge(m)
        rec, ser = load_section_input(sections_json)
        if rec is not None:
            # Unconditional, --diagnostic included: a series whose provenance names another
            # dataset, another axis or a surface that has since changed is not a weaker number,
            # it is a different tunnel (§1C).
            check_section_record(
                rec,
                m.dataset_id,
                sha256_tree(dataset_dir, DATASET_HASH_PATTERNS),
                dataset_dir,
                m,
                cl,
            )
            console.print(
                f"sections [bold]{rec.section_id}[/] ({rec.source.kind}"
                + (
                    f", depth {rec.source.depth_source}, surface {rec.source.surface_id}, "
                    f"run {rec.source.run_id}"
                    if rec.source.kind == "surface"
                    else ""
                )
                + f", {len(ser.sections)} stations along {rec.reference_axis})"
            )

        claim = Claim.VOLUME_ACCURACY
        refusal: tuple[type[Exception], str] | None = None
        if not j.allows(claim):
            refusal = (
                ProtocolViolation,
                f"{m.dataset_id}: protocol {j.primary.value} cannot claim volume_accuracy "
                f"({'; '.join(j.refusals) or 'no chainage holdout declared'}); "
                "pass --diagnostic for non-claim numbers",
            )
        elif not j.holdout_ranges_m:
            refusal = (
                ProtocolViolation,
                f"{m.dataset_id}: volume_accuracy is defined on the declared geometry holdout, "
                "and this manifest declares no chainage range to evaluate over",
            )
        elif rec is None:
            refusal = (ContractError, f"{sections_json}: {BARE_SERIES}")
        elif not rec.supports_accuracy_claim:
            refusal = (ContractError, _unverified_sections(rec))
        if refusal is not None:
            if not diagnostic:
                raise refusal[0](refusal[1])
            console.print(f"[yellow]diagnostic: {refusal[1]}[/]")
            claim = Claim.GEOMETRY_DIAGNOSTIC

        if claim is Claim.VOLUME_ACCURACY and not holdout_only:
            # Same downgrade `eval geometry` makes, for the same reason: volume_accuracy is
            # defined over the holdout (Claim docstring), and the manifest grants it only
            # because those ranges were kept out of initialisation. Integrating the whole drift
            # measures the run against the geometry it was fitted to, so it reports a number.
            console.print(
                "[yellow]warning: --no-holdout-only integrates the training chainage too, so "
                "this volume is fit to the data the run saw; reporting as diagnostic[/]"
            )
            claim = Claim.GEOMETRY_DIAGNOSTIC

        evidence = None
        if claim is Claim.VOLUME_ACCURACY:
            # The identity checks above tie the record to this dataset, this axis and this
            # surface. None of them says the *areas* came from that surface -- the series lives
            # inside the record, so an edited area, or an invalid station flipped to a plausible
            # number to close a gap, satisfies every one of them. A claim re-derives instead,
            # and comes back with the one coverage number the station grid cannot flatter.
            evidence = check_claim_evidence(rec, m, cl, dataset_dir, j.holdout_ranges_m)
            if evidence.refusal is not None:
                if not diagnostic:
                    raise ContractError(evidence.refusal)
                console.print(f"[yellow]diagnostic: {evidence.refusal}[/]")
                claim = Claim.GEOMETRY_DIAGNOSTIC

        axis = reference_axis_of(m) if m.centerline else "unknown"
        # On the claim path the integration is restricted to the holdout ranges, because that is
        # what volume_accuracy is defined as (Claim docstring) and what the manifest excluded
        # from initialisation. Mixing the training chainage in would measure the run against the
        # geometry it was fitted to.
        ranges = list(j.holdout_ranges_m) if claim is Claim.VOLUME_ACCURACY else None
        if claim is Claim.VOLUME_ACCURACY:
            # Decided before integrating, not after: a holdout with no two consecutive observed
            # sections has no volume to report at all, and it should reach the coverage refusal
            # (which says what is missing) rather than a bare "nothing to integrate".
            segments, coverage = plan_integration(ser, ranges)
            if not coverage.complete:
                reason = _incomplete_coverage(coverage, ranges)
                # --diagnostic buys the partial volume over the holdout -- but only when there
                # is one. With no integrable pair in it the downgrade would fall through to
                # `integrate_sections` and leave as a bare "nothing to integrate", so the
                # refusal carries the remedy that works instead.
                if not diagnostic or not segments:
                    raise ContractError(reason)
                console.print(f"[yellow]diagnostic: {reason}[/]")
                # The holdout stays: the refusal above offers "the partial volume with this
                # coverage reported alongside it", and quietly swapping in the whole drift
                # would answer a different question than the one it just described.
                claim = Claim.GEOMETRY_DIAGNOSTIC
        rep = integrate_sections(ser, axis, ranges=ranges)
        rep.claim = claim.value
        if evidence is not None:
            rep.max_point_gap_m = evidence.max_point_gap_m
        if rec is not None:
            rep.section_id = rec.section_id
            rep.source = rec.source
            rep.section_parameters = dict(rec.parameters)
        cov = rep.coverage
        console.print(
            f"\\[{claim.value}] V = {rep.volume_m3:.2f} m³ over "
            f"{rep.start_chainage_m}-{rep.end_chainage_m} m "
            f"({rep.valid_section_count} valid / {rep.missing_section_count} missing sections)"
        )
        console.print(
            f"  integrated {cov.covered_length_m:.2f} of {cov.requested_length_m:.2f} m "
            f"({cov.coverage_fraction * 100:.1f}%) in {len(rep.segments)} segment(s); "
            f"gaps: {cov.describe_gaps()}"
        )
        # Coverage is about the station grid, which the caller chose. This says how much of the
        # integrated span the slabs actually looked at, so a coarse grid cannot read as a dense
        # measurement. Reported, not gated — see CoverageReport.sampled_fraction.
        console.print(
            f"  sampled {cov.sampled_length_m:.2f} m of that ({cov.sampled_fraction * 100:.1f}%) "
            f"in {ser.interval_m:g} m stations of {ser.thickness_m:g} m slabs; "
            f"{cov.interpolated_bin_fraction * 100:.1f}% of wall bins interpolated"
        )
        if rep.max_point_gap_m is not None:
            console.print(
                f"  longest span with no reconstructed point: {rep.max_point_gap_m:.2f} m"
            )
        result = {"volume": rep}
        if design_radius_m:
            dc = compare_to_design(ser, design_radius_m, ranges=ranges)
            console.print(
                f"overbreak {dc.overbreak_m3:.2f} m³  underbreak {dc.underbreak_m3:.2f} m³ "
                f"(design A={dc.design_area_m2:.2f} m², same {len(dc.overbreak_segments_m3)} "
                "segment(s))"
            )
            dc.claim = claim.value
            result["design"] = dc
        dump_json(result, out)

    run_guarded(go)


@app.command()
def change(
    a_json: Path = typer.Argument(...),
    b_json: Path = typer.Argument(...),
    epoch_a: str = typer.Option("ep1"),
    epoch_b: str = typer.Option("ep2"),
    out: Path | None = typer.Option(None),
) -> None:
    """ΔA(s), ΔV between two epochs' section series over the common chainage (§11).

    Diagnostic only: a validated change_volume claim needs the Phase 7 epoch-pair protocol.
    """
    from minegs.core.errors import ContractError
    from minegs.eval.change import diff_sections
    from minegs.eval.sections import load_section_input

    def go() -> None:
        # Either input shape: a section artifact or a bare pre-1C series. Neither is checked
        # against a dataset here, because this command names no dataset -- which is part of why
        # its result can never be more than diagnostic.
        rec_a, a = load_section_input(a_json)
        rec_b, b = load_section_input(b_json)
        # The one thing that *can* be compared without a dataset: chainage is only the same
        # quantity in both epochs if both were cut along the same axis. Subtracting areas
        # indexed on two different polylines is not a change, it is a coordinate difference.
        axes = {r.reference_axis for r in (rec_a, rec_b) if r is not None}
        if len(axes) > 1:
            raise ContractError(
                f"these section series were cut along different reference axes ({sorted(axes)}); "
                "their chainages do not refer to the same stations, so differencing them "
                "measures the axes, not the change"
            )
        rep = diff_sections(a, b, epoch_a, epoch_b, axes.pop() if axes else "centerline")
        console.print(
            f"\\[{rep.claim}] ΔV = {rep.delta_volume_m3:+.2f} m³ over "
            f"{rep.start_chainage_m}-{rep.end_chainage_m} m"
        )
        console.print(
            "[yellow]diagnostic only:[/] epoch comparability (frames, scale basis, reference "
            "axis, leak-free ranges) is not verified; the epoch-pair protocol is Phase 7"
        )
        dump_json(rep, out)

    run_guarded(go)


@app.command()
def render(
    renders_dir: Path = typer.Argument(..., help="<name>.png rendered for test images"),
    dataset_dir: Path = typer.Argument(...),
    lpips: bool = typer.Option(False),
    out: Path | None = typer.Option(None),
) -> None:
    """PSNR / SSIM (/ LPIPS) on the manifest's test groups (§5 novel_view)."""
    import numpy as np
    from PIL import Image

    from minegs.core.manifest import Manifest
    from minegs.eval.protocol import Claim, require
    from minegs.eval.render.metrics import evaluate_pairs

    def go() -> None:
        m = Manifest.load_dataset(dataset_dir, strict_layout=False)
        require(m, Claim.RENDER_QUALITY)
        pairs = {}
        for name in m.test_images():
            r = renders_dir / name
            g = dataset_dir / "images" / name
            if r.exists() and g.exists():
                pairs[name] = (
                    np.asarray(Image.open(r).convert("RGB")),
                    np.asarray(Image.open(g).convert("RGB")),
                )
        rep = evaluate_pairs(pairs, m.split.test_groups, with_lpips=lpips)
        console.print(
            f"n={rep.n_images} PSNR={rep.psnr:.2f} SSIM={rep.ssim:.4f}"
            + (f" LPIPS={rep.lpips:.4f}" if rep.lpips is not None else "")
        )
        dump_json(rep, out)

    run_guarded(go)
