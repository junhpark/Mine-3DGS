from __future__ import annotations

from pathlib import Path

import typer

from minegs.cli._common import console, dump_json, run_guarded

app = typer.Typer(no_args_is_help=True)


def _load_dataset_and_centerline(dataset_dir: Path):
    from minegs.eval.geometry.evaluate import load_dataset_and_centerline

    return load_dataset_and_centerline(dataset_dir)


def _to_tls(pc, m):
    from minegs.eval.geometry.evaluate import to_tls

    notes: list[str] = []
    out = to_tls(pc, m, notes)
    for n in notes:
        console.print(f"[yellow]{n}[/]")
    return out


def _resolve_pred(pred: Path, dataset_dir: Path, m, diagnostic: bool):
    """The geometry gate's own resolver (``minegs/eval/geometry/evaluate.py``), printed.

    One implementation, two callers: this command and the Phase 2 workflow. The notes it
    returns are warnings that used to be console-only, so they now reach a report as well.
    """
    from minegs.eval.geometry.evaluate import resolve_prediction

    resolved = resolve_prediction(pred, dataset_dir, m, diagnostic)
    for note in resolved.notes:
        console.print(f"[yellow]{note}[/]" if note.startswith("warning") else note)
    return resolved.points, resolved.surface


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
    sfm_dir: Path = typer.Argument(..., help="an SfM artifact from `ingest video sfm`"),
    out_dir: Path = typer.Argument(..., help="where the registration artifact is written"),
    basis: str = typer.Option(
        ...,
        help="known_target (independent metric evidence) | sim3_to_tls (coordinates read "
        "off the TLS reference)",
    ),
    targets: Path = typer.Option(..., help="id,x,y,z CSV in the reconstruction's own frame"),
    targets_tls: Path = typer.Option(..., help="id,x,y,z CSV in TLS_GLOBAL"),
    target_ranges_m: str | None = typer.Option(
        None, help="chainage the TLS-side targets came from, e.g. '0:20,60:80'"
    ),
    icp_target_ply: Path | None = typer.Option(
        None, help="TLS subset to refine against (TLS_GLOBAL)"
    ),
    icp_ranges_m: str | None = typer.Option(None, help="chainage the ICP target covers"),
    icp_whole_reference: bool = typer.Option(
        False, help="the ICP target is the whole reference: diagnostic only, never a claim"
    ),
    reference_ply: Path | None = typer.Option(None, help="cloud the residuals are measured on"),
    icp_max_dist_m: float = typer.Option(0.5),
    icp_iters: int = typer.Option(50),
    max_rmse_m: float | None = typer.Option(None, help="quality gate: maximum residual"),
    min_inlier_ratio: float | None = typer.Option(None, help="quality gate: minimum inliers"),
    min_correspondences: int | None = typer.Option(None, help="quality gate: minimum matches"),
    overwrite: bool = typer.Option(False),
) -> None:
    """Measure SFM_INTERNAL → TLS_GLOBAL and record what the measurement rests on (§7).

    The transform is a Sim(3): initial alignment from correspondences, optional rigid ICP
    refinement, composed in that order so the measured scale survives.

    What decides whether the result can carry a metric claim is not the scale's origin but the
    *support* — every piece of geometry that moved the transform, the ICP target included. An
    ICP against the whole reference is diagnostic however the targets were obtained, because
    nothing is then held out from it.
    """

    def go() -> None:
        from minegs.eval.register.run import register_sfm

        thresholds = {
            k: v
            for k, v in (
                ("max_rmse_m", max_rmse_m),
                ("min_inlier_ratio", min_inlier_ratio),
                ("min_correspondences", min_correspondences),
            )
            if v is not None
        }
        rec, root = register_sfm(
            sfm_dir,
            out_dir,
            basis=basis,  # type: ignore[arg-type]
            targets_sfm=targets,
            targets_tls=targets_tls,
            target_ranges_m=_ranges(target_ranges_m),
            icp_target_ply=icp_target_ply,
            icp_ranges_m=_ranges(icp_ranges_m),
            icp_target_is_whole_reference=icp_whole_reference,
            reference_ply=reference_ply,
            icp_max_dist_m=icp_max_dist_m,
            icp_iters=icp_iters,
            thresholds=thresholds or None,
            overwrite=overwrite,
        )
        d = rec.diagnostics
        console.print(f"registration [bold]{rec.registration_id}[/] in {root}")
        console.print(
            f"  scale={rec.scale:.5f}  rmse={d.rmse_m:.4f} m  inliers={d.inlier_ratio:.3f}  "
            f"n={d.n_source}"
        )
        console.print(f"  support: {rec.support_ranges_m}")
        console.print(f"  metric claim allowed: [bold]{rec.claim_allowed}[/]")
        for reason in rec.claim_refusals:
            console.print(f"  [yellow]no claim:[/] {reason}")

    run_guarded(go)


def _ranges(text: str | None) -> list[tuple[float, float]] | None:
    """``'0:20,60:80'`` -> [(0, 20), (60, 80)]. ``None`` stays None: unknown is not empty."""
    if text is None:
        return None
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition(":")
        out.append((float(lo), float(hi)))
    return out


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
    from minegs.eval.geometry.evaluate import evaluate_geometry

    def go() -> None:
        result = evaluate_geometry(pred, dataset_dir, tls_ply, holdout_only, max_dist_m, diagnostic)
        for note in result.notes:
            console.print(f"[yellow]{note}[/]" if note.startswith("warning") else note)
        rep = result.report
        console.print(
            f"\\[{rep.claim}] acc median={rep.accuracy.median_m * 1000:.1f}mm "
            f"p95={rep.accuracy.p95_m * 1000:.1f}mm  "
            f"comp median={rep.completeness.median_m * 1000:.1f}mm "
            f"p95={rep.completeness.p95_m * 1000:.1f}mm  chamfer={rep.chamfer_m * 1000:.1f}mm"
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


@app.command("compare-paths")
def compare_paths_cmd(
    tls_dataset: Path = typer.Argument(..., help="the TLS-assisted dataset"),
    image_dataset: Path = typer.Argument(..., help="the image/360-only dataset"),
    tls_sections: Path = typer.Option(..., help="TLS-assisted prediction sections"),
    tls_reference_sections: Path = typer.Option(..., help="reference sections in the TLS dataset"),
    image_sections: Path = typer.Option(..., help="image-only prediction sections"),
    image_reference_sections: Path = typer.Option(
        ..., help="reference sections in the image-only dataset"
    ),
    ranges: str | None = typer.Option(
        None, help="chainage to compare over; default is the holdout both datasets declare"
    ),
    tls_e2e_report: Path | None = typer.Option(None, help="e2e report of the TLS-assisted run"),
    image_e2e_report: Path | None = typer.Option(None, help="e2e report of the image-only run"),
    comparison_id: str = typer.Option("path-comparison"),
    out: Path | None = typer.Option(None, help="directory for path_comparison.json"),
) -> None:
    """TLS-assisted against image-only, over the domain both of them observed (§Phase 3).

    Both sides must have been cut on one grid and be evaluated over one declared holdout, and
    the numbers are integrated over the intersection of what each actually observed — what
    either side missed is reported, not quietly dropped. This is a comparison of two
    reconstruction paths; it is not a claim, and it upgrades neither dataset's protocol.
    """
    from minegs.core.errors import ContractError
    from minegs.eval.compare import (
        COMPARISON_FILE,
        compare_paths,
        path_context,
        require_same_holdout,
    )
    from minegs.eval.sections import check_section_record, load_section_input

    def load(path: Path, dataset_dir: Path, what: str):
        m, cl = _load_dataset_and_centerline(dataset_dir)
        rec, _ = load_section_input(path)
        if rec is None:
            raise ContractError(
                f"{path} is a bare section series, so nothing ties it to {what}. A path "
                "comparison is between two datasets; a series that names neither cannot be in it."
            )
        from minegs.core.provenance import sha256_tree
        from minegs.train.runner.base import DATASET_HASH_PATTERNS

        check_section_record(
            rec, m.dataset_id, sha256_tree(dataset_dir, DATASET_HASH_PATTERNS), dataset_dir, m, cl
        )
        return rec

    def go() -> None:
        tls_ctx = path_context(tls_dataset, e2e_report=tls_e2e_report)
        img_ctx = path_context(image_dataset, e2e_report=image_e2e_report)
        if tls_ctx["dataset_id"] == img_ctx["dataset_id"]:
            raise ContractError(
                f"both sides name dataset {tls_ctx['dataset_id']!r}; a path comparison is "
                "between two reconstructions of one tunnel, not a dataset against itself"
            )
        explicit = _ranges(ranges)
        domain = explicit if explicit else require_same_holdout(tls_ctx, img_ctx)
        rep = compare_paths(
            tls_pred=load(tls_sections, tls_dataset, "the TLS-assisted dataset"),
            tls_ref=load(tls_reference_sections, tls_dataset, "the TLS-assisted dataset"),
            image_pred=load(image_sections, image_dataset, "the image-only dataset"),
            image_ref=load(image_reference_sections, image_dataset, "the image-only dataset"),
            ranges=domain,
            tls_context=tls_ctx,
            image_context=img_ctx,
            comparison_id=comparison_id,
        )
        if explicit:
            rep.notes.insert(
                0,
                "the comparison domain was given on the command line, not taken from a declared "
                "holdout; these numbers are diagnostic",
            )
        console.print(
            f"comparison [bold]{rep.comparison_id}[/] over {rep.common_length_m:.2f} m "
            f"common of {sum(hi - lo for lo, hi in rep.requested_intervals_m):.2f} m requested"
        )
        for label, r in (("tls_assisted", rep.tls_assisted), ("image_only", rep.image_only)):
            v = r.paired.volume
            console.print(
                f"  {label:<12} {r.dataset_id}: V={v.predicted_volume_m3}, "
                f"|ΔV|={v.absolute_error_m3}, "
                f"median |ΔA|={r.paired.sections.median_absolute_error_m2}"
            )
        console.print(
            f"  ΔV(image - tls) = {rep.volume_difference_m3}  "
            f"Δmedian |ΔA| = {rep.section_median_difference_m2}"
        )
        console.print(f"  real_execution={rep.real_execution}")
        for n in rep.notes:
            console.print(f"  [yellow]{n}[/]")
        console.print(f"[bold]{rep.maturity_statement}[/]")
        dump_json(rep, (out / COMPARISON_FILE) if out is not None else None)

    run_guarded(go)
