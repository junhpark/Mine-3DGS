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
    pred_ply: Path = typer.Argument(..., help="surface samples from the run (LOCAL_METRIC)"),
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

    from minegs.core.errors import ProtocolViolation
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
        pred = _to_tls(read_ply(pred_ply), m)
        ref = read_ply(tls_ply)
        if ref.frame != "TLS_GLOBAL":
            console.print(f"[yellow]TLS reference frame is {ref.frame}; expected TLS_GLOBAL[/]")
        pxyz, rxyz = pred.xyz, ref.xyz
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
        rep = compare_clouds(pxyz, rxyz, max_dist_m)
        rep.chainage_range_m = rng
        rep.claim = claim.value
        console.print(
            f"\\[{claim.value}] acc median={rep.accuracy.median_m * 1000:.1f}mm p95={rep.accuracy.p95_m * 1000:.1f}mm  comp median={rep.completeness.median_m * 1000:.1f}mm p95={rep.completeness.p95_m * 1000:.1f}mm  chamfer={rep.chamfer_m * 1000:.1f}mm"
        )
        dump_json(rep, out)

    run_guarded(go)


@app.command()
def sections(
    ply: Path = typer.Argument(...),
    dataset_dir: Path = typer.Argument(...),
    interval_m: float = typer.Option(1.0),
    thickness_m: float = typer.Option(0.1),
    angle_bins: int = typer.Option(180),
    start_m: float | None = typer.Option(None),
    end_m: float | None = typer.Option(None),
    out: Path | None = typer.Option(None),
) -> None:
    """Cross-section areas A(s) along the centerline (§11)."""
    from minegs.core.pointcloud import read_ply
    from minegs.eval.sections import extract_sections

    def go() -> None:
        m, cl = _load_dataset_and_centerline(dataset_dir)
        pc = _to_tls(read_ply(ply), m)
        ser = extract_sections(pc.xyz, cl, interval_m, thickness_m, angle_bins, start_m, end_m)
        console.print(
            f"sections valid {ser.valid_count()}/{len(ser.sections)}  mean A = {float(__import__('numpy').nanmean(ser.areas())):.3f} m²"
        )
        dump_json(ser, out)

    run_guarded(go)


@app.command()
def volume(
    sections_json: Path = typer.Argument(...),
    dataset_dir: Path = typer.Argument(...),
    design_radius_m: float | None = typer.Option(
        None, help="circular design profile for over/underbreak"
    ),
    diagnostic: bool = typer.Option(False),
    out: Path | None = typer.Option(None),
) -> None:
    """∫A(s)ds from a sections JSON (+ design comparison) (§11)."""
    import json

    from minegs.core.errors import ProtocolViolation
    from minegs.core.manifest import Manifest
    from minegs.eval.protocol import Claim, judge
    from minegs.eval.sections.sections import SectionSeries
    from minegs.eval.volume import compare_to_design, integrate_sections

    def go() -> None:
        m = Manifest.load_dataset(dataset_dir, strict_layout=False)
        j = judge(m)
        claim = Claim.VOLUME_ACCURACY
        if not j.allows(claim):
            if not diagnostic:
                raise ProtocolViolation(
                    f"{m.dataset_id}: protocol {j.primary.value} cannot claim volume_accuracy; pass --diagnostic"
                )
            claim = Claim.GEOMETRY_DIAGNOSTIC
        ser = SectionSeries.model_validate(json.loads(sections_json.read_text()))
        axis = (
            f"centerline:{m.centerline.source}:{m.centerline.file}" if m.centerline else "unknown"
        )
        rep = integrate_sections(ser, axis)
        rep.claim = claim.value
        console.print(
            f"\\[{claim.value}] V = {rep.volume_m3:.2f} m³ over {rep.start_chainage_m}-{rep.end_chainage_m} m ({rep.valid_section_count} valid / {rep.missing_section_count} missing sections)"
        )
        result = {"volume": rep}
        if design_radius_m:
            dc = compare_to_design(ser, design_radius_m)
            console.print(
                f"overbreak {dc.overbreak_m3:.2f} m³  underbreak {dc.underbreak_m3:.2f} m³ (design A={dc.design_area_m2:.2f} m²)"
            )
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
    import json

    from minegs.eval.change import diff_sections
    from minegs.eval.sections.sections import SectionSeries

    def go() -> None:
        a = SectionSeries.model_validate(json.loads(a_json.read_text()))
        b = SectionSeries.model_validate(json.loads(b_json.read_text()))
        rep = diff_sections(a, b, epoch_a, epoch_b, "centerline")
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
