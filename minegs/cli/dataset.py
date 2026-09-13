from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from minegs.cli._common import console, dump_json, run_guarded

app = typer.Typer(no_args_is_help=True)


@app.command()
def synthetic(
    root: Path = typer.Argument(..., help="Output root: <root>/raw, <root>/dataset"),
    length_m: float = typer.Option(120.0, help="Tunnel length"),
    radius_m: float = typer.Option(2.5),
    n_stations: int | None = typer.Option(None, help="Overrides station spacing"),
    station_spacing_m: float = typer.Option(15.0),
    image_size: int = typer.Option(96),
    seed: int = typer.Option(0),
) -> None:
    """Generate the synthetic tunnel dataset (Phase 0A gate)."""
    from minegs.core.synthetic import SyntheticSpec, generate

    spacing = length_m / n_stations if n_stations else station_spacing_m
    spec = SyntheticSpec(
        length_m=length_m,
        radius_m=radius_m,
        station_spacing_m=spacing,
        image_size=image_size,
        seed=seed,
    )

    def go() -> None:
        r = generate(root, spec)
        console.print(
            f"dataset [bold]{r.manifest.dataset_id}[/] -> {r.dataset_dir}  images={len(r.manifest.all_images())} tls_points={len(r.tls_points_global)}"
        )

    run_guarded(go)


@app.command()
def validate(
    dataset_dir: Path = typer.Argument(...),
    strict: bool = typer.Option(False, help="Fail on consistency issues too"),
) -> None:
    """Check the on-disk layout + manifest schema, then report consistency issues (§4)."""
    from minegs.core.errors import ContractError
    from minegs.core.manifest import Manifest

    def go() -> None:
        m = Manifest.load_dataset(dataset_dir)
        issues = m.consistency_issues()
        console.print(
            f"[green]layout + schema OK[/]  dataset_id={m.dataset_id} schema={m.schema_version} groups={len(m.capture_groups)} images={len(m.all_images())}"
        )
        for i in issues:
            console.print(f"  [yellow]issue:[/] {i}")
        if issues and strict:
            raise ContractError(f"{len(issues)} consistency issue(s)")
        if not issues:
            console.print("[green]no consistency issues[/]")

    run_guarded(go)


@app.command()
def info(
    dataset_dir: Path = typer.Argument(...), as_json: bool = typer.Option(False, "--json")
) -> None:
    """Summarise a dataset: frames, groups, split, chunks, protocol."""
    from minegs.core.manifest import Manifest
    from minegs.eval.protocol import judge

    def go() -> None:
        m = Manifest.load_dataset(dataset_dir, strict_layout=False)
        j = judge(m)
        if as_json:
            dump_json({"manifest": m, "judgement": j}, None)
            return
        t = Table(title=f"{m.dataset_id}  (schema {m.schema_version}, source={m.source})")
        t.add_column("group")
        t.add_column("type")
        t.add_column("#img", justify="right")
        t.add_column("chainage")
        t.add_column("role")
        for gid, g in m.capture_groups.items():
            role = (
                "test"
                if gid in m.split.test_groups
                else "train"
                if gid in m.split.train_groups
                else "-"
            )
            if gid in m.initialization.groups:
                role += "+init"
            t.add_row(gid, g.type, str(len(g.members)), str(g.span()), role)
        console.print(t)
        console.print(f"T_tls_from_local t = {m.T_tls_from_local.t.round(3).tolist()}   unit=m")
        if m.chunks:
            console.print(f"chunks: {[(c.id, c.range_m) for c in m.chunks.items]}")
        console.print(
            f"protocols: {[p.value for p in j.protocols]}   claims: {[c.value for c in j.claims]}"
        )
        for r in j.reasons:
            console.print(f"  [dim]{r}[/]")
        for r in j.refusals:
            console.print(f"  [red]refused:[/] {r}")

    run_guarded(go)


@app.command()
def migrate(
    manifest: Path = typer.Argument(...),
    out: Path | None = typer.Option(None, help="Default: in place"),
) -> None:
    """Migrate manifest.json to the current schema_version."""
    from minegs.core.manifest import Manifest

    def go() -> None:
        m = Manifest.load(manifest)
        p = m.save(out or manifest)
        console.print(f"schema_version -> {m.schema_version}: {p}")

    run_guarded(go)


@app.command()
def chunks(
    dataset_dir: Path = typer.Argument(...),
    length_m: float = typer.Option(80.0),
    overlap_m: float = typer.Option(15.0),
    write: bool = typer.Option(False, help="Write the plan into manifest.chunks"),
) -> None:
    """Plan chainage chunks along the dataset centerline (§10)."""
    from minegs.core.centerline import Centerline
    from minegs.core.chunking import assign_groups, plan_chunks
    from minegs.core.errors import ContractError
    from minegs.core.manifest import Manifest

    def go() -> None:
        m = Manifest.load_dataset(dataset_dir, strict_layout=False)
        if m.centerline is None:
            raise ContractError("manifest has no centerline; import one first")
        cl = Centerline.from_csv(
            dataset_dir / m.centerline.file, m.centerline.frame, m.centerline.source
        )
        plan = assign_groups(
            plan_chunks(cl.s_start, cl.s_end, length_m, overlap_m, cl), m.group_chainage()
        )
        for c in plan.items:
            console.print(f"{c.id}: {c.range_m}  groups={c.groups}")
        if write:
            m.chunks = plan
            m.save_dataset(dataset_dir)
            console.print("manifest.chunks updated")

    run_guarded(go)


@app.command("centerline-extract")
def centerline_extract(
    ply: Path = typer.Argument(...),
    out: Path = typer.Argument(...),
    bin_m: float = typer.Option(2.0),
) -> None:
    """Extract a centerline from a TLS cloud (fallback when no design line exists, §10)."""
    from minegs.core.centerline import Centerline
    from minegs.core.pointcloud import read_ply

    def go() -> None:
        pc = read_ply(ply)
        cl = Centerline.extract_from_points(
            pc.xyz, bin_m=bin_m, frame=pc.frame if pc.frame != "UNKNOWN" else "TLS_GLOBAL"
        )
        cl.to_csv(out)
        console.print(f"centerline length={cl.length:.1f} m vertices={len(cl.vertices)} -> {out}")

    run_guarded(go)


# ---------------------------------------------------------------- Phase 0C


@app.command("synthetic-staging")
def synthetic_staging(
    root: Path = typer.Argument(..., help="Output root: <root>/raw, <root>/staging"),
    length_m: float = typer.Option(90.0),
    station_spacing_m: float = typer.Option(15.0),
    image_size: int = typer.Option(96),
    image_mode: str = typer.Option("pinhole_cube", help="pinhole_cube | spherical"),
    seed: int = typer.Option(0),
) -> None:
    """Write a synthetic Phase 0B.3 staging tree (the Phase 0C G1 fixture)."""
    from minegs.core.errors import ContractError
    from minegs.core.synthetic_staging import StagingSpec, generate_staging

    def go() -> None:
        if image_mode not in ("pinhole_cube", "spherical"):
            raise ContractError("image_mode must be pinhole_cube or spherical")
        r = generate_staging(
            root,
            StagingSpec(
                length_m=length_m,
                station_spacing_m=station_spacing_m,
                image_size=image_size,
                image_mode=image_mode,  # type: ignore[arg-type]
                seed=seed,
            ),
        )
        console.print(
            f"staging -> {r.staging_dir}  stations={len(r.station_poses)} images={len(r.image_poses_cam)}"
        )

    run_guarded(go)


@app.command("calibrate-camera")
def calibrate_camera(
    staging: Path = typer.Argument(..., help="Phase 0B.3 staging tree"),
    out: Path = typer.Option(Path("camera_convention.json"), "--out"),
    stations: int = typer.Option(
        3, help="spatially separated stations to sample (>= 3; fewer is refused)"
    ),
    max_points: int = typer.Option(150_000),
    min_margin: float = typer.Option(5.0, help="required lead of the best convention (RGB units)"),
) -> None:
    """Golden gate step 1: measure the E57 pinhole axis convention against the TLS points."""
    from minegs.dataset.calibrate import calibrate_camera_convention
    from minegs.dataset.staging_input import load_staging

    def go() -> None:
        cal = calibrate_camera_convention(
            load_staging(staging), n_stations=stations, max_points=max_points, min_margin=min_margin
        )
        cal.save(out)
        label = cal.convention.label if cal.convention else "-"
        console.print(
            f"status=[bold]{cal.status}[/] convention={label} scoring={cal.scoring} "
            f"best={cal.best_score:.2f} runner_up={cal.runner_up_score} margin={cal.margin}"
        )
        for s in cal.stations:
            console.print(f"  {s.station_id}: best={s.best_label} margin={s.margin}")
        for n in cal.notes:
            console.print(f"  [yellow]{n}[/]")
        console.print(f"wrote {out}")
        if cal.status != "selected":
            from minegs.core.errors import ContractError

            raise ContractError(f"camera convention not selected ({cal.status}); see {out}")

    run_guarded(go)


@app.command("from-e57")
def from_e57(
    staging: Path = typer.Argument(..., help="Phase 0B.3 staging tree"),
    out: Path = typer.Argument(..., help="dataset/ directory to create"),
    config: Path = typer.Option(..., "--config", help="build config YAML/JSON (see --example)"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Materialise the staging tree into the dataset contract (Phase 0C)."""
    from minegs.dataset.build_config import load_build_config
    from minegs.dataset.materialize import build_dataset

    def go() -> None:
        cfg = load_build_config(config)
        r = build_dataset(staging, out, cfg, overwrite=overwrite, config_path=config)
        rep = r.report
        console.print(
            f"dataset [bold]{rep['dataset_id']}[/] -> {r.dataset_dir}  stations={rep['n_stations']} "
            f"images={rep['n_images']} init_points={rep['initialization']['points_out']}"
        )
        console.print(
            f"T_tls_from_local t={[round(v, 3) for v in rep['frames']['local_origin_tls']]}  "
            f"source_mode={rep['frames']['source_mode']}"
        )
        if rep["camera_convention"]:
            console.print(
                f"camera convention: {rep['camera_convention']['label']} ({rep['camera_convention']['source']})"
            )
        console.print(f"protocols={rep['protocols']} claims={rep['claims']}")
        for x in rep["refusals"]:
            console.print(f"  [red]refused:[/] {x}")

    run_guarded(go)


@app.command("build-config-example")
def build_config_example() -> None:
    """Print an example Phase 0C build config."""
    import yaml

    from minegs.dataset.build_config import example_config

    console.print(yaml.safe_dump(example_config(), sort_keys=False))


@app.command("golden-gate")
def golden_gate(
    dataset_dir: Path = typer.Argument(...),
    staging: Path = typer.Option(
        ..., "--staging", help="the staging tree the dataset was built from"
    ),
    out: Path = typer.Option(..., "--out", help="report directory"),
    stations: int = typer.Option(3),
    max_points: int = typer.Option(150_000),
) -> None:
    """Golden gate: reprojection overlays, numerical checks, report.json, Viser TLS sample."""
    from minegs.dataset.golden_gate import run_golden_gate

    def go() -> None:
        rep = run_golden_gate(
            dataset_dir,
            staging,
            out,
            n_stations=stations,
            max_points=max_points,
            raise_on_fail=False,
        )
        console.print(
            f"structural_result=[bold]{rep['structural_result']}[/]  "
            f"real_data_validation_status={rep['real_data_validation_status']}"
        )
        for p in rep["structural_problems"]:
            console.print(f"  [red]{p}[/]")
        for s in rep["stations"]:
            console.print(
                f"  {s['station_id']}: best_convention={s.get('best_convention')} margin={s.get('margin')}"
            )
        console.print(f"overlays: {len(rep['overlays'])} -> {out / 'overlays'}")
        console.print(f"viser: {rep['viser']}")
        if rep["structural_result"] != "pass":
            from minegs.core.errors import ContractError

            raise ContractError(
                f"golden gate structural_result=fail; diagnostics written to {out} "
                f"({len(rep['structural_problems'])} problem(s))"
            )

    run_guarded(go)
