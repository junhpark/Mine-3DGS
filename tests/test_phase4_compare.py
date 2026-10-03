"""Phase 4 C3 — baseline against advanced: refuse first, compare second, claim nothing.

The two "runs" here are records and sections over one synthetic tunnel, the advanced one cut
from a slightly different cloud. No training happened; the point is what the comparison will
and will not put side by side, and that what it reports is what was measured.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from minegs.core.errors import ContractError
from minegs.core.manifest import Manifest
from minegs.core.pointcloud import read_ply
from minegs.core.provenance import ProvenanceRecord, sha256_file, sha256_tree
from minegs.eval.compare import RunInputs, compare_runs
from minegs.eval.geometry.evaluate import load_dataset_and_centerline
from minegs.eval.sections.build import build_section_record
from minegs.eval.sections.models import SectionSource
from minegs.eval.volume.paired import compare_to_reference
from minegs.train.runner.base import DATASET_HASH_PATTERNS, RunRecord, RunStatus

from phase4_scene import tls_scene

CUT = {"interval_m": 1.0, "thickness_m": 0.5, "angle_bins": 36}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("p4cmp")
    sc = tls_scene(root / "scene")
    ds = sc.dataset_dir
    m, cl = load_dataset_and_centerline(ds)
    cloud = read_ply(sc.cloud).xyz
    ref = build_section_record(
        cloud,
        SectionSource(
            kind="raw_cloud", point_sha256=sha256_file(sc.cloud), point_path=str(sc.cloud)
        ),
        ds,
        m,
        cl,
        **CUT,
    )
    return type(
        "W",
        (),
        {
            "root": root,
            "ds": ds,
            "m": m,
            "cl": cl,
            "cloud": cloud,
            "ref": ref,
            "cloud_path": sc.cloud,
        },
    )


def _run(world, name: str, profile: str, **over) -> Path:
    run_dir = world.root / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = RunRecord(
        run_id=name,
        dataset_id=over.pop("dataset_id", world.m.dataset_id),
        dataset_hash=over.pop("dataset_hash", sha256_tree(world.ds, DATASET_HASH_PATTERNS)),
        backend={"name": "gsplat", "version": "1.5.3"},
        profile={
            "name": profile,
            "data_factor": 4 if profile == "light" else 2,
            "requests": {"appearance_embedding": profile != "light"},
            "backend_args": {"strategy": "default" if profile == "light" else "mcmc"},
        },
        runner="local",
        status=over.pop("status", RunStatus.SUCCEEDED),
        T_local_from_internal=over.pop("T", np.eye(4).tolist()),
        train_seconds=over.pop("train_seconds", 100.0),
        peak_gpu_memory_gb=over.pop("mem", None),
        gaussian_count=over.pop("gaussians", 1000),
        trainer_config={
            "strategy": "DefaultStrategy" if profile == "light" else "MCMCStrategy",
            "app_opt": profile != "light",
        },
        provenance=ProvenanceRecord(),
        **over,
    )
    rec.save(run_dir / "run.json")
    return run_dir


def _sections(world, run_id: str, scale: float, **cut):
    """Predicted sections of `run_id`: the scanned wall pulled toward the axis by `scale`."""
    s, _ = world.cl.project(world.cloud)
    axis = world.cl.point_at(s)
    pts = axis + (world.cloud - axis) * scale
    src = SectionSource(
        kind="surface",
        surface_id=f"surface_{run_id}",
        run_id=run_id,
        depth_source="minegs_render",
        point_sha256="b" * 64,
        point_path=f"/nonexistent/{run_id}",
    )
    return build_section_record(pts, src, world.ds, world.m, world.cl, **{**CUT, **cut})


def _report(gpu: bool, renderer: bool) -> dict:
    return {
        "training": {"real_gpu_execution": gpu},
        "reconstruction": {"real_renderer_execution": renderer},
    }


def _inputs(world, run_dir, sections, **kw):
    return RunInputs(run_dir, sections, kw.pop("ref", world.ref), **kw)


@pytest.fixture(scope="module")
def pair(world):
    base = _run(world, "run_base", "light", mem=4.0)
    adv = _run(world, "run_adv", "heavy", mem=None, train_seconds=300.0)
    return (
        base,
        _sections(world, "run_base", 0.97),
        adv,
        _sections(world, "run_adv", 0.99),
    )


def test_two_runs_on_one_dataset_are_compared_over_what_both_observed(world, pair):
    base, sb, adv, sa = pair
    geo_b = {
        "max_dist_m": 1.0,
        "chainage_range_m": [38.0, 46.0],
        "frame": "TLS_GLOBAL",
        "claim": "geometry_accuracy",
        "accuracy": {"median": 0.05, "p95": 0.2},
        "completeness": {"median": 0.06, "p95": 0.3},
        "chamfer_m": 0.07,
    }
    rep = compare_runs(
        world.ds,
        _inputs(world, base, sb, geometry=geo_b, e2e_report=_report(False, False)),
        _inputs(world, adv, sa, geometry=None, e2e_report=_report(False, False)),
        comparison_id="c1",
    )
    assert rep.requested_intervals_m == [(38.0, 46.0)]
    assert rep.common_intervals_m and rep.common_length_m > 0
    # the numbers are compare_to_reference's own, on the common domain
    want_b = compare_to_reference(sb, world.ref, rep.common_intervals_m)
    want_a = compare_to_reference(sa, world.ref, rep.common_intervals_m)
    assert rep.baseline.metrics["volume"]["absolute_error_m3"] == want_b.volume.absolute_error_m3
    d = rep.differences["volume"]["absolute_error_m3"]
    assert d == pytest.approx(want_a.volume.absolute_error_m3 - want_b.volume.absolute_error_m3)
    assert rep.differences["sections"]["median_absolute_error_m2"] == pytest.approx(
        want_a.sections.median_absolute_error_m2 - want_b.sections.median_absolute_error_m2
    )
    # missing stays missing: one side has no geometry report, the other no peak memory
    assert rep.baseline.metrics["geometry"]["accuracy_median_m"] == 0.05
    assert rep.advanced.metrics["geometry"]["accuracy_median_m"] is None
    assert rep.differences["geometry"]["accuracy_median_m"] is None
    assert rep.advanced.metrics["memory"]["peak_gpu_memory_gb"] is None
    assert rep.differences["memory"]["peak_gpu_memory_gb"] is None
    assert rep.differences["runtime"]["train_seconds"] == 200.0
    # no verdict, no loss, G3 pending, and substituted runs are structural only
    body = rep.model_dump(mode="json")
    assert "No improvement is claimed" in body.pop("maturity_statement")
    text = json.dumps(body).lower()
    assert "improv" not in text and "better" not in text and "loss" not in text
    assert "verdict" not in text
    assert rep.g3_status == "PENDING" and rep.real_execution is False
    assert any("structural" in n for n in rep.notes)
    assert any("data_factor" in n for n in rep.notes) and any("preset" in n for n in rep.notes)


def test_real_only_when_both_sides_report_both_stages_real(world, pair):
    base, sb, adv, sa = pair
    real = _report(True, True)
    rep = compare_runs(
        world.ds,
        _inputs(world, base, sb, e2e_report=real),
        _inputs(world, adv, sa, e2e_report=real),
        comparison_id="c",
    )
    assert rep.real_execution is True
    for b_rep, a_rep in ((real, _report(True, False)), (real, None), (_report(False, True), real)):
        rep = compare_runs(
            world.ds,
            _inputs(world, base, sb, e2e_report=b_rep),
            _inputs(world, adv, sa, e2e_report=a_rep),
            comparison_id="c",
        )
        assert rep.real_execution is False


def test_runs_on_different_datasets_are_refused(world, pair):
    _, _, _, sa = pair
    base = _run(world, "run_other", "light", dataset_hash="0" * 64)
    with pytest.raises(ContractError, match="two experiments"):
        compare_runs(
            world.ds,
            _inputs(world, base, _sections(world, "run_other", 0.97)),
            _inputs(world, pair[2], sa),
            comparison_id="c",
        )


def test_a_run_outside_the_metric_frame_is_refused(world, pair):
    T = np.eye(4)
    T[:3, :3] *= 0.05
    base = _run(world, "run_scaled", "light", T=T.tolist())
    with pytest.raises(ContractError, match="non-identity"):
        compare_runs(
            world.ds,
            _inputs(world, base, _sections(world, "run_scaled", 0.97)),
            _inputs(world, pair[2], pair[3]),
            comparison_id="c",
        )


def test_a_failed_run_is_refused(world, pair):
    base = _run(world, "run_failed", "light", status=RunStatus.FAILED)
    with pytest.raises(ContractError, match="not succeeded"):
        compare_runs(
            world.ds,
            _inputs(world, base, _sections(world, "run_failed", 0.97)),
            _inputs(world, pair[2], pair[3]),
            comparison_id="c",
        )


def test_sections_of_another_run_are_refused(world, pair):
    base, _, adv, sa = pair
    with pytest.raises(ContractError, match="not this run's sections"):
        compare_runs(world.ds, _inputs(world, base, sa), _inputs(world, adv, sa), comparison_id="c")


def test_the_same_run_twice_is_refused(world, pair):
    base, sb, _, _ = pair
    with pytest.raises(ContractError, match="same run"):
        compare_runs(
            world.ds, _inputs(world, base, sb), _inputs(world, base, sb), comparison_id="c"
        )


def test_different_grids_are_refused(world, pair):
    base, sb, adv, _ = pair
    coarse_pred = _sections(world, "run_adv", 0.99, interval_m=2.0)
    coarse_ref = build_section_record(
        world.cloud, world.ref.source, world.ds, world.m, world.cl, **{**CUT, "interval_m": 2.0}
    )
    with pytest.raises(ContractError, match="different grids"):
        compare_runs(
            world.ds,
            _inputs(world, base, sb),
            _inputs(world, adv, coarse_pred, ref=coarse_ref),
            comparison_id="c",
        )


def test_different_references_are_refused(world, pair):
    base, sb, adv, sa = pair
    other = world.ref.model_copy(deep=True)
    other.source = SectionSource(
        kind="raw_cloud", point_sha256="e" * 64, point_path="/elsewhere.ply"
    )
    with pytest.raises(ContractError, match="different references"):
        compare_runs(
            world.ds,
            _inputs(world, base, sb),
            _inputs(world, adv, sa, ref=other),
            comparison_id="c",
        )


def test_geometry_measured_differently_is_refused(world, pair):
    base, sb, adv, sa = pair
    g1 = {
        "max_dist_m": 1.0,
        "chainage_range_m": [38.0, 46.0],
        "frame": "TLS_GLOBAL",
        "claim": "geometry_accuracy",
    }
    g2 = {**g1, "max_dist_m": 0.5}
    with pytest.raises(ContractError, match="max_dist_m"):
        compare_runs(
            world.ds,
            _inputs(world, base, sb, geometry=g1),
            _inputs(world, adv, sa, geometry=g2),
            comparison_id="c",
        )


def test_a_dataset_without_a_holdout_needs_explicit_ranges(world, tmp_path):
    import shutil

    ds = tmp_path / "ds"
    shutil.copytree(world.ds, ds)
    m = Manifest.load_dataset(ds)
    m.split.geometry_holdout = None
    m.initialization.excluded_chainage_ranges_m = []
    m.save_dataset(ds)
    m, cl = load_dataset_and_centerline(ds)
    w = type("W", (), {"root": tmp_path, "ds": ds, "m": m, "cl": cl, "cloud": world.cloud})
    w.ref = build_section_record(world.cloud, world.ref.source, ds, m, cl, **CUT)
    base, adv = _run(w, "b", "light"), _run(w, "a", "heavy")
    sb, sa = _sections(w, "b", 0.97), _sections(w, "a", 0.99)
    with pytest.raises(ContractError, match="declares no geometry holdout"):
        compare_runs(ds, _inputs(w, base, sb), _inputs(w, adv, sa), comparison_id="c")
    rep = compare_runs(
        ds, _inputs(w, base, sb), _inputs(w, adv, sa), comparison_id="c", ranges=[(10.0, 20.0)]
    )
    assert rep.requested_intervals_m == [(10.0, 20.0)] and rep.protocol["holdout_ranges_m"] == []
    assert "diagnostic" in rep.notes[0]


def test_cli_compare_runs_writes_the_comparison(world, pair, tmp_path):
    from minegs.cli.main import app
    from typer.testing import CliRunner

    base, sb, adv, sa = pair
    files = {}
    for name, rec in (("sb", sb), ("sa", sa), ("ref", world.ref)):
        files[name] = tmp_path / f"{name}.json"
        rec.save(files[name])
    r = CliRunner().invoke(
        app,
        [
            "eval",
            "compare-runs",
            str(world.ds),
            "--baseline-run",
            str(base),
            "--advanced-run",
            str(adv),
            "--baseline-sections",
            str(files["sb"]),
            "--advanced-sections",
            str(files["sa"]),
            "--baseline-reference-sections",
            str(files["ref"]),
            "--advanced-reference-sections",
            str(files["ref"]),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert r.exit_code == 0, r.output
    data = json.loads((tmp_path / "out" / "run_comparison.json").read_text())
    assert data["g3_status"] == "PENDING" and data["baseline"]["run_id"] == "run_base"
