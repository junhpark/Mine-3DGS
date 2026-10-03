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


def _report(run_id: str, gpu: bool, renderer: bool, **geometry) -> dict:
    """A Phase 2 report about `run_id` and its surface."""
    return {
        "training": {"run_id": run_id, "real_gpu_execution": gpu},
        "reconstruction": {"surface_id": f"surface_{run_id}", "real_renderer_execution": renderer},
        "geometry": geometry,
    }


#: What a run's own record says when a GPU ran the pinned upstream (runner.base.real_gpu_evidence).
def _gpu_runtime() -> dict:
    from minegs.train.backends.gsplat import PINNED_GSPLAT, UPSTREAM_TRAINER_SHA256

    return {
        "gpu_model": "NVIDIA RTX 6000",
        "torch_cuda_available": True,
        "gsplat": PINNED_GSPLAT,
        "trainer_sha256": UPSTREAM_TRAINER_SHA256,
    }


def _distance(median: float, p95: float) -> dict:
    return {
        "n": 100,
        "mean_m": median,
        "rmse_m": median,
        "median_m": median,
        "p90_m": p95,
        "p95_m": p95,
        "max_m": 1.0,
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
    # the shape `minegs eval geometry` writes (GeometryReport), not a hand-made one
    geo_b = {
        "max_dist_m": 1.0,
        "chainage_range_m": [38.0, 46.0],
        "frame": "TLS_GLOBAL",
        "claim": "geometry_accuracy",
        "accuracy": _distance(0.05, 0.2),
        "completeness": _distance(0.06, 0.3),
        "chamfer_m": 0.07,
    }
    rep = compare_runs(
        world.ds,
        _inputs(world, base, sb, geometry=geo_b, e2e_report=_report("run_base", False, False)),
        _inputs(world, adv, sa, geometry=None, e2e_report=_report("run_adv", False, False)),
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
    assert rep.baseline.metrics["geometry"]["completeness_p95_m"] == 0.3
    assert any("attributed to run_base by the caller" in n for n in rep.notes)
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


def test_real_only_when_both_runs_recorded_it_and_both_sides_report_it(world, pair):
    _, sb, _, sa = pair
    base = _run(world, "run_base_gpu", "light", runtime=_gpu_runtime())
    adv = _run(world, "run_adv_gpu", "heavy", runtime=_gpu_runtime())
    sb = _sections(world, "run_base_gpu", 0.97)
    sa = _sections(world, "run_adv_gpu", 0.99)
    rb, ra = _report("run_base_gpu", True, True), _report("run_adv_gpu", True, True)
    rep = compare_runs(
        world.ds,
        _inputs(world, base, sb, e2e_report=rb),
        _inputs(world, adv, sa, e2e_report=ra),
        comparison_id="c",
    )
    assert rep.real_execution is True
    for b_rep, a_rep in (
        (rb, _report("run_adv_gpu", True, False)),
        (rb, None),
        (_report("run_base_gpu", False, True), ra),
    ):
        rep = compare_runs(
            world.ds,
            _inputs(world, base, sb, e2e_report=b_rep),
            _inputs(world, adv, sa, e2e_report=a_rep),
            comparison_id="c",
        )
        assert rep.real_execution is False


def test_a_report_cannot_make_a_run_real_that_its_record_does_not_show(world, pair):
    """Phase 4 C4: the flags are the run's own evidence; a report can only lower them."""
    base, sb, adv, sa = pair  # records with no GPU and no pinned trainer recorded
    rep = compare_runs(
        world.ds,
        _inputs(world, base, sb, e2e_report=_report("run_base", True, True)),
        _inputs(world, adv, sa, e2e_report=_report("run_adv", True, True)),
        comparison_id="c",
    )
    assert rep.real_execution is False
    assert any("not counted as real" in n for n in rep.notes)
    assert any("structural" in n for n in rep.notes)
    # a trainer that is not the pinned upstream file is not real gsplat training either
    other = {**_gpu_runtime(), "trainer_sha256": "0" * 64}
    b2 = _run(world, "run_b_other", "light", runtime=other)
    rep = compare_runs(
        world.ds,
        _inputs(
            world,
            b2,
            _sections(world, "run_b_other", 0.97),
            e2e_report=_report("run_b_other", True, True),
        ),
        _inputs(world, adv, sa, e2e_report=_report("run_adv", True, True)),
        comparison_id="c",
    )
    assert rep.real_execution is False


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda r: r["training"].__setitem__("run_id", "some_other_run"), "not run_base"),
        (lambda r: r.__setitem__("dataset", {"dataset_hash": "f" * 64}), "another dataset"),
        (lambda r: r["reconstruction"].__setitem__("surface_id", "surface_x"), "surface_x"),
    ],
)
def test_an_e2e_report_about_something_else_is_refused(world, pair, edit, match):
    base, sb, adv, sa = pair
    r = _report("run_base", True, True)
    edit(r)
    with pytest.raises(ContractError, match=match):
        compare_runs(
            world.ds,
            _inputs(world, base, sb, e2e_report=r),
            _inputs(world, adv, sa),
            comparison_id="c",
        )


def test_geometry_is_checked_against_the_run_s_own_report(world, pair):
    base, sb, adv, sa = pair
    geo = {
        "max_dist_m": 1.0,
        "chainage_range_m": [38.0, 46.0],
        "accuracy": _distance(0.05, 0.2),
        "completeness": _distance(0.06, 0.3),
        "chamfer_m": 0.07,
    }
    recorded = {
        "accuracy_median_m": 0.05,
        "accuracy_p95_m": 0.2,
        "completeness_median_m": 0.06,
        "completeness_p95_m": 0.3,
        "chamfer_m": 0.07,
    }
    ok = compare_runs(
        world.ds,
        _inputs(
            world, base, sb, geometry=geo, e2e_report=_report("run_base", False, False, **recorded)
        ),
        _inputs(world, adv, sa, e2e_report=_report("run_adv", False, False, **recorded)),
        comparison_id="c",
    )
    # bound by the report: no "attributed by the caller" note, and the advanced side's
    # geometry comes from its own e2e report
    assert not any("attributed" in n and "geometry" in n for n in ok.notes)
    assert ok.advanced.metrics["geometry"]["chamfer_m"] == 0.07
    with pytest.raises(ContractError, match="disagrees with the geometry"):
        compare_runs(
            world.ds,
            _inputs(
                world,
                base,
                sb,
                geometry=geo,
                e2e_report=_report("run_base", False, False, **{**recorded, "chamfer_m": 0.5}),
            ),
            _inputs(world, adv, sa),
            comparison_id="c",
        )
    with pytest.raises(ContractError, match="not over a range of this comparison"):
        compare_runs(
            world.ds,
            _inputs(world, base, sb, geometry={**geo, "chainage_range_m": [0.0, 60.0]}),
            _inputs(world, adv, sa),
            comparison_id="c",
        )
    with pytest.raises(ContractError, match="not a GeometryReport"):
        compare_runs(
            world.ds,
            _inputs(world, base, sb, geometry={"accuracy": {"median": 0.05}}),
            _inputs(world, adv, sa),
            comparison_id="c",
        )


def test_render_reports_are_validated_and_marked_as_attributed(world, pair):
    base, sb, adv, sa = pair
    groups = list(world.m.split.test_groups)
    ren = {"test_groups": groups, "n_images": 3, "psnr": 20.0, "ssim": 0.7}
    rep = compare_runs(
        world.ds,
        _inputs(world, base, sb, render=ren),
        _inputs(world, adv, sa, render={**ren, "psnr": 21.0}),
        comparison_id="c",
    )
    assert rep.differences["render"]["psnr"] == 1.0 and rep.differences["render"]["lpips"] is None
    assert any("RenderReport, which names no run" in n for n in rep.notes)
    with pytest.raises(ContractError, match="test groups"):
        compare_runs(
            world.ds,
            _inputs(world, base, sb, render={**ren, "test_groups": ["nope"]}),
            _inputs(world, adv, sa, render={**ren, "test_groups": ["nope"]}),
            comparison_id="c",
        )


def test_a_run_s_own_prediction_is_not_a_reference(world, pair):
    base, sb, adv, sa = pair
    with pytest.raises(ContractError, match="not a scanned reference"):
        compare_runs(
            world.ds,
            _inputs(world, base, sb, ref=sa),
            _inputs(world, adv, sa, ref=sa),
            comparison_id="c",
        )


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


def test_depth_from_the_init_s_own_source_is_said_so(world, pair):
    """Phase 4 C4: separate artifacts are not independent information (AD-1)."""
    base, sb, _, _ = pair
    rel = {"shared_source": "tls_survey", "fraction_of_samples_at_init_points": 0.5}
    adv = _run(
        world,
        "run_adv_dsup",
        "heavy",
        depth_supervision={
            "supervision_id": "d",
            "artifact_sha256": "a" * 64,
            "init_relation": rel,
        },
    )
    rep = compare_runs(
        world.ds,
        _inputs(world, base, sb),
        _inputs(world, adv, _sections(world, "run_adv_dsup", 0.99)),
        comparison_id="c",
    )
    assert rep.advanced.configuration["depth_supervision"]["init_relation"] == rel
    assert any("same tls_survey as its initialisation; 50%" in n for n in rep.notes)
