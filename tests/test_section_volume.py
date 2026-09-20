"""Phase 1C — the section/volume evidence boundary and gap-safe integration (§11).

Two things are under test, and they are independent of each other.

**The boundary.** Phase 1A and 1B ended at ``surface``; ``eval volume`` still granted
``volume_accuracy`` on the strength of the dataset manifest alone, so sections cut from
``raw/tls_full.ply`` reached the same claim as sections cut from a reconstruction. Here the
chain runs to the end — a succeeded run, depth minegs rendered, a verified surface, a section
artifact — and every way of entering it further down is refused.

**The integration.** ``np.trapezoid`` over the valid stations draws one straight line across a
gap and reports the area under it as measured volume. Gaps are now integration boundaries, and
what was not integrated is reported as an interval.

Structural, as everything since Phase 1A has been. The renderer is substituted (``StandInRenderer``
from ``test_depth_render``), so these tests say nothing about whether any of the numbers are
*accurate* — only about what may and may not be called a claim.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from minegs.cli.main import app
from minegs.core.errors import ContractError
from minegs.eval.sections import SectionRecord, extract_sections, load_section_input
from minegs.eval.sections.sections import Section, SectionSeries
from minegs.eval.surface.depth import build_depth_surface
from minegs.eval.surface.render import render_depths
from minegs.eval.volume import compare_to_design, integrate_sections
from minegs.eval.volume.coverage import (
    CoverageReport,
    integration_segments,
    merge_intervals,
    plan_integration,
    summarise_coverage,
)
from minegs.ingest.common.colmap_io import read_model
from typer.testing import CliRunner

from test_depth_render import StandInRenderer, make_run

runner = CliRunner()

HOLDOUT = (20.0, 26.0)  # dataset_small's declared geometry holdout


# ---------------------------------------------------------------- the real chain, once


def _rendered_surface(ds: Path, root: Path, depth_m: float, run_id: str) -> Path:
    """run -> minegs-rendered depth -> verified surface, exactly as production builds it."""
    run_dir = root / "run"
    make_run(ds, run_dir, run_id=run_id)
    _, depth_dir = render_depths(
        run_dir, ds, root / "depth", renderer=StandInRenderer(depth_m=depth_m)
    )
    rec, surface_dir = build_depth_surface(depth_dir, ds, run_dir, root / "surface")
    assert rec.depth_source == "minegs_render"
    return surface_dir


def _external_surface(ds: Path, root: Path, depth_m: float = 7.0) -> Path:
    """A surface fused from bare ``.npy`` files: structurally fine, provenance-free (§1A)."""
    depth_dir = root / "depth"
    depth_dir.mkdir(parents=True)
    model = read_model(ds / "sparse" / "0")
    for im in model.images.values():
        cam = model.cameras[im.camera_id]
        np.save(
            depth_dir / (Path(im.name).stem + ".npy"),
            np.full((cam.height, cam.width), depth_m, np.float32),
        )
    run_dir = root / "run"
    make_run(ds, run_dir, run_id="run_external")
    rec, surface_dir = build_depth_surface(depth_dir, ds, run_dir, root / "surface")
    assert rec.depth_source == "external_unverified"
    return surface_dir


@pytest.fixture(scope="module")
def chain(dataset_small, tmp_path_factory):
    """Three surfaces over one dataset, differing only in what backs them.

    ``full`` and ``gappy`` are both ``minegs_render``; the constant render range is the only
    difference, and it decides how much of the tunnel the back-projected shells cover. At 7 m
    the coverage is continuous, at 4 m it leaves unobserved bands — including inside the
    holdout, which is what an incomplete claim looks like in practice.
    """
    root = tmp_path_factory.mktemp("p1c")
    ds = dataset_small.dataset_dir
    return SimpleNamespace(
        dataset_dir=ds,
        full=_rendered_surface(ds, root / "full", 7.0, "run_1c_full"),
        gappy=_rendered_surface(ds, root / "gappy", 4.0, "run_1c_gappy"),
        external=_external_surface(ds, root / "ext"),
        raw_ply=ds / "init_points.ply",
        root=root,
    )


def sections_argv(pred: Path, ds: Path, out: Path, **kw) -> list[str]:
    argv = ["eval", "sections", str(pred), str(ds), "--out", str(out)]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


def cut(pred: Path, ds: Path, out: Path, **kw) -> SectionRecord:
    r = runner.invoke(app, sections_argv(pred, ds, out, **kw))
    assert r.exit_code == 0, r.output
    return SectionRecord.load(out)


def volume_argv(sec: Path, ds: Path, out: Path | None = None, **flags) -> list[str]:
    argv = ["eval", "volume", str(sec), str(ds)]
    if out is not None:
        argv += ["--out", str(out)]
    if flags.get("diagnostic"):
        argv.append("--diagnostic")
    if flags.get("no_holdout_only"):
        argv.append("--no-holdout-only")
    if flags.get("design_radius_m") is not None:
        argv += ["--design-radius-m", str(flags["design_radius_m"])]
    return argv


def volume_json(out: Path) -> dict:
    return json.loads(out.read_text())["volume"]


def flat(r) -> str:
    """CLI output with rich's line wrapping undone, so a message can be matched whole."""
    return " ".join(r.output.split())


# ---------------------------------------------------------------- T1/T2: a raw point cloud


def test_t1_sections_from_a_raw_ply_cannot_claim_volume_accuracy(chain, tmp_path):
    """The dataset grants volume_accuracy; the *cloud* is what cannot carry it (§1C)."""
    sec = tmp_path / "raw.json"
    rec = cut(chain.raw_ply, chain.dataset_dir, sec, interval_m=1, thickness_m=0.5, angle_bins=72)
    assert rec.source.kind == "raw_cloud" and rec.source.surface_id is None
    assert not rec.supports_accuracy_claim

    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir))
    assert r.exit_code == 2, r.output
    assert "raw point cloud" in flat(r) and "--diagnostic" in flat(r)


def test_t2_the_same_raw_ply_still_computes_a_diagnostic_volume(chain, tmp_path):
    sec, out = tmp_path / "raw.json", tmp_path / "vol.json"
    cut(chain.raw_ply, chain.dataset_dir, sec, interval_m=1, thickness_m=0.5, angle_bins=72)

    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir, out, diagnostic=True))
    assert r.exit_code == 0, r.output
    vol = volume_json(out)
    assert vol["claim"] == "geometry_diagnostic"
    assert vol["volume_m3"] > 0 and vol["segments"]
    # the record travels into the report, so a reader of volume.json can see what it stands on
    assert vol["source"]["kind"] == "raw_cloud"


# ---------------------------------------------------------------- T3: unverified depth


def test_t3_sections_from_an_externally_fused_surface_cannot_claim(chain, tmp_path):
    """A surface is necessary, not sufficient: the depth behind it still has to be minegs'."""
    sec = tmp_path / "ext.json"
    rec = cut(chain.external, chain.dataset_dir, sec, interval_m=1, thickness_m=0.5, angle_bins=72)
    assert rec.source.kind == "surface" and rec.source.depth_source == "external_unverified"
    assert not rec.supports_accuracy_claim

    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir))
    assert r.exit_code == 2, r.output
    assert "external_unverified" in flat(r) and rec.source.surface_id in flat(r)


# ---------------------------------------------------------------- T4: the chain reaches volume


def test_t4_a_rendered_surface_with_complete_holdout_coverage_reaches_volume_accuracy(
    chain, tmp_path
):
    """run -> rendered depth -> verified surface -> sections -> volume_accuracy, end to end."""
    sec, out = tmp_path / "full.json", tmp_path / "vol.json"
    rec = cut(chain.full, chain.dataset_dir, sec, interval_m=1, thickness_m=0.5, angle_bins=72)
    assert rec.source.depth_source == "minegs_render" and rec.supports_accuracy_claim

    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir, out))
    assert r.exit_code == 0, r.output
    vol = volume_json(out)
    assert vol["claim"] == "volume_accuracy"
    cov = vol["coverage"]
    assert [tuple(i) for i in cov["requested_intervals_m"]] == [HOLDOUT]
    assert [tuple(i) for i in cov["integrated_intervals_m"]] == [HOLDOUT]
    assert cov["missing_intervals_m"] == [] and cov["coverage_fraction"] == pytest.approx(1.0)
    assert vol["source"]["run_id"] == "run_1c_full"
    assert vol["section_id"] == rec.section_id


# ---------------------------------------------------------------- T5: identity


def test_t5_a_record_from_another_dataset_is_refused_even_diagnostically(chain, tmp_path):
    """Wrong dataset is not a weaker number, it is a different tunnel — so no flag excuses it."""
    sec = tmp_path / "full.json"
    cut(chain.full, chain.dataset_dir, sec, interval_m=1, thickness_m=0.5, angle_bins=72)

    raw = json.loads(sec.read_text())
    raw["dataset_id"] = "some_other_dataset"
    sec.write_text(json.dumps(raw))
    for argv in (
        volume_argv(sec, chain.dataset_dir),
        volume_argv(sec, chain.dataset_dir, diagnostic=True),
    ):
        r = runner.invoke(app, argv)
        assert r.exit_code == 2, r.output
        assert "some_other_dataset" in flat(r)

    raw["dataset_id"] = json.loads((chain.dataset_dir / "manifest.json").read_text())["dataset_id"]
    raw["dataset_hash"] = "0" * 64
    sec.write_text(json.dumps(raw))
    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir, diagnostic=True))
    assert r.exit_code == 2 and "dataset_hash" in flat(r)


def test_a_record_whose_surface_changed_underneath_it_is_refused(chain, tmp_path):
    """The surface is re-verified while it is still on disk, not taken from the copy."""
    sec = tmp_path / "full.json"
    cut(chain.full, chain.dataset_dir, sec, interval_m=1, thickness_m=0.5, angle_bins=72)
    points = chain.full / "surface_points.ply"
    keep = points.read_bytes()
    try:
        points.write_bytes(keep[:-40])
        r = runner.invoke(app, volume_argv(sec, chain.dataset_dir, diagnostic=True))
        assert r.exit_code == 2, r.output
        assert "hashes to" in flat(r)
    finally:
        points.write_bytes(keep)


def test_a_record_whose_stations_were_edited_no_longer_matches_the_axis(chain, tmp_path):
    """The station grid is re-derived from the dataset's own centerline, not believed."""
    sec = tmp_path / "full.json"
    cut(chain.full, chain.dataset_dir, sec, interval_m=1, thickness_m=0.5, angle_bins=72)
    raw = json.loads(sec.read_text())
    raw["series"]["sections"] = raw["series"]["sections"][:10]
    sec.write_text(json.dumps(raw))

    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir, diagnostic=True))
    assert r.exit_code == 2, r.output
    assert "not cut along the axis this dataset now declares" in flat(r)


# ---------------------------------------------------------------- T6: a bare legacy series


def test_t6_a_bare_section_series_is_diagnostic_only(chain, tmp_path):
    """Pre-1C JSON still works everywhere it worked, and claims nothing."""
    sec, out = tmp_path / "bare.json", tmp_path / "vol.json"
    full = tmp_path / "full.json"
    cut(chain.full, chain.dataset_dir, full, interval_m=1, thickness_m=0.5, angle_bins=72)
    sec.write_text(json.dumps(json.loads(full.read_text())["series"]))
    record, series = load_section_input(sec)
    assert record is None and series.sections

    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir))
    assert r.exit_code == 2, r.output
    assert "bare section series" in flat(r)

    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir, out, diagnostic=True))
    assert r.exit_code == 0, r.output
    vol = volume_json(out)
    assert vol["claim"] == "geometry_diagnostic" and vol["section_id"] is None


def test_eval_change_reads_both_shapes(chain, tmp_path):
    """`eval change` predates the artifact and must keep accepting what it always did."""
    full, bare = tmp_path / "full.json", tmp_path / "bare.json"
    cut(chain.full, chain.dataset_dir, full, interval_m=2, thickness_m=0.5, angle_bins=72)
    bare.write_text(json.dumps(json.loads(full.read_text())["series"]))
    r = runner.invoke(app, ["eval", "change", str(full), str(bare)])
    assert r.exit_code == 0, r.output
    assert "geometry_diagnostic" in r.output


# ---------------------------------------------------------------- T7/T8: the holdout


def test_t7_a_claim_integrates_the_holdout_only(chain, tmp_path):
    """The series spans the whole drift; volume_accuracy is about the held-out part of it."""
    sec = tmp_path / "full.json"
    claim_out, diag_out = tmp_path / "claim.json", tmp_path / "diag.json"
    rec = cut(chain.full, chain.dataset_dir, sec, interval_m=1, thickness_m=0.5, angle_bins=72)
    assert rec.series.chainages().min() < HOLDOUT[0] and rec.series.chainages().max() > HOLDOUT[1]

    assert runner.invoke(app, volume_argv(sec, chain.dataset_dir, claim_out)).exit_code == 0
    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir, diag_out, no_holdout_only=True))
    assert r.exit_code == 0, r.output
    claim, diag = volume_json(claim_out), volume_json(diag_out)

    # asking for the whole drift is allowed and is not a claim, exactly as in `eval geometry`
    assert claim["claim"] == "volume_accuracy" and diag["claim"] == "geometry_diagnostic"
    assert "fit to the data the run saw" in flat(r)
    assert claim["start_chainage_m"] == pytest.approx(HOLDOUT[0])
    assert claim["end_chainage_m"] == pytest.approx(HOLDOUT[1])
    # the whole-drift number is the whole drift's: ~10x the span, so ~10x the volume
    assert diag["coverage"]["requested_length_m"] > 50
    assert diag["volume_m3"] > 5 * claim["volume_m3"]


@pytest.fixture(scope="module")
def two_ranges(dataset_small, tmp_path_factory):
    """A copy of the dataset declaring two disjoint holdout ranges.

    Only the manifest is edited: the claim gate reads the declared ranges, and the surface is
    rendered from the run rather than from ``init_points.ply``, so nothing downstream depends on
    the init exclusion these ranges now name. The point is the integration boundary, not the leak
    contract, which ``judge`` owns and ``test_eval`` covers.
    """
    root = tmp_path_factory.mktemp("p1c_2r")
    ds = root / "dataset"
    shutil.copytree(dataset_small.dataset_dir, ds)
    ranges = [[14.0, 18.0], [26.0, 30.0]]
    m = json.loads((ds / "manifest.json").read_text())
    m["split"]["geometry_holdout"]["chainage_ranges_m"] = ranges
    m["initialization"]["excluded_chainage_ranges_m"] = ranges
    (ds / "manifest.json").write_text(json.dumps(m, indent=2))
    return SimpleNamespace(
        dataset_dir=ds, surface=_rendered_surface(ds, root / "full", 7.0, "run_1c_2r")
    )


def test_t8_two_holdout_ranges_are_never_integrated_across(two_ranges, tmp_path):
    sec, out = tmp_path / "s.json", tmp_path / "v.json"
    cut(
        two_ranges.surface,
        two_ranges.dataset_dir,
        sec,
        interval_m=1,
        thickness_m=0.5,
        angle_bins=72,
    )
    r = runner.invoke(app, volume_argv(sec, two_ranges.dataset_dir, out))
    assert r.exit_code == 0, r.output
    vol = volume_json(out)

    assert vol["claim"] == "volume_accuracy"
    assert [tuple(i) for i in vol["coverage"]["integrated_intervals_m"]] == [
        (14.0, 18.0),
        (26.0, 30.0),
    ]
    assert len(vol["segments"]) == 2
    assert vol["volume_m3"] == pytest.approx(sum(g["volume_m3"] for g in vol["segments"]))
    # 8 m of holdout, not the 16 m from 14 to 30
    assert vol["coverage"]["requested_length_m"] == pytest.approx(8.0)
    assert vol["coverage"]["covered_length_m"] == pytest.approx(8.0)
    assert all(
        g["end_chainage_m"] - g["start_chainage_m"] == pytest.approx(4.0) for g in vol["segments"]
    )


# ---------------------------------------------------------------- T9/T10/T13: the integral


def _series(
    areas: list[float | None], interval_m: float = 1.0, angle_bins: int = 4
) -> SectionSeries:
    """A hand-made series: ``None`` is a station nothing was observed at."""
    sections = [
        Section(
            chainage_m=float(i) * interval_m,
            area_m2=a,
            valid=a is not None,
            n_points=0 if a is None else 100,
            empty_bins=angle_bins if a is None else 0,
            radii_m=[None] * angle_bins
            if a is None
            else [float(np.sqrt(2 * a / angle_bins / np.sin(2 * np.pi / angle_bins)))] * angle_bins,
            center=[float(i) * interval_m, 0.0, 0.0],
            tangent=[1.0, 0.0, 0.0],
        )
        for i, a in enumerate(areas)
    ]
    return SectionSeries(
        frame="TLS_GLOBAL",
        interval_m=interval_m,
        thickness_m=0.5,
        angle_bins=angle_bins,
        start_chainage_m=0.0,
        end_chainage_m=float(len(areas) - 1) * interval_m,
        sections=sections,
    )


def test_t9_a_gap_is_an_integration_boundary_not_a_straight_line():
    """areas 10, 10, -, 10, 10 at s = 0..4: two 10 m³ segments, not one 40 m³ trapezoid."""
    rep = integrate_sections(_series([10.0, 10.0, None, 10.0, 10.0]), "centerline:test")

    assert rep.volume_m3 == pytest.approx(20.0)
    assert [(g.start_chainage_m, g.end_chainage_m) for g in rep.segments] == [
        (0.0, 1.0),
        (3.0, 4.0),
    ]
    assert rep.volume_m3 == pytest.approx(sum(g.volume_m3 for g in rep.segments))


def test_t10_the_missing_interval_and_the_coverage_fraction_are_exact():
    rep = integrate_sections(_series([10.0, 10.0, None, 10.0, 10.0]), "centerline:test")
    cov = rep.coverage

    assert cov.requested_intervals_m == [(0.0, 4.0)] and cov.requested_length_m == pytest.approx(
        4.0
    )
    assert cov.integrated_intervals_m == [(0.0, 1.0), (3.0, 4.0)]
    assert cov.missing_intervals_m == [(1.0, 3.0)]
    assert cov.covered_length_m == pytest.approx(2.0)
    assert cov.coverage_fraction == pytest.approx(0.5)
    assert cov.missing_section_count == 1 and cov.missing_chainages_m == [2.0]
    assert not cov.complete


def test_an_isolated_valid_station_integrates_to_nothing_and_is_not_covered():
    """One observation has no neighbour to integrate towards; inventing half an interval is
    the imputation this whole module refuses."""
    rep = integrate_sections(_series([10.0, 10.0, None, 10.0, None, 10.0, 10.0]), "x")

    assert [(g.start_chainage_m, g.end_chainage_m) for g in rep.segments] == [
        (0.0, 1.0),
        (5.0, 6.0),
    ]
    assert rep.coverage.missing_intervals_m == [(1.0, 5.0)]
    assert rep.coverage.valid_section_count == 5  # the lone station at s=3 is still observed


def test_a_series_with_no_two_consecutive_observations_refuses_rather_than_returning_zero():
    with pytest.raises(ContractError, match="nothing to integrate"):
        integrate_sections(_series([10.0, None, 10.0, None, 10.0]), "x")


def test_duplicate_stations_are_refused():
    ser = _series([10.0, 10.0, 10.0])
    ser.sections[2].chainage_m = ser.sections[1].chainage_m
    with pytest.raises(ContractError, match="share chainage"):
        integrate_sections(ser, "x")


def test_a_non_finite_area_is_not_an_observation():
    """`valid` is the extractor's word; an inf area would still propagate into the total."""
    ser = _series([10.0, 10.0, 10.0, 10.0])
    ser.sections[1].area_m2 = float("inf")
    rep = integrate_sections(ser, "x")
    assert np.isfinite(rep.volume_m3)
    assert rep.coverage.missing_intervals_m == [(0.0, 2.0)]


def test_t13_overbreak_and_underbreak_are_integrated_over_the_same_segments():
    """Design comparison used the same trapezoid-across-gaps, and must not any more."""
    ser = _series([10.0, 10.0, None, 10.0, 10.0])
    design = float(np.sqrt(2 * 5.0 / 4 / np.sin(2 * np.pi / 4)))  # half the area of a section
    dc = compare_to_design(ser, design)

    assert dc.coverage.missing_intervals_m == [(1.0, 3.0)]
    assert len(dc.overbreak_segments_m3) == 2
    assert dc.overbreak_m3 == pytest.approx(sum(dc.overbreak_segments_m3))
    # every observed section has the same overbreak area, so the volume is that area times the
    # 2 m actually covered -- not times the 4 m the stations span
    per_section = dc.overbreak_m2[0]
    assert dc.overbreak_m3 == pytest.approx(2.0 * per_section)
    assert dc.overbreak_m3 < 4.0 * per_section
    assert dc.underbreak_m3 == pytest.approx(0.0, abs=1e-9)
    assert dc.chainage_m == [0.0, 1.0, 2.0, 3.0, 4.0] and dc.overbreak_m2[2] is None


def test_design_comparison_restricted_to_a_range_lists_only_what_it_integrated():
    dc = compare_to_design(_series([10.0] * 7), 1.0, ranges=[(2.0, 4.0)])
    assert dc.chainage_m == [2.0, 3.0, 4.0]
    assert dc.coverage.requested_intervals_m == [(2.0, 4.0)]
    assert len(dc.actual_area_m2) == len(dc.chainage_m) == len(dc.overbreak_m2)


def test_segments_never_span_two_ranges():
    ser = _series([10.0] * 11)
    segs = integration_segments(ser, [(0.0, 3.0), (7.0, 10.0)])
    assert [(g.start_chainage_m, g.end_chainage_m) for g in segs] == [(0.0, 3.0), (7.0, 10.0)]
    cov = summarise_coverage(ser, segs, [(0.0, 3.0), (7.0, 10.0)])
    assert cov.complete and cov.requested_length_m == pytest.approx(6.0)


def test_a_holdout_the_series_does_not_reach_is_missing_coverage_not_full_coverage():
    """Coverage is against what was asked for, never against what happens to be in the file."""
    segs = integration_segments(_series([10.0] * 5), [(0.0, 10.0)])
    cov = summarise_coverage(_series([10.0] * 5), segs, [(0.0, 10.0)])
    assert cov.integrated_intervals_m == [(0.0, 4.0)]
    assert cov.missing_intervals_m == [(4.0, 10.0)]
    assert cov.coverage_fraction == pytest.approx(0.4)


# ---------------------------------------------------------------- T11/T12: incomplete coverage


def test_t11_incomplete_holdout_coverage_refuses_the_claim(chain, tmp_path):
    """Everything else is in place — rendered depth, verified surface, the right dataset."""
    sec = tmp_path / "gappy.json"
    rec = cut(chain.gappy, chain.dataset_dir, sec, interval_m=1, thickness_m=0.5, angle_bins=72)
    assert rec.supports_accuracy_claim

    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir))
    assert r.exit_code == 2, r.output
    assert "volume_accuracy is a claim about the whole declared holdout" in flat(r)
    assert "22-26 m" in flat(r)  # the unobserved band, named rather than integrated across


def test_t12_the_same_gappy_sections_give_a_partial_diagnostic_volume(chain, tmp_path):
    """The refusal offers "the partial volume with this coverage reported alongside it", so
    --diagnostic gives exactly that: the holdout still, not the whole drift silently swapped in
    (that is what --no-holdout-only is for)."""
    sec, out = tmp_path / "gappy.json", tmp_path / "vol.json"
    cut(chain.gappy, chain.dataset_dir, sec, interval_m=1, thickness_m=0.5, angle_bins=72)

    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir, out, diagnostic=True))
    assert r.exit_code == 0, r.output
    vol = volume_json(out)
    assert vol["claim"] == "geometry_diagnostic"
    cov = vol["coverage"]
    assert [tuple(i) for i in cov["requested_intervals_m"]] == [HOLDOUT]
    assert cov["missing_intervals_m"] and 0.0 < cov["coverage_fraction"] < 1.0
    assert vol["volume_m3"] == pytest.approx(sum(g["volume_m3"] for g in vol["segments"]))

    whole = tmp_path / "whole.json"
    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir, whole, no_holdout_only=True))
    assert r.exit_code == 0, r.output
    assert volume_json(whole)["coverage"]["requested_length_m"] > 50


def test_the_incomplete_coverage_refusal_reports_the_missing_length(chain, tmp_path):
    """It used to print the *covered* length as the amount with no observed sections."""
    sec = tmp_path / "gappy.json"
    cut(chain.gappy, chain.dataset_dir, sec, interval_m=1, thickness_m=0.5, angle_bins=72)
    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir))
    assert r.exit_code == 2, r.output
    # 22-26 m of the 20-26 m holdout is unobserved: 4 of 6 m, 66.7 %
    assert "4.00 m of 6.00 m of it (66.7%) has no observed sections" in flat(r)


def test_slabs_wider_than_the_station_spacing_cannot_carry_a_claim(chain, tmp_path):
    """Overlapping slabs let a station with no geometry of its own take its neighbours'.

    On the gappy surface a 6 m slab at 1 m spacing fills every hole in the holdout and coverage
    comes out complete. That is not a data-quality threshold: it is the condition under which
    A(s) is a measurement at s rather than a smoothing of the stations around it.
    """
    sec, out = tmp_path / "s.json", tmp_path / "v.json"
    rec = cut(chain.gappy, chain.dataset_dir, sec, interval_m=1, thickness_m=6.0, angle_bins=72)
    covered = [s for s in rec.series.sections if HOLDOUT[0] <= s.chainage_m <= HOLDOUT[1]]
    assert all(s.valid for s in covered)  # the hole is gone, filled from metres away

    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir))
    assert r.exit_code == 2, r.output
    assert "consecutive slabs overlap" in flat(r)
    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir, out, diagnostic=True))
    assert r.exit_code == 0 and volume_json(out)["claim"] == "geometry_diagnostic"


def test_the_claim_reports_the_longest_span_with_no_reconstructed_point(chain, tmp_path):
    """Measured on the cloud, so a coarser station grid cannot flatter it.

    The coarse endpoint-aligned cut below is granted (see the test above it), and this is the
    number that says what it rests on: several metres of the holdout hold no point at all.
    """
    fine, coarse = tmp_path / "fine.json", tmp_path / "coarse.json"
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    cut(chain.full, chain.dataset_dir, fine, interval_m=1, thickness_m=0.5, angle_bins=72)
    assert runner.invoke(app, volume_argv(fine, chain.dataset_dir, a)).exit_code == 0
    assert volume_json(a)["max_point_gap_m"] < 0.5

    cut(
        chain.gappy,
        chain.dataset_dir,
        coarse,
        interval_m=6,
        thickness_m=0.5,
        angle_bins=72,
        start_m=HOLDOUT[0],
        end_m=HOLDOUT[1],
    )
    r = runner.invoke(app, volume_argv(coarse, chain.dataset_dir, b))
    assert r.exit_code == 0, r.output
    vol = volume_json(b)
    assert vol["claim"] == "volume_accuracy" and vol["coverage"]["coverage_fraction"] == 1.0
    assert vol["max_point_gap_m"] > 2.0
    assert "longest span with no reconstructed point" in flat(r)


# ---------------------------------------------------------------- the reference axis


@pytest.fixture(scope="module")
def off_tree_axis(dataset_small, tmp_path_factory):
    """A dataset whose centerline lives where ``DATASET_HASH_PATTERNS`` does not look.

    The patterns cover ``centerline.csv`` at the dataset root, so for the usual layout the
    dataset hash already notices an edited axis. The manifest may name another path, and then
    the axis digest on the record is the only thing standing between a series and a polyline
    it was never cut along.
    """
    root = tmp_path_factory.mktemp("p1c_axis")
    ds = root / "dataset"
    shutil.copytree(dataset_small.dataset_dir, ds)
    (ds / "axis").mkdir()
    shutil.copy(ds / "centerline.csv", ds / "axis" / "centerline.csv")
    m = json.loads((ds / "manifest.json").read_text())
    m["centerline"]["file"] = "axis/centerline.csv"
    (ds / "manifest.json").write_text(json.dumps(m, indent=2))
    return SimpleNamespace(dataset_dir=ds, axis=ds / "axis" / "centerline.csv")


def test_an_axis_edited_behind_the_dataset_hash_is_still_caught(off_tree_axis, tmp_path):
    from minegs.core.provenance import sha256_tree
    from minegs.train.runner.base import DATASET_HASH_PATTERNS

    ds = off_tree_axis.dataset_dir
    sec = tmp_path / "s.json"
    cut(ds / "init_points.ply", ds, sec, interval_m=2, thickness_m=0.5, angle_bins=72)
    before = sha256_tree(ds, DATASET_HASH_PATTERNS)

    rows = off_tree_axis.axis.read_text().splitlines()
    shifted = [rows[0]] + [
        ",".join([c[0], c[1], f"{float(c[2]) + 0.4:.4f}", c[3]])
        for c in (r.split(",") for r in rows[1:])
    ]
    off_tree_axis.axis.write_text("\n".join(shifted) + "\n")
    assert sha256_tree(ds, DATASET_HASH_PATTERNS) == before  # invisible to the dataset hash

    r = runner.invoke(app, volume_argv(sec, ds, diagnostic=True))
    assert r.exit_code == 2, r.output
    assert "The axis was edited" in flat(r)


def test_a_record_naming_another_axis_is_refused(chain, tmp_path):
    sec = tmp_path / "s.json"
    cut(chain.raw_ply, chain.dataset_dir, sec, interval_m=2, thickness_m=0.5, angle_bins=72)
    raw = json.loads(sec.read_text())
    raw["reference_axis"] = "centerline:extracted:other.csv"
    sec.write_text(json.dumps(raw))

    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir, diagnostic=True))
    assert r.exit_code == 2, r.output
    assert "chainage means something different" in flat(r)


# ---------------------------------------------------------------- the artifact itself


def test_the_written_record_round_trips_and_names_its_whole_chain(chain, tmp_path):
    sec = tmp_path / "full.json"
    rec = cut(chain.full, chain.dataset_dir, sec, interval_m=1, thickness_m=0.5, angle_bins=72)

    assert rec.schema_version == SectionRecord.SCHEMA_VERSION
    assert rec.frame == "TLS_GLOBAL" and rec.series.frame == "TLS_GLOBAL"
    assert rec.source.surface_id and rec.source.run_id and rec.source.point_sha256
    assert rec.reference_axis == "centerline:design:centerline.csv"
    assert rec.parameters["interval_m"] == 1.0 and rec.parameters["angle_bins"] == 72
    assert rec.provenance.parent_ids[0] == rec.dataset_id
    again, series = load_section_input(sec)
    assert again.section_id == rec.section_id and len(series.sections) == len(rec.series.sections)


def test_a_sections_file_that_is_neither_shape_is_named_as_such(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"hello": "world"}))
    with pytest.raises(ContractError, match="not a section artifact"):
        load_section_input(bad)
    with pytest.raises(ContractError, match="no such sections file"):
        load_section_input(tmp_path / "nope.json")


def test_sections_outside_the_reference_axis_have_nothing_to_cut(chain, tmp_path):
    r = runner.invoke(
        app,
        sections_argv(
            chain.raw_ply, chain.dataset_dir, tmp_path / "s.json", start_m=500, end_m=600
        ),
    )
    assert r.exit_code == 2, r.output
    assert "nothing to section" in flat(r)


def test_extract_sections_is_unchanged_by_the_artifact(chain):
    """The record wraps the series; it does not alter how one is cut."""
    from minegs.core.centerline import Centerline
    from minegs.core.manifest import Manifest
    from minegs.core.pointcloud import read_ply

    ds = chain.dataset_dir
    m = Manifest.load_dataset(ds, strict_layout=False)
    cl = Centerline.from_csv(ds / m.centerline.file, m.centerline.frame, m.centerline.source)
    pc = read_ply(chain.full / "surface_points.ply").transformed(m.T_tls_from_local, "TLS_GLOBAL")
    direct = extract_sections(pc.xyz, cl, 2.0, 0.5, 72)
    assert direct.valid_count() > 0 and direct.frame == "TLS_GLOBAL"


def test_edited_parameters_no_longer_describe_the_series(chain, tmp_path):
    """The station grid is re-derived from `parameters`, so the two must agree."""
    sec = tmp_path / "s.json"
    cut(chain.raw_ply, chain.dataset_dir, sec, interval_m=2, thickness_m=0.5, angle_bins=72)
    raw = json.loads(sec.read_text())
    raw["parameters"]["interval_m"] = 1.0
    sec.write_text(json.dumps(raw))

    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir, diagnostic=True))
    assert r.exit_code == 2, r.output
    assert "one of the two was edited" in flat(r)


def test_a_malformed_integration_range_is_refused_not_silently_empty():
    """merge_intervals would drop it, and a dropped range asks for nothing — which reports as
    nothing missing, i.e. as complete coverage."""
    with pytest.raises(ContractError, match=r"hi > lo"):
        integration_segments(_series([10.0] * 5), [(4.0, 2.0)])


def test_complete_coverage_of_nothing_is_not_complete():
    assert not CoverageReport().complete
    ser = _series([10.0] * 5)
    cov = summarise_coverage(ser, integration_segments(ser), None)
    assert cov.complete and cov.requested_intervals_m == [(0.0, 4.0)]


def test_covered_and_missing_partition_the_request_exactly():
    ser = _series([10.0, 10.0, None, 10.0, 10.0, 10.0, None])
    for ranges in (None, [(0.5, 5.5)], [(0.0, 1.0), (3.0, 6.0)], [(1.5, 2.5)]):
        segs, cov = plan_integration(ser, ranges)
        got = sorted(cov.integrated_intervals_m + cov.missing_intervals_m)
        assert merge_intervals(got) == cov.requested_intervals_m, ranges
        assert cov.covered_length_m + sum(
            hi - lo for lo, hi in cov.missing_intervals_m
        ) == pytest.approx(cov.requested_length_m)
        assert 0.0 <= cov.coverage_fraction <= 1.0
        if segs:
            rep = integrate_sections(ser, "x", ranges=ranges)
            assert rep.volume_m3 == pytest.approx(sum(g.volume_m3 for g in rep.segments))


def test_the_volume_report_says_at_what_resolution_the_claim_was_made(chain, tmp_path):
    sec, out = tmp_path / "s.json", tmp_path / "v.json"
    cut(chain.full, chain.dataset_dir, sec, interval_m=1, thickness_m=0.5, angle_bins=72)
    assert runner.invoke(app, volume_argv(sec, chain.dataset_dir, out)).exit_code == 0
    vol = volume_json(out)
    assert vol["section_parameters"]["thickness_m"] == 0.5
    assert vol["section_parameters"]["angle_bins"] == 72
    assert vol["section_interval_m"] == 1.0


def test_a_holdout_with_no_two_consecutive_sections_reaches_the_coverage_refusal(chain, tmp_path):
    """Not a bare "nothing to integrate": the refusal has to say what is missing.

    A 5 m grid over the gappy surface puts one valid station (20 m) and one invalid one (25 m)
    inside the holdout, so there is no pair to integrate anywhere in it.
    """
    sec = tmp_path / "s.json"
    rec = cut(chain.gappy, chain.dataset_dir, sec, interval_m=5, thickness_m=0.5, angle_bins=72)
    inside = [s for s in rec.series.sections if HOLDOUT[0] <= s.chainage_m <= HOLDOUT[1]]
    assert sum(s.valid for s in inside) < 2

    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir))
    assert r.exit_code == 2, r.output
    assert "volume_accuracy is a claim about the whole declared holdout" in flat(r)
    assert "20-26 m" in flat(r)


def test_edited_areas_do_not_survive_being_re_cut_from_the_surface(chain, tmp_path):
    """Every identity check passes: same dataset, same axis, same surface, same grid.

    What the identity checks cannot say is that the *areas* came from that surface, because the
    series lives inside the record. A claim re-derives them.
    """
    sec, out = tmp_path / "s.json", tmp_path / "v.json"
    cut(chain.full, chain.dataset_dir, sec, interval_m=1, thickness_m=0.5, angle_bins=72)
    assert runner.invoke(app, volume_argv(sec, chain.dataset_dir, out)).exit_code == 0
    honest = volume_json(out)["volume_m3"]

    raw = json.loads(sec.read_text())
    for s in raw["series"]["sections"]:
        if s["area_m2"] is not None:
            s["area_m2"] *= 3.0
    sec.write_text(json.dumps(raw))

    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir, tmp_path / "v2.json"))
    assert r.exit_code == 2, r.output
    assert "does not reproduce this series' area_m2" in flat(r)
    # ...and the honest run really was three times smaller, so the edit would have mattered
    assert honest > 0


def test_a_gap_cannot_be_closed_by_flipping_a_station_to_observed(chain, tmp_path):
    """The gappy surface refuses on coverage; filling the hole in the JSON must not fix it."""
    sec = tmp_path / "s.json"
    rec = cut(chain.gappy, chain.dataset_dir, sec, interval_m=1, thickness_m=0.5, angle_bins=72)
    assert runner.invoke(app, volume_argv(sec, chain.dataset_dir)).exit_code == 2

    raw = json.loads(sec.read_text())
    good = [s for s in rec.series.sections if s.valid]
    fill, radii = good[0].area_m2, list(good[0].radii_m)
    for s in raw["series"]["sections"]:
        if not s["valid"]:
            s.update(valid=True, area_m2=fill, radii_m=radii, n_points=100, empty_bins=0)
    sec.write_text(json.dumps(raw))

    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir))
    assert r.exit_code == 2, r.output
    assert "cannot be made observed by editing the series" in flat(r)


def test_a_claim_needs_the_surface_to_still_be_there(chain, tmp_path):
    """Archived evidence is a claim nothing can check, so it is a refusal -- not a downgrade."""
    sec, out = tmp_path / "s.json", tmp_path / "v.json"
    cut(chain.full, chain.dataset_dir, sec, interval_m=1, thickness_m=0.5, angle_bins=72)
    moved = chain.root / "archived_surface"
    shutil.move(str(chain.full), str(moved))
    try:
        r = runner.invoke(app, volume_argv(sec, chain.dataset_dir))
        assert r.exit_code == 2, r.output
        assert "it is not there" in flat(r) and "--diagnostic" in flat(r)
        # the diagnostic number is still available, and still labelled
        r = runner.invoke(app, volume_argv(sec, chain.dataset_dir, out, diagnostic=True))
        assert r.exit_code == 0, r.output
        assert volume_json(out)["claim"] == "geometry_diagnostic"
    finally:
        shutil.move(str(moved), str(chain.full))


def test_a_coarse_station_grid_reports_how_little_of_the_span_it_sampled(chain, tmp_path):
    """Coverage is a statement about the station grid, and the grid is the caller's choice.

    Two stations 6 m apart integrate the 6 m between them completely — by the trapezoid rule,
    which is ordinary practice — so a hole that no station lands in is invisible to
    ``coverage_fraction``. Here the gappy surface has an unobserved band at 23-25 m that the 1 m
    grid refuses on and a 6 m grid pinned to the holdout ends does not see. What stops that from
    reading as a dense measurement is ``sampled_fraction``, reported on the claim, and the
    section parameters travelling into the report.

    Gated, it would need a minimum section resolution — the same kind of number the product
    owner ruled out for coverage until real data justifies one. So this test pins the behaviour
    rather than asserting a refusal: it is a known, reported limitation, not a silent one.
    """
    fine, coarse = tmp_path / "fine.json", tmp_path / "coarse.json"
    out = tmp_path / "v.json"
    args = {"thickness_m": 0.5, "angle_bins": 72, "start_m": HOLDOUT[0], "end_m": HOLDOUT[1]}
    cut(chain.gappy, chain.dataset_dir, fine, interval_m=1, **args)
    assert runner.invoke(app, volume_argv(fine, chain.dataset_dir)).exit_code == 2

    cut(chain.gappy, chain.dataset_dir, coarse, interval_m=6, **args)
    r = runner.invoke(app, volume_argv(coarse, chain.dataset_dir, out))
    assert r.exit_code == 0, r.output
    vol = volume_json(out)
    assert vol["claim"] == "volume_accuracy"
    assert vol["coverage"]["coverage_fraction"] == pytest.approx(1.0)
    # ...and the report says what it rests on: two 0.5 m slabs across 6 m of holdout
    assert vol["coverage"]["sampled_fraction"] < 0.1
    assert vol["coverage"]["sampled_length_m"] == pytest.approx(0.5)
    assert vol["section_parameters"]["interval_m"] == 6.0
    assert "sampled 0.50 m of that (8.3%)" in flat(r)
    # ...and the number the grid cannot flatter, measured on the cloud itself
    assert vol["max_point_gap_m"] > 2.0


def test_sampled_length_is_the_union_of_the_integrated_slabs():
    ser = _series([10.0, 10.0, 10.0], interval_m=1.0)
    ser.thickness_m = 0.5
    cov = summarise_coverage(ser, integration_segments(ser), None)
    assert cov.covered_length_m == pytest.approx(2.0)
    # slabs [-0.25,0.25], [0.75,1.25], [1.75,2.25] clipped to [0,2] -> 0.25 + 0.5 + 0.25
    assert cov.sampled_length_m == pytest.approx(1.0)
    assert cov.sampled_fraction == pytest.approx(0.5)

    ser.thickness_m = 4.0  # slabs wider than the span: sampled is clipped, never > covered
    cov = summarise_coverage(ser, integration_segments(ser), None)
    assert cov.sampled_length_m == pytest.approx(2.0) and cov.sampled_fraction == pytest.approx(1.0)


def test_the_headline_numbers_are_self_consistent_across_a_gap():
    """mean_area x covered_length is the volume; mean_area x (end - start) is not, and the
    report says so rather than letting a reader multiply the envelope."""
    rep = integrate_sections(_series([10.0, 10.0, None, 10.0, 10.0]), "x")

    assert rep.start_chainage_m == 0.0 and rep.end_chainage_m == 4.0  # the envelope
    assert rep.coverage.covered_length_m == pytest.approx(2.0)  # what was integrated
    assert rep.mean_area_m2 * rep.coverage.covered_length_m == pytest.approx(rep.volume_m3)
    assert rep.mean_area_m2 * (rep.end_chainage_m - rep.start_chainage_m) > rep.volume_m3


def test_a_section_with_an_area_but_no_radii_is_named_not_a_traceback():
    """np.interp with no sample points raises a bare ValueError from inside the claim path."""
    ser = _series([10.0, 10.0, 10.0])
    ser.sections[1].radii_m = [None] * ser.angle_bins
    with pytest.raises(ContractError, match="wall radii"):
        compare_to_design(ser, 1.0)
    # the area integral does not need the radii, so it is unaffected
    assert integrate_sections(ser, "x").volume_m3 == pytest.approx(20.0)


def test_a_station_deleted_from_the_series_is_a_gap_not_a_longer_trapezoid():
    """The only gap the NaN rule sees is a station that is present and invalid.

    Delete the row instead and its neighbours become adjacent: the trapezoid runs straight
    across the hole and the report says nothing is missing. The series declares its own station
    spacing, so a step wider than that is a gap too.
    """
    ser = _series([10.0, 10.0, 10.0, 10.0, 10.0])
    del ser.sections[2]
    rep = integrate_sections(ser, "x")

    assert rep.volume_m3 == pytest.approx(20.0)
    assert [(g.start_chainage_m, g.end_chainage_m) for g in rep.segments] == [
        (0.0, 1.0),
        (3.0, 4.0),
    ]
    assert rep.coverage.missing_intervals_m == [(1.0, 3.0)]
    assert not rep.coverage.complete


def test_a_station_only_one_epoch_has_is_a_gap_in_the_difference():
    """`np.intersect1d` drops it, leaving the two neighbours adjacent in the common grid."""
    from minegs.eval.change import diff_sections

    a = _series([10.0, 10.0, 10.0, 10.0, 10.0])
    b = _series([12.0, 12.0, 12.0, 12.0, 12.0])
    del b.sections[2]
    rep = diff_sections(a, b, "ep1", "ep2", "cl")

    assert rep.delta_volume_m3 == pytest.approx(2.0 * 2.0)  # 2 m2 over 1 + 1 m, not over 4 m
    assert rep.differenced_intervals_m == [(0.0, 1.0), (3.0, 4.0)]
    assert rep.missing_intervals_m == [(1.0, 3.0)]


def test_change_refuses_two_series_cut_along_different_axes(chain, tmp_path):
    """Chainage is only the same quantity in both epochs if the axis is."""
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    cut(chain.full, chain.dataset_dir, a, interval_m=2, thickness_m=0.5, angle_bins=72)
    raw = json.loads(a.read_text())
    raw["reference_axis"] = "centerline:extracted:other.csv"
    b.write_text(json.dumps(raw))

    r = runner.invoke(app, ["eval", "change", str(a), str(b)])
    assert r.exit_code == 2, r.output
    assert "different reference axes" in flat(r)


def test_change_records_the_axis_the_sections_declare(chain, tmp_path):
    a, out = tmp_path / "a.json", tmp_path / "c.json"
    cut(chain.full, chain.dataset_dir, a, interval_m=2, thickness_m=0.5, angle_bins=72)
    r = runner.invoke(app, ["eval", "change", str(a), str(a), "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert json.loads(out.read_text())["reference_axis"] == "centerline:design:centerline.csv"


def test_the_design_comparison_is_labelled_like_the_volume_beside_it(chain, tmp_path):
    sec, out = tmp_path / "s.json", tmp_path / "v.json"
    cut(chain.full, chain.dataset_dir, sec, interval_m=1, thickness_m=0.5, angle_bins=72)
    r = runner.invoke(app, volume_argv(sec, chain.dataset_dir, out, design_radius_m=2.4))
    assert r.exit_code == 0, r.output
    both = json.loads(out.read_text())
    assert both["volume"]["claim"] == "volume_accuracy" == both["design"]["claim"]


def test_interpolated_wall_bins_are_reported_on_the_claim(chain, tmp_path):
    """extract_sections fills up to a tenth of the angle bins and still calls the section valid.

    That is the same imputation the chainage axis refuses, so the claim says how much there is.
    """
    sec, out = tmp_path / "s.json", tmp_path / "v.json"
    cut(chain.full, chain.dataset_dir, sec, interval_m=1, thickness_m=0.5, angle_bins=72)
    assert runner.invoke(app, volume_argv(sec, chain.dataset_dir, out)).exit_code == 0
    assert volume_json(out)["coverage"]["interpolated_bin_fraction"] == pytest.approx(0.0)

    ser = _series([10.0, 10.0, 10.0], angle_bins=4)
    ser.sections[1].empty_bins = 1
    cov = summarise_coverage(ser, integration_segments(ser), None)
    assert cov.interpolated_bin_fraction == pytest.approx(1 / 12)
