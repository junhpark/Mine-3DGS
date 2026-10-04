"""Phase 5 C3 — the chunk runs of one plan put back together, and every way of mixing them refused.

The "runs" are records and the "surfaces" are the scanned wall pulled toward the axis by a
different factor per chunk, restricted to that chunk's support, so which chunk a stitched station
came from can be read off its area. No training or rendering happens here; the point is what the
composition will and will not put together, and that each station is counted once.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from minegs.chunks.compose import (
    ChunkInputs,
    ChunkRunSet,
    compose_chunk_set,
    stitch_sections,
)
from minegs.chunks.plan import build_chunk_plan
from minegs.core.errors import ContractError
from minegs.core.pointcloud import read_ply
from minegs.core.provenance import ProvenanceRecord, sha256_file, sha256_tree
from minegs.eval.geometry.evaluate import load_dataset_and_centerline
from minegs.eval.sections.build import build_section_record
from minegs.eval.sections.models import SectionSource
from minegs.train.backends.gsplat import PINNED_GSPLAT
from minegs.train.runner.base import DATASET_HASH_PATTERNS, RunRecord, RunStatus

from phase5_scene import CORE_M, HOLDOUT, OVERLAP_M, long_tunnel

CUT = {"interval_m": 2.0, "thickness_m": 0.5, "angle_bins": 36}
SCALE = {"K000": 0.97, "K001": 0.98, "K002": 0.99}
LIGHT = {"name": "light", "data_factor": 4}
IDS = ("K000", "K001", "K002")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("p5c")
    sc = long_tunnel(root / "scene")
    ds = sc.dataset_dir
    plan, plan_path = build_chunk_plan(ds, CORE_M, OVERLAP_M)
    m, cl = load_dataset_and_centerline(ds)
    cloud = read_ply(sc.cloud).xyz
    s, _ = cl.project(cloud)
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
    w = SimpleNamespace(
        root=root, ds=ds, plan=plan, plan_path=plan_path, m=m, cl=cl, cloud=cloud, s=s, ref=ref
    )
    w.runs = {c: _run(w, f"run_{c}", c) for c in IDS}
    w.sections = {c: _sections(w, c, f"run_{c}") for c in IDS}
    return w


def _sections(w, chunk_id: str, run_id: str, scale: float | None = None, **cut):
    """The sections of `run_id`'s surface: the wall over the chunk's support, scaled per chunk."""
    lo, hi = w.plan.chunk(chunk_id).support_range_m
    keep = (w.s >= lo) & (w.s <= hi)
    axis = w.cl.point_at(w.s[keep])
    pts = axis + (w.cloud[keep] - axis) * (SCALE[chunk_id] if scale is None else scale)
    src = SectionSource(
        kind="surface",
        surface_id=f"surface_{run_id}",
        run_id=run_id,
        depth_source="minegs_render",
        point_sha256=hashlib.sha256(run_id.encode()).hexdigest(),
        point_path=f"/nonexistent/{run_id}",
    )
    return build_section_record(pts, src, w.ds, w.m, w.cl, **{**CUT, **cut})


def _binding(plan, plan_path, chunk_id: str) -> dict:
    c = plan.chunk(chunk_id)
    return {
        "plan_id": plan.plan_id,
        "plan_digest": plan.plan_digest,
        "plan_path": str(Path(plan_path).resolve()),
        "chunk_id": c.chunk_id,
        "ordinal": c.ordinal,
        "n_chunks": len(plan.chunks),
        "core_range_m": list(c.core_range_m),
        "support_range_m": list(c.support_range_m),
        "capture_groups": list(c.capture_groups),
        "images": list(c.images),
        "actual_image_support_m": list(c.actual_image_support_m),
        "scene_scale": None,
    }


def _run(w, name: str, chunk_id: str, **over) -> Path:
    run_dir = w.root / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model.ply").write_bytes(name.encode())
    rec = RunRecord(
        run_id=name,
        dataset_id=over.pop("dataset_id", w.m.dataset_id),
        dataset_hash=over.pop("dataset_hash", sha256_tree(w.ds, DATASET_HASH_PATTERNS)),
        backend={"name": "gsplat", "version": PINNED_GSPLAT},
        profile=over.pop("profile", LIGHT),
        runner="local",
        status=over.pop("status", RunStatus.SUCCEEDED),
        T_local_from_internal=over.pop("T", np.eye(4).tolist()),
        chunk_id=chunk_id,
        chunk=over.pop("chunk") if "chunk" in over else _binding(w.plan, w.plan_path, chunk_id),
        final_model="model.ply",
        gaussian_count=1000,
        provenance=ProvenanceRecord(),
        **over,
    )
    rec.save(run_dir / "run.json")
    return run_dir


def _inputs(w, **swap) -> list[ChunkInputs]:
    """One input per chunk, with any of them swapped for (run_dir, sections) or dropped (None)."""
    out = []
    for c in IDS:
        item = swap.get(c, (w.runs[c], w.sections[c]))
        if item is not None:
            out.append(ChunkInputs(*item))
    return out


def _compose(w, inputs=None, **kw) -> ChunkRunSet:
    return compose_chunk_set(
        w.ds,
        kw.pop("plan", w.plan_path),
        _inputs(w) if inputs is None else inputs,
        kw.pop("ref", w.ref),
        chunk_set_id=kw.pop("chunk_set_id", "cset_test"),
        **kw,
    )


def _areas(sec) -> dict[float, float | None]:
    return {round(s.chainage_m, 6): s.area_m2 for s in sec.series.sections}


# ================================================================ the complete set


@pytest.fixture(scope="module")
def complete(world):
    return _compose(world)


def test_a_complete_set_covers_the_whole_core_once(world, complete):
    cs = complete
    assert cs.complete and cs.missing_chunks == [] and cs.missing_intervals_m == []
    assert cs.coverage_fraction == pytest.approx(1.0)
    lo, hi = world.plan.core_extent()
    assert cs.requested_core_length_m == pytest.approx(hi - lo)
    assert cs.completed_core_length_m == pytest.approx(hi - lo)
    assert cs.plan_id == world.plan.plan_id and cs.plan_digest == world.plan.plan_digest
    assert cs.dataset_id == world.m.dataset_id and cs.frame == "LOCAL_METRIC"
    assert [e.chunk_id for e in cs.chunks] == list(IDS)
    assert all(e.status == "succeeded" and e.sections_id for e in cs.chunks)
    # every station of the reference grid, each owned by exactly one chunk
    stations = world.ref.series.chainages()
    owners = dict(cs.stitched["owners"])
    assert sum(owners.values()) == cs.stitched["station_count"] == len(stations)
    expect = np.bincount(world.plan.owner_index(stations), minlength=3)
    assert [owners[c] for c in IDS] == expect.tolist() == [50, 50, 51]
    assert [e.owned_stations for e in cs.chunks] == [50, 50, 51]
    # the models are indexed, not merged
    assert all(e.model["gaussian_count"] == 1000 and e.model["sha256"] for e in cs.chunks)


def test_each_station_comes_from_its_owner_and_overlap_is_never_counted_twice(world):
    stitched, counts = stitch_sections(world.plan, world.sections, world.ref)
    got = _areas(stitched)
    by_chunk = {c: _areas(world.sections[c]) for c in IDS}
    owner = world.plan.owner_index(np.array(list(got)))
    for (s, area), k in zip(got.items(), owner, strict=True):
        assert area == by_chunk[IDS[int(k)]][s], s
    # inside the overlap both chunks observed the wall; only the owner's section is used
    assert by_chunk["K000"][110.0] is not None and by_chunk["K001"][110.0] is not None
    assert got[110.0] == by_chunk["K001"][110.0] != by_chunk["K000"][110.0]
    assert got[90.0] == by_chunk["K000"][90.0] != by_chunk["K001"][90.0]
    # the boundary station belongs to the chunk after it, the axis end to the last
    assert got[100.0] == by_chunk["K001"][100.0] and got[200.0] == by_chunk["K002"][200.0]
    assert got[300.0] == by_chunk["K002"][300.0]
    assert sum(counts.values()) == len(got) == len(set(got))


def test_the_stitched_volume_is_the_integral_of_owned_sections_only(world, complete):
    stitched, _ = stitch_sections(world.plan, world.sections, world.ref)
    s = stitched.series.chainages()
    a = stitched.series.areas()
    ok = np.isfinite(a)
    v = complete.evaluation["extent"]["volume"]["predicted_volume_m3"]
    assert v == pytest.approx(np.trapezoid(a[ok], s[ok]), rel=1e-9)
    # adding the overlaps would integrate 2 x 40 m twice: the stitched total is well below it
    per_chunk_support = 0.0
    for c in IDS:
        lo, hi = world.plan.chunk(c).support_range_m
        sec = world.sections[c].series
        cs_, ca = sec.chainages(), sec.areas()
        m = np.isfinite(ca) & (cs_ >= lo) & (cs_ <= hi)
        per_chunk_support += float(np.trapezoid(ca[m], cs_[m]))
    assert per_chunk_support > v * 1.2


def test_the_holdout_is_evaluated_and_seams_are_numbers_not_verdicts(world, complete):
    ev = complete.evaluation
    assert ev["holdout"]["ranges_m"] == [list(HOLDOUT)] and ev["holdout"]["diagnostic"] is False
    assert "volume_accuracy" in ev["protocol"]["claims"] and ev["protocol"]["refusals"] == []
    assert ev["holdout"]["paired_valid_count"] == 4  # 92, 94, 96, 98
    # K000 owns the holdout: its pull toward the axis is what the error measures
    assert ev["holdout"]["mean_signed_error_m2"] < 0
    assert ev["extent"]["diagnostic"] is True
    assert set(ev["per_chunk"]) == set(IDS)
    assert all(v["diagnostic"] is True for v in ev["per_chunk"].values())
    assert [(x["left"], x["right"]) for x in complete.seams] == [("K000", "K001"), ("K001", "K002")]
    assert complete.seams[0]["overlap_m"] == [80.0, 120.0]
    assert complete.seams[1]["overlap_m"] == [180.0, 220.0]
    for seam in complete.seams:
        lo, hi = seam["overlap_m"]
        left, right = _areas(world.sections[seam["left"]]), _areas(world.sections[seam["right"]])
        both = [s for s in left if lo <= s <= hi and left[s] is not None and right[s] is not None]
        assert seam["n_paired_sections"] == len(both) >= 19  # 21 stations; edges see half a slab
        # K001 sits further out than K000, K002 further than K001
        assert seam["mean_signed_area_difference_m2"] > 0
        assert seam["median_abs_area_difference_m2"] > 0
    text = complete.model_dump_json()
    for verdict in ('"pass"', '"passed"', '"PASS"', "seam_free", "seam-free"):
        assert verdict not in text
    assert not any(k in s for s in complete.seams for k in ("pass", "ok", "verdict"))
    assert complete.real_execution is False and complete.g3_status == "PENDING"
    assert any("agreement between chunks, not accuracy" in n for n in complete.notes)


def test_the_set_round_trips_and_records_its_parents(world, complete, tmp_path):
    complete.save(tmp_path / "chunk_run_set.json")
    back = ChunkRunSet.load(tmp_path / "chunk_run_set.json")
    assert back.model_dump() == complete.model_dump()
    assert set(back.provenance.parent_ids) >= {world.plan.plan_id, *(f"run_{c}" for c in IDS)}


def test_explicit_ranges_are_diagnostic(world):
    cs = _compose(world, ranges=[(150.0, 160.0)])
    assert cs.evaluation["holdout"]["diagnostic"] is True
    assert cs.evaluation["holdout"]["ranges_m"] == [[150.0, 160.0]]
    assert any("diagnostic, not a held-out" in n for n in cs.notes)


# ================================================================ incomplete sets


def test_a_missing_chunk_is_refused_and_reported_as_missing_only_when_asked(world):
    with pytest.raises(ContractError, match=r"\['K002'\] have no succeeded run"):
        _compose(world, _inputs(world, K002=None))
    cs = _compose(world, _inputs(world, K002=None), allow_incomplete=True)
    assert cs.complete is False and cs.missing_chunks == ["K002"]
    k2 = world.plan.chunk("K002").core_range_m
    assert cs.missing_intervals_m == [k2]
    lo, hi = world.plan.core_extent()
    assert cs.completed_core_length_m == pytest.approx(k2[0] - lo)
    assert cs.coverage_fraction == pytest.approx((k2[0] - lo) / (hi - lo))
    assert cs.chunks[2].status == "missing" and cs.chunks[2].run_id is None
    # K002's stations are unobserved, not zero and not K001's overlap
    stitched, _ = stitch_sections(
        world.plan, {c: world.sections[c] for c in ("K000", "K001")}, world.ref
    )
    got = _areas(stitched)
    assert got[210.0] is None and got[200.0] is None
    assert cs.seams[1]["n_paired_sections"] is None
    k2_owned = [s for s in got if s >= k2[0]]
    assert len(k2_owned) == 51 and all(got[s] is None for s in k2_owned)
    unobserved = sum(a is None for a in got.values())
    assert cs.evaluation["extent"]["missing_prediction_count"] == unobserved >= 51
    assert cs.real_execution is False


def test_a_failed_chunk_run_is_refused_and_never_composed(world):
    failed = _run(world, "run_K002_failed", "K002", status=RunStatus.FAILED)
    with pytest.raises(ContractError, match="K002 run run_K002_failed is failed"):
        _compose(world, _inputs(world, K002=(failed, world.sections["K002"])))
    cs = _compose(
        world, _inputs(world, K002=(failed, world.sections["K002"])), allow_incomplete=True
    )
    assert cs.missing_chunks == ["K002"] and cs.chunks[2].status == "failed"
    assert cs.chunks[2].sections_id is None and cs.chunks[2].model is None


def test_nothing_succeeded_is_nothing_to_compose(world):
    with pytest.raises(ContractError, match="nothing to compose"):
        _compose(world, [], allow_incomplete=True)


# ================================================================ mixing


def test_a_run_of_another_dataset_is_refused(world):
    other = _run(world, "run_other_ds", "K001", dataset_id="another_tunnel")
    with pytest.raises(ContractError, match="was trained on another_tunnel"):
        _compose(world, _inputs(world, K001=(other, world.sections["K001"])))
    stale = _run(world, "run_stale_ds", "K001", dataset_hash="0" * 64)
    with pytest.raises(ContractError, match="not on this dataset as it is now"):
        _compose(world, _inputs(world, K001=(stale, world.sections["K001"])))


def test_a_chunk_of_another_plan_is_refused(world, tmp_path):
    other_plan, other_path = build_chunk_plan(world.ds, 150.0, 10.0, out_dir=tmp_path / "op")
    assert other_plan.plan_digest != world.plan.plan_digest
    foreign = _run(world, "run_other_plan", "K001", chunk=_binding(other_plan, other_path, "K001"))
    with pytest.raises(ContractError, match=f"of plan {other_plan.plan_id}, not of"):
        _compose(world, _inputs(world, K001=(foreign, world.sections["K001"])))


def test_a_run_whose_core_or_support_disagrees_with_the_plan_is_refused(world):
    b = _binding(world.plan, world.plan_path, "K001")
    b["support_range_m"] = [90.0, 210.0]
    shifted = _run(world, "run_shifted", "K001", chunk=b)
    with pytest.raises(ContractError, match="another core/support for K001"):
        _compose(world, _inputs(world, K001=(shifted, world.sections["K001"])))
    b = _binding(world.plan, world.plan_path, "K000")
    relabelled = _run(world, "run_relabelled", "K001", chunk=b)
    with pytest.raises(ContractError, match="names chunk K001 but its binding is for K000"):
        _compose(world, _inputs(world, K001=(relabelled, world.sections["K001"])))


def test_a_run_that_is_not_a_chunk_is_refused(world):
    whole = _run(world, "run_whole", None, chunk=None)
    with pytest.raises(ContractError, match="not a chunk of a plan"):
        _compose(world, _inputs(world, K001=(whole, world.sections["K001"])))


def test_runs_of_different_profiles_are_one_experiment_too_many(world):
    heavy = _run(world, "run_heavy", "K001", profile={"name": "heavy", "data_factor": 2})
    with pytest.raises(ContractError, match="one set is one experiment"):
        _compose(world, _inputs(world, K001=(heavy, world.sections["K001"])))
    # same name, one setting changed by an override: still another experiment
    tweaked = _run(world, "run_tweaked", "K002", profile={"name": "light", "data_factor": 2})
    with pytest.raises(ContractError, match="one set is one experiment"):
        _compose(world, _inputs(world, K002=(tweaked, world.sections["K002"])))


def test_a_chunk_or_a_run_given_twice_is_refused(world):
    again = _run(world, "run_K001_again", "K001")
    twice = [*_inputs(world), ChunkInputs(again, _sections(world, "K001", "run_K001_again"))]
    with pytest.raises(ContractError, match="chunk K001 is given twice"):
        _compose(world, twice)
    same = [*_inputs(world), ChunkInputs(world.runs["K001"], world.sections["K001"])]
    with pytest.raises(ContractError, match="run run_K001 is given twice"):
        _compose(world, same)


def test_outputs_outside_the_metric_frame_are_refused(world):
    internal = _run(world, "run_internal", "K001", frame_of_outputs="BACKEND_INTERNAL")
    with pytest.raises(ContractError, match="outputs are in BACKEND_INTERNAL"):
        _compose(world, _inputs(world, K001=(internal, world.sections["K001"])))
    T = np.eye(4)
    T[:3, :3] *= 1.5
    scaled = _run(world, "run_scaled", "K001", T=T.tolist())
    with pytest.raises(ContractError, match="non-identity"):
        _compose(world, _inputs(world, K001=(scaled, world.sections["K001"])))
    T = np.eye(4)
    T[0, 3] = 12.0
    shifted = _run(world, "run_offset", "K001", T=T.tolist())
    with pytest.raises(ContractError, match="non-identity"):
        _compose(world, _inputs(world, K001=(shifted, world.sections["K001"])))


# ================================================================ sections


def test_sections_not_from_that_run_are_refused(world):
    with pytest.raises(ContractError, match="not from the surface of run run_K001"):
        _compose(world, _inputs(world, K001=(world.runs["K001"], world.sections["K000"])))
    with pytest.raises(ContractError, match="not from the surface of run run_K001"):
        _compose(world, _inputs(world, K001=(world.runs["K001"], world.ref)))


def test_sections_cut_from_the_reference_cloud_are_refused(world):
    sec = world.sections["K001"]
    forged = sec.model_copy(
        update={
            "source": sec.source.model_copy(update={"point_sha256": world.ref.source.point_sha256})
        }
    )
    with pytest.raises(ContractError, match="cut from the reference cloud"):
        _compose(world, _inputs(world, K001=(world.runs["K001"], forged)))


def test_a_reference_that_is_not_a_scan_is_refused(world):
    with pytest.raises(ContractError, match="not a scanned cloud"):
        _compose(world, ref=world.sections["K001"])


def test_sections_on_another_grid_are_refused(world):
    coarse = _sections(world, "K001", "run_K001", interval_m=4.0)
    with pytest.raises(ContractError, match="interval_m"):
        _compose(world, _inputs(world, K001=(world.runs["K001"], coarse)))
    bins = _sections(world, "K001", "run_K001", angle_bins=24)
    with pytest.raises(ContractError, match="angle_bins"):
        _compose(world, _inputs(world, K001=(world.runs["K001"], bins)))
    # cut over the chunk's own support: a chunk-local grid is not the whole-axis grid
    local = _sections(world, "K001", "run_K001", start_m=80.0, end_m=220.0)
    with pytest.raises(ContractError, match="stations"):
        _compose(world, _inputs(world, K001=(world.runs["K001"], local)))


def test_a_repeated_chainage_is_refused(world):
    sec = world.sections["K001"]
    rows = list(sec.series.sections)
    dup = sec.model_copy(
        update={"series": sec.series.model_copy(update={"sections": [*rows[:-1], rows[-2]]})}
    )
    with pytest.raises(ContractError, match="station"):
        _compose(world, _inputs(world, K001=(world.runs["K001"], dup)))
    # stitching on its own refuses it too, without the record checks in front of it
    with pytest.raises(ContractError, match="missing, moved or repeated"):
        stitch_sections(world.plan, {**world.sections, "K001": dup}, world.ref)
    ref_rows = list(world.ref.series.sections)
    ref_dup = SimpleNamespace(
        **{k: getattr(world.ref, k) for k in ("reference_axis", "dataset_id", "dataset_hash")},
        frame=world.ref.frame,
        series=world.ref.series.model_copy(update={"sections": [*ref_rows, ref_rows[-1]]}),
    )
    with pytest.raises(ContractError, match="repeats a chainage"):
        stitch_sections(world.plan, world.sections, ref_dup)


def test_a_station_no_chunk_owns_is_refused(world):
    rows = list(world.ref.series.sections)
    beyond = rows[-1].model_copy(update={"chainage_m": 310.0})
    ref_long = SimpleNamespace(
        reference_axis=world.ref.reference_axis,
        dataset_id=world.ref.dataset_id,
        dataset_hash=world.ref.dataset_hash,
        frame=world.ref.frame,
        series=world.ref.series.model_copy(update={"sections": [*rows, beyond]}),
    )
    with pytest.raises(ContractError, match="no chunk owns them"):
        stitch_sections(world.plan, {}, ref_long)
    with pytest.raises(ContractError, match=r"\['K009'\] are not in plan"):
        stitch_sections(world.plan, {"K009": world.sections["K001"]}, world.ref)


# ================================================================ the plan itself


def test_a_plan_with_a_gap_or_a_changed_dataset_composes_nothing(world, tmp_path):
    data = json.loads(Path(world.plan_path).read_text())
    data["chunks"][1]["core_range_m"] = [102.0, 200.0]
    gap = tmp_path / "gap" / "chunk_plan.json"
    gap.parent.mkdir()
    gap.write_text(json.dumps(data))
    with pytest.raises(ContractError, match="gap between"):
        _compose(world, plan=gap)
    data = json.loads(Path(world.plan_path).read_text())
    data["chunks"][1]["core_range_m"] = [98.0, 200.0]
    double = tmp_path / "double" / "chunk_plan.json"
    double.parent.mkdir()
    double.write_text(json.dumps(data))
    with pytest.raises(ContractError, match="owned twice"):
        _compose(world, plan=double)


def test_a_changed_dataset_is_refused_before_anything_is_composed(world, tmp_path):
    ds2 = tmp_path / "ds"
    shutil.copytree(world.ds, ds2)
    cl = ds2 / "centerline.csv"
    cl.write_text(cl.read_text() + "\n")
    with pytest.raises(ContractError, match="chainage means something else"):
        compose_chunk_set(
            ds2,
            ds2 / "chunks" / world.plan.plan_id / "chunk_plan.json",
            _inputs(world),
            world.ref,
            chunk_set_id="cset_changed",
        )


# ================================================================ the CLI


def test_the_cli_writes_the_chunk_run_set_and_refuses_an_incomplete_one(world, tmp_path):
    from minegs.cli.main import app
    from typer.testing import CliRunner

    paths = {}
    for c in IDS:
        paths[c] = tmp_path / f"sections_{c}.json"
        world.sections[c].save(paths[c])
    ref = tmp_path / "sections_ref.json"
    world.ref.save(ref)
    head = ["eval", "chunk-set", str(world.ds), "--chunk-plan", str(world.plan_path)]
    head += ["--reference-sections", str(ref)]
    argv = [*head]
    for c in IDS:
        argv += ["--run", str(world.runs[c]), "--sections", str(paths[c])]
    res = CliRunner().invoke(app, [*argv, "--out", str(tmp_path / "out")])
    assert res.exit_code == 0, res.output
    cs = ChunkRunSet.load(tmp_path / "out" / "chunk_run_set.json")
    assert cs.complete and cs.coverage_fraction == pytest.approx(1.0)
    assert "G3: PENDING" in res.output

    short = [*head]
    for c in ("K000", "K001"):
        short += ["--run", str(world.runs[c]), "--sections", str(paths[c])]

    def said(res) -> str:
        return " ".join(res.output.split())

    res = CliRunner().invoke(app, short)
    assert res.exit_code != 0 and "have no succeeded run" in said(res)
    res = CliRunner().invoke(app, [*short, "--allow-incomplete", "--out", str(tmp_path / "o2")])
    assert res.exit_code == 0, res.output
    assert ChunkRunSet.load(tmp_path / "o2" / "chunk_run_set.json").missing_chunks == ["K002"]
    # a succeeded run without its sections, and sections of a run that was not given
    res = CliRunner().invoke(app, [*short, "--run", str(world.runs["K002"])])
    assert res.exit_code != 0 and "no sections cut from its surface" in said(res)
    res = CliRunner().invoke(app, [*short, "--sections", str(paths["K002"])])
    assert res.exit_code != 0 and "not in --run" in said(res)


# ================================================================ rendering a chunk


def test_a_chunk_run_renders_and_back_projects_only_its_own_views(world, tmp_path):
    from minegs.eval.surface.depth import (
        build_depth_surface,
        depth_digest,
        rederive_depth_source,
    )
    from minegs.eval.surface.render import render_depths
    from minegs.ingest.common.colmap_io import read_model

    from test_depth_render import StandInRenderer, make_run

    model = read_model(world.ds / "sparse" / "0")

    run_dir = tmp_path / "runs" / "run_K001_render"
    make_run(
        world.ds,
        run_dir,
        run_id="run_K001_render",
        chunk_id="K001",
        chunk=_binding(world.plan, world.plan_path, "K001"),
    )
    renderer = StandInRenderer()
    seen: list[str] = []
    real_render = renderer.render

    def spy(checkpoint, cameras, images, expected_step=None):
        seen.extend(im.name for im in images.values())
        yield from real_render(checkpoint, cameras, images, expected_step)

    renderer.render = spy
    manifest, depth_dir = render_depths(run_dir, world.ds, tmp_path / "depth", renderer=renderer)
    views = set(world.plan.chunk("K001").views)
    assert set(seen) == views and len(seen) == len(views)
    assert {d.image_name for d in manifest.depths} == views
    rec, _ = build_depth_surface(depth_dir, world.ds, run_dir, tmp_path / "surface")
    assert rec.run_id == "run_K001_render" and rec.depth_source == "minegs_render"
    assert rec.parameters["depth_manifest_id"] == manifest.manifest_id
    # the surface records, and a claim re-derives, the chunk's views — not the dataset's
    assert rec.depth_map_count == rec.parameters["expected_views"] == len(views)
    assert rec.depth_sha256 == depth_digest(
        depth_dir, {i: im for i, im in model.images.items() if im.name in views}
    )
    assert rederive_depth_source(rec, world.ds) is None
    # K001's views include test groups in its support: rendering is evaluation, not training
    assert set(world.plan.chunk("K001").images) < views


def test_a_chunk_run_whose_plan_moved_or_vanished_cannot_be_rendered(world, tmp_path):
    from minegs.eval.surface.render import render_depths

    from test_depth_render import FakeRenderer, make_run

    b = _binding(world.plan, world.plan_path, "K001")
    other_plan, other_path = build_chunk_plan(world.ds, 150.0, 10.0, out_dir=tmp_path / "op")
    b["plan_path"] = str(Path(other_path).resolve())
    run_dir = tmp_path / "runs" / "run_swapped"
    make_run(world.ds, run_dir, run_id="run_swapped", chunk_id="K001", chunk=b)
    with pytest.raises(ContractError, match=f"now holds {other_plan.plan_id}"):
        render_depths(run_dir, world.ds, tmp_path / "d1", renderer=FakeRenderer())
    run_dir = tmp_path / "runs" / "run_unbound"
    make_run(world.ds, run_dir, run_id="run_unbound", chunk_id="K001")
    with pytest.raises(ContractError, match="without a chunk plan"):
        render_depths(run_dir, world.ds, tmp_path / "d2", renderer=FakeRenderer())


# ================================================================ C4 hostile-review fixes


def test_a_reference_cut_over_part_of_the_axis_is_refused(world):
    """S5: on a sub-range grid a core can own no station and the set would still say complete."""
    sub = build_section_record(
        world.cloud,
        world.ref.source,
        world.ds,
        world.m,
        world.cl,
        **CUT,
        start_m=0.0,
        end_m=150.0,
    )
    inputs = [
        ChunkInputs(world.runs[c], _sections(world, c, f"run_{c}", start_m=0.0, end_m=150.0))
        for c in IDS
    ]
    with pytest.raises(ContractError, match="whole-axis grid"):
        _compose(world, inputs, ref=sub)


def test_a_grid_too_coarse_for_a_core_is_a_gap_not_coverage(world):
    """S5: a whole-axis grid whose interval is longer than a core leaves that core unread."""
    coarse = {"interval_m": 250.0}
    ref = build_section_record(
        world.cloud, world.ref.source, world.ds, world.m, world.cl, **{**CUT, **coarse}
    )
    inputs = [ChunkInputs(world.runs[c], _sections(world, c, f"run_{c}", **coarse)) for c in IDS]
    with pytest.raises(ContractError, match=r"\['K001'\] own no station"):
        _compose(world, inputs, ref=ref)


def test_the_holdout_is_diagnostic_when_the_protocol_refuses_it(world, monkeypatch):
    """S8: numbers over the holdout are only held out when the dataset protocol says so."""
    from minegs.eval import protocol

    real = protocol.judge

    def refusing(manifest):
        j = real(manifest)
        return j.model_copy(
            update={
                "claims": [c for c in j.claims if c is not protocol.Claim.VOLUME_ACCURACY],
                "refusals": [*j.refusals, "holdout points are in the init"],
            }
        )

    monkeypatch.setattr(protocol, "judge", refusing)
    cs = _compose(world)
    assert cs.evaluation["holdout"]["diagnostic"] is True
    assert cs.evaluation["protocol"]["refusals"] == ["holdout points are in the init"]
    assert any("holdout numbers are diagnostic" in n for n in cs.notes)
    assert not any("only the holdout evaluation is about held-out" in n for n in cs.notes)


def test_chunks_supervised_by_different_depth_artifacts_are_refused(world):
    sup = {"artifact_sha256": "d" * 64, "path": "/x"}
    other = _run(world, "run_other_depth", "K001", depth_supervision=sup)
    with pytest.raises(ContractError, match="depth artifact dddddddddddd"):
        _compose(world, _inputs(world, K001=(other, world.sections["K001"])))
    assert _compose(world).depth_supervision_sha256 is None


def test_a_seam_narrower_than_float_slack_is_reported_empty_not_refused(world, tmp_path):
    plan, path = build_chunk_plan(world.ds, 100.02, 4e-10, out_dir=tmp_path / "tiny")
    assert [c.chunk_id for c in plan.chunks] == list(IDS)
    inputs = []
    for c in IDS:
        name = f"run_tiny_{c}"
        run = _run(world, name, c, chunk=_binding(plan, path, c))
        inputs.append(ChunkInputs(run, _sections(world, c, name)))
    cs = _compose(world, inputs, plan=path)
    assert cs.complete and all(s["n_paired_sections"] is None for s in cs.seams)


def test_a_chunk_the_profile_thinned_is_named_in_the_notes(world):
    b = {**_binding(world.plan, world.plan_path, "K001"), "staged_images": 10, "planned_images": 93}
    thin = _run(world, "run_thin", "K001", chunk=b)
    cs = _compose(world, _inputs(world, K001=(thin, _sections(world, "K001", "run_thin"))))
    assert any("K001 (10/93)" in n for n in cs.notes)
    assert not any("max_images" in n for n in _compose(world).notes)
