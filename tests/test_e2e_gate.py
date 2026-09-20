"""Phase 2 — the synthetic end-to-end structural gate (§Phase 2 C6).

One survey, carried the whole way: a genuine E57 written at test time, through the real
inventory and extractor, the real dataset builder, validator, protocol judge and golden gate, a
run the real ``check_run`` accepts, the real depth manifest and its verification, the real
surface fusion and its re-derived promotion, the real geometry gate, the real section builder
and its record checks, and the real paired validation — into the report.

Two things are substituted and only two, because this machine has no GPU: the *trainer*, whose
stand-in still has to leave a run every real validator accepts, and the *renderer*, whose
stand-in mints a depth manifest naming a renderer this build does not ship. Both are recorded
as substitutions, and the report says so beside every number. **This is not G2.** A synthetic
renderer's output is not evidence about a mine, and nothing here may be read as saying it is.

What the gate *is* evidence about is the control and evidence path: that the stages hand each
other identities rather than trust, that a moved input is refused rather than reused, and that
the report says no more than its artifacts do. T1–T10 below are the adversarial cases the
phase directive names; each one breaks something a well-meaning implementation would let
through, on this workflow rather than on a fixture built for the purpose.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from minegs.core.errors import ContractError
from minegs.core.pointcloud import PointCloud, read_ply, write_ply
from minegs.core.synthetic_staging import StagingSpec, generate_staging
from minegs.e2e.models import STAGE_ORDER, Phase2Report, Stage, StageRecord, StageStatus
from minegs.e2e.runner import E2EConfig, StageContext, Workflow
from minegs.e2e.stages import default_specs

from e57_survey import write_survey_e57
from test_depth_render import StandInRenderer
from test_e2e import stand_in_trainer

pye57 = pytest.importorskip("pye57", reason="the end-to-end gate writes a real E57")

#: Small enough to write, extract and evaluate in a test; large enough that a 6 m holdout holds
#: several one-metre sections and a wall a surface can actually be fused from.
GATE_SPEC = StagingSpec(length_m=60.0, station_spacing_m=12.0, points_per_m=3000, image_size=64)
HOLDOUT = (20.0, 26.0)


def gate_build_config(root: Path, centerline: Path, convention: Path) -> Path:
    """The operator's build config. ``convention`` does not exist yet, so it gets measured."""
    spec = {
        "dataset_id": "syn_p2_gate",
        "source_frame": {"mode": "explicit_identity", "note": "synthetic survey frame"},
        "camera": {"mode": "e57_pinhole", "convention_file": str(convention)},
        "split": {"test_every": 3},
        "centerline": {"file": str(centerline), "frame": "SOURCE"},
        "geometry_holdout": {"ranges_m": [list(HOLDOUT)]},
        "initialization": {"voxel_m": 0.05, "max_points": 120_000, "sparse_max_points": 20_000},
        "capture_epoch": {"id": "ep1"},
    }
    path = root / "build_config.json"
    path.write_text(json.dumps(spec, indent=2))
    return path


@pytest.fixture(scope="module")
def survey(tmp_path_factory):
    """A real E57, a design centerline and the held-out TLS the survey is measured against."""
    root = tmp_path_factory.mktemp("survey")
    scene = generate_staging(root / "synthetic", GATE_SPEC)
    e57 = write_survey_e57(scene.staging_dir, root / "raw" / "gate_survey.e57")
    centerline = root / "centerline_source.csv"
    scene.centerline_source.to_csv(centerline)
    # float64: this survey's SOURCE frame carries a UTM-scale offset, and float32 would quantise
    # it to half a metre. The build config adopts SOURCE as TLS_GLOBAL, so the survey cloud *is*
    # the reference — the true tunnel, and never an input to the reconstruction.
    tls = write_ply(
        PointCloud(scene.points_source.xyz, rgb=scene.points_source.rgb, frame="TLS_GLOBAL"),
        root / "raw" / "tls_full.ply",
        xyz_dtype="f8",
    )
    return SimpleNamespace(root=root, e57=e57, centerline=centerline, tls=tls, scene=scene)


@pytest.fixture(scope="module")
def gate(survey, tmp_path_factory):
    """The whole chain, from the E57 file, driven by the workflow the CLI drives."""
    root = tmp_path_factory.mktemp("gate")
    cfg = E2EConfig(
        source_e57=str(survey.e57),
        staging_dir=str(root / "staging"),
        build_config=str(gate_build_config(root, survey.centerline, root / "convention.json")),
        dataset_dir=str(root / "dataset"),
        tls_reference_ply=str(survey.tls),
        stride=2,
        interval_m=1.0,
        thickness_m=0.5,
        angle_bins=72,
        max_dist_m=1.0,
    )
    wf = Workflow(root / "wf", cfg)
    state = wf.execute(
        default_specs(),
        through=Stage.REPORT,
        renderer=StandInRenderer(depth_m=7.0),
        trainer=stand_in_trainer,
    )
    return SimpleNamespace(root=root, wf=wf, state=state, config=cfg, survey=survey)


def reopened(gate, tmp_path: Path) -> Workflow:
    """The same ledger in a directory of its own, so a refusal test cannot disturb the gate."""
    root = tmp_path / "ledger"
    root.mkdir(parents=True, exist_ok=True)
    (root / "workflow.json").write_text((gate.root / "wf" / "workflow.json").read_text())
    return Workflow(root)


def outputs(gate, stage: Stage) -> dict:
    return dict(gate.state.stages[stage].outputs)


# ---------------------------------------------------------------- the chain itself


def test_the_whole_chain_runs_from_a_real_e57_through_real_production_functions(gate):
    """No stage is monkeypatched. The two substitutions are the declared hardware seams."""
    assert [gate.state.stages[s].status for s in STAGE_ORDER] == [StageStatus.SUCCEEDED] * len(
        STAGE_ORDER
    )
    ingest = outputs(gate, Stage.INGEST)
    assert ingest["mode"] == "extract" and not ingest.get("adopted")
    assert ingest["n_scans_extracted"] == 5 and ingest["n_images_extracted"] == 30
    assert ingest["n_images_skipped"] == 0
    assert ingest["mapping_status_counts"] == {"confirmed": 30}
    assert ingest["registration"] == "registered" and ingest["output_frame"] == "SOURCE"

    dataset = outputs(gate, Stage.DATASET)
    assert dataset["golden_gate_passed"] is True
    assert dataset["camera_convention_status"].startswith("measured:")
    assert "geometry_accuracy" in dataset["claims"] and "volume_accuracy" in dataset["claims"]
    assert dataset["geometry_holdout_ranges_m"] == [list(HOLDOUT)]

    assert outputs(gate, Stage.SURFACE)["depth_source"] == "minegs_render"
    assert outputs(gate, Stage.GEOMETRY)["claim"] == "geometry_accuracy"
    assert outputs(gate, Stage.SECTIONS_VOLUME)["paired_valid_count"] > 0
    assert Path(outputs(gate, Stage.REPORT)["report_md_path"]).is_file()


def test_the_camera_convention_is_measured_from_the_extracted_survey(gate):
    """Phase 0C's evidence, earned again from the file rather than carried in beside it."""
    convention = json.loads((gate.root / "convention.json").read_text())
    assert convention["status"] == "selected"
    assert convention["margin"] > convention["min_margin"]
    measured = outputs(gate, Stage.DATASET)["camera_convention"]
    assert measured["source"] == "calibrated" and measured["label"] == "cam(+X,-Y,-Z)"
    assert measured["R_e57cam_from_cam"] == convention["convention"]["R_e57cam_from_cam"]


# ---------------------------------------------------------------- T1–T4: identity


def test_t1_a_replaced_survey_is_not_the_survey_the_ledger_describes(gate, tmp_path):
    """T1: the E57's digest is what every stage downstream ultimately hangs off."""
    wf = reopened(gate, tmp_path)
    original = gate.survey.e57.read_bytes()
    try:
        gate.survey.e57.write_bytes(original + b"a different survey")
        # Every stage, not just ingest: the chained upstream fingerprint carries it down.
        assert wf.stale_stages(default_specs()) == list(STAGE_ORDER)
        with pytest.raises(ContractError, match="source_sha256"):
            wf.execute(default_specs(), through=Stage.INGEST)
    finally:
        gate.survey.e57.write_bytes(original)


def test_t2_a_changed_dataset_is_not_the_dataset_the_run_was_trained_on(gate, tmp_path):
    """T2: the run's evidence is about a dataset hash, and that hash is re-read from disk."""
    wf = reopened(gate, tmp_path)
    image = sorted((Path(gate.config.dataset_dir) / "images").rglob("*.png"))[0]
    original = image.read_bytes()
    try:
        image.write_bytes(original + b"\n")
        stale = wf.stale_stages(default_specs())
        # DATASET's own inputs are the staging tree and the build config; neither moved. What
        # moved is what the dataset now hashes to, which is TRAIN's input and everything after.
        assert Stage.DATASET not in stale
        after_train = STAGE_ORDER[STAGE_ORDER.index(Stage.TRAIN) :]
        assert stale == list(after_train)
        with pytest.raises(ContractError, match="dataset_hash"):
            wf.execute(default_specs())
    finally:
        image.write_bytes(original)


def test_t3_depth_will_not_touch_a_run_the_train_stage_did_not_finish(gate, tmp_path):
    """T3: a failed TRAIN is not a run; DEPTH refuses before it can render anything."""
    wf = reopened(gate, tmp_path)
    wf.state.stages[Stage.TRAIN] = StageRecord(
        stage=Stage.TRAIN, status=StageStatus.FAILED, failure_reason="ContractError: backend died"
    )
    ctx = StageContext(stage=Stage.DEPTH, config=wf.config, work_dir=wf.work_dir, state=wf.state)
    with pytest.raises(ContractError, match="stage depth needs train, which is failed"):
        default_specs()[Stage.DEPTH].inputs(ctx)


def test_t4_a_surface_whose_points_moved_cannot_be_compared(gate, tmp_path):
    """T4: the record is a claim *about* a file, so the file is what gets checked."""
    wf = reopened(gate, tmp_path)
    surface_dir = Path(outputs(gate, Stage.SURFACE)["surface_dir"])
    (ply,) = surface_dir.glob("*.ply")
    original = ply.read_bytes()
    try:
        ply.write_bytes(original + b"\n")
        with pytest.raises(ContractError, match="not the points this surface was built from"):
            wf.execute(
                default_specs(),
                rebuild_from=Stage.SECTIONS_VOLUME,
                renderer=StandInRenderer(depth_m=7.0),
                trainer=stand_in_trainer,
            )
    finally:
        ply.write_bytes(original)


# ---------------------------------------------------------------- T5–T8: the measurement


def test_t5_nothing_outside_the_holdout_reaches_a_g2_metric(gate):
    """T5: the claim is accuracy *on held-out TLS*, and the stage offers no flag to widen it."""
    geometry = outputs(gate, Stage.GEOMETRY)
    report = json.loads(Path(geometry["report_path"]).read_text())
    assert report["chainage_range_m"] == list(HOLDOUT)
    assert report["claim"] == "geometry_accuracy"
    command = gate.state.stages[Stage.GEOMETRY].command
    assert command["holdout_only"] is True and command["diagnostic"] is False

    sv = outputs(gate, Stage.SECTIONS_VOLUME)
    paired = json.loads(Path(sv["paired_validation_path"]).read_text())
    assert paired["sections"]["requested_intervals_m"] == [list(HOLDOUT)]
    for station in paired["sections"]["stations"]:
        assert HOLDOUT[0] - 1e-9 <= station["chainage_m"] <= HOLDOUT[1] + 1e-9


def test_t6_two_series_cut_on_different_grids_are_not_comparable(gate):
    """T6: "the area at 22 m" means nothing shared unless both sides cut the same way."""
    from minegs.eval.geometry.evaluate import load_dataset_and_centerline
    from minegs.eval.sections import SectionRecord, build_section_record, section_source
    from minegs.eval.volume import compare_to_reference

    sv = outputs(gate, Stage.SECTIONS_VOLUME)
    pred = SectionRecord.load(sv["sections_predicted_path"])
    dataset_dir = Path(gate.config.dataset_dir)
    manifest, centerline = load_dataset_and_centerline(dataset_dir)
    coarse = build_section_record(
        read_ply(gate.survey.tls).xyz,
        section_source(None, gate.survey.tls),
        dataset_dir,
        manifest,
        centerline,
        interval_m=2.0,
        thickness_m=0.5,
        angle_bins=72,
    )
    with pytest.raises(ContractError, match=r"not cut on the same grid|one station list"):
        compare_to_reference(pred, coarse, [HOLDOUT])


@pytest.fixture(scope="module")
def gapped(gate):
    """The same prediction with a 22–24 m band of its surface removed.

    A reconstruction that does not reach part of the tunnel is the ordinary case, and it is
    exactly where a trapezoid across the hole would invent volume and shrink the error.
    """
    from minegs.eval.geometry.evaluate import (
        load_dataset_and_centerline,
        resolve_prediction,
        to_tls,
    )
    from minegs.eval.sections import SectionRecord, build_section_record, section_source
    from minegs.eval.volume import compare_to_reference

    sv = outputs(gate, Stage.SECTIONS_VOLUME)
    dataset_dir = Path(gate.config.dataset_dir)
    manifest, centerline = load_dataset_and_centerline(dataset_dir)
    resolved = resolve_prediction(
        Path(outputs(gate, Stage.SURFACE)["surface_dir"]), dataset_dir, manifest, diagnostic=False
    )
    xyz = to_tls(resolved.points, manifest).xyz
    s, _ = centerline.project(xyz)
    keep = ~((s >= 22.0) & (s <= 24.0))
    cut = {"interval_m": 1.0, "thickness_m": 0.5, "angle_bins": 72}
    holed = build_section_record(
        xyz[keep],
        section_source(resolved.surface, Path(outputs(gate, Stage.SURFACE)["surface_dir"])),
        dataset_dir,
        manifest,
        centerline,
        **cut,
    )
    ref = SectionRecord.load(sv["sections_reference_path"])
    return SimpleNamespace(
        record=holed,
        reference=ref,
        validation=compare_to_reference(holed, ref, [HOLDOUT]),
        whole=compare_to_reference(
            SectionRecord.load(sv["sections_predicted_path"]), ref, [HOLDOUT]
        ),
    )


def test_t7_no_section_or_volume_error_is_integrated_across_a_missing_station(gapped):
    """T7: missing geometry is reported as missing, never replaced by a trapezoid."""
    volume = gapped.validation.volume
    integrated = [tuple(i) for i in volume.integrated_intervals_m]

    assert integrated and all(hi <= 22.0 + 1e-9 or lo >= 24.0 - 1e-9 for lo, hi in integrated)
    assert any(lo <= 22.0 and hi >= 24.0 for lo, hi in volume.missing_intervals_m)
    assert volume.common_covered_length_m < volume.requested_length_m
    # The band is the prediction's hole, and the report says which side is missing.
    assert any(lo <= 22.5 <= hi for lo, hi in volume.reference_only_intervals_m)
    # A station only one side observed is not an error of zero: it is not an error at all.
    inside = [st for st in gapped.validation.sections.stations if 22.0 <= st.chainage_m <= 24.0]
    assert inside and all(st.pred_area_m2 is None for st in inside)
    assert all(st.absolute_error_m2 is None for st in inside)


def test_t8_differing_coverage_is_compared_on_the_common_domain_only(gapped):
    """T8: 100 m³ over one coverage minus 102 m³ over another is not a 2 m³ error."""
    holed, whole = gapped.validation.volume, gapped.whole.volume

    assert whole.coverage_fraction > holed.coverage_fraction
    # The reference volume is *not* the same number in the two comparisons: it is integrated
    # over the spans both sides observed, so removing prediction shrinks both sides together.
    assert holed.reference_volume_m3 < whole.reference_volume_m3
    assert holed.predicted_volume_m3 < whole.predicted_volume_m3
    # And the error stays an error about the tunnel they share, not about the hole.
    assert holed.absolute_error_m3 == pytest.approx(
        abs(holed.predicted_volume_m3 - holed.reference_volume_m3), abs=1e-9
    )
    assert gapped.validation.sections.missing_prediction_count > 0
    assert gapped.validation.sections.missing_reference_count == 0


# ---------------------------------------------------------------- T9–T10: the report


def test_t9_every_number_in_the_report_is_the_number_in_the_artifact(gate):
    """T9: the report aggregates. A second computation is a second answer."""
    sv = outputs(gate, Stage.SECTIONS_VOLUME)
    paired = json.loads(Path(sv["paired_validation_path"]).read_text())
    geometry = json.loads(Path(outputs(gate, Stage.GEOMETRY)["report_path"]).read_text())
    report = Phase2Report.load(outputs(gate, Stage.REPORT)["report_json_path"])

    assert report.volume.predicted_volume_m3 == paired["volume"]["predicted_volume_m3"]
    assert report.volume.reference_volume_m3 == paired["volume"]["reference_volume_m3"]
    assert report.volume.absolute_error_m3 == paired["volume"]["absolute_error_m3"]
    assert report.volume.relative_error == paired["volume"]["relative_error"]
    assert report.volume.coverage_fraction == paired["volume"]["coverage_fraction"]
    assert report.sections.mean_absolute_error_m2 == paired["sections"]["mean_absolute_error_m2"]
    assert report.geometry.chamfer_m == geometry["chamfer_m"]
    assert report.geometry.accuracy_median_m == geometry["accuracy"]["median_m"]
    assert report.geometry.completeness_p95_m == geometry["completeness"]["p95_m"]
    assert report.dataset.dataset_id == outputs(gate, Stage.DATASET)["dataset_id"]
    assert report.source.sha256 == outputs(gate, Stage.INGEST)["source_sha256"]


def test_t10_the_report_does_not_overstate_what_produced_it(gate):
    """T10: a structural gate that reads like a G2 is worse than no gate at all."""
    report = Phase2Report.load(outputs(gate, Stage.REPORT)["report_json_path"])
    md = Path(outputs(gate, Stage.REPORT)["report_md_path"]).read_text()

    assert report.maturity.real_data_validation_status == "not_validated"
    assert report.maturity.human_visual_review_status == "pending"
    assert report.training.real_gpu_execution is False
    assert report.reconstruction.real_renderer_execution is False
    assert "NOT VALIDATED" in report.maturity.statement and "PENDING" in report.maturity.statement
    assert any("substituted" in note for note in report.maturity.notes)
    assert "real GPU training executed: **False**" in md
    # The geometry claim is the one the protocol allows, copied — not promoted by the report.
    assert report.geometry.claim == outputs(gate, Stage.GEOMETRY)["claim"]


# ---------------------------------------------------------------- the operator's entry point


def cli(*args) -> object:
    from minegs.cli.main import app
    from typer.testing import CliRunner

    return CliRunner().invoke(app, list(args))


def test_status_says_where_the_workflow_is_without_touching_it(gate):
    before = (gate.root / "wf" / "workflow.json").read_text()
    result = cli("e2e", "status", "--work-dir", str(gate.root / "wf"))

    assert result.exit_code == 0, result.output
    assert "complete through" in result.output and "report" in result.output
    assert (gate.root / "wf" / "workflow.json").read_text() == before


def test_report_regenerates_the_document_and_runs_no_stage(gate, tmp_path):
    before = (gate.root / "wf" / "workflow.json").read_text()
    result = cli("e2e", "report", "--work-dir", str(gate.root / "wf"), "--out", str(tmp_path / "r"))

    assert result.exit_code == 0, result.output
    assert (tmp_path / "r" / "phase2_report.json").is_file()
    assert (tmp_path / "r" / "phase2_report.md").is_file()
    assert "not_validated" in result.output and "pending" in result.output
    assert (gate.root / "wf" / "workflow.json").read_text() == before


def test_the_cli_offers_no_way_to_substitute_the_hardware():
    """The seams are reached from Python by a test that records it, never from the shell."""
    result = cli("e2e", "run", "--help")
    assert result.exit_code == 0
    options = [word for word in result.output.split() if word.startswith("--")]
    for flag in ("--renderer", "--trainer", "--fake-renderer", "--stand-in", "--no-gpu"):
        assert flag not in options


def test_reporting_on_a_directory_with_no_workflow_says_so(tmp_path):
    result = cli("e2e", "report", "--work-dir", str(tmp_path))
    assert result.exit_code == 2
    assert "no workflow there" in result.output


def test_a_stage_name_that_is_not_a_stage_lists_the_ones_that_are(gate, tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(gate.config.model_dump(mode="json")))
    result = cli(
        "e2e", "run", str(path), "--work-dir", str(tmp_path / "fresh"), "--through", "trainn"
    )

    assert result.exit_code == 2
    assert "is not a workflow stage" in result.output
    assert "sections_volume" in result.output


def test_reopening_a_workflow_with_a_different_config_is_refused(gate, tmp_path):
    """The ledger's fingerprints are statements about settings; adopting new ones erases that."""
    cfg = gate.config.model_dump(mode="json")
    cfg["interval_m"] = 2.0
    path = tmp_path / "other.json"
    path.write_text(json.dumps(cfg))
    result = cli("e2e", "run", str(path), "--work-dir", str(gate.root / "wf"))

    assert result.exit_code == 2
    assert "different workflow config" in result.output


def test_a_relative_path_in_a_config_is_read_against_the_config(tmp_path):
    from minegs.e2e.runner import load_e2e_config

    (tmp_path / "survey.e57").write_bytes(b"not read here")
    (tmp_path / "cfg.json").write_text(json.dumps({"source_e57": "survey.e57", "profile": "light"}))
    cfg = load_e2e_config(tmp_path / "cfg.json")

    assert cfg.source_e57 == str(tmp_path / "survey.e57")
    assert np.isclose(cfg.interval_m, 1.0)  # untouched defaults stay defaults
