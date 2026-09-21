"""Phase 2 — the end-to-end contract (§Phase 2).

C0 freezes the shape before anything executes: the stage graph, the ledger that makes resuming
safe, and the report that may not say more than its artifacts do. The orchestration that fills
them in follows in later checkpoints; what is asserted here is the contract they have to keep.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from minegs.core.errors import ContractError
from minegs.core.provenance import ProvenanceRecord, sha256_file
from minegs.e2e.models import (
    STAGE_ORDER,
    MaturitySummary,
    Phase2Report,
    Stage,
    StageRecord,
    StageStatus,
    WorkflowState,
)
from minegs.e2e.runner import (
    E2EConfig,
    StageContext,
    StageOutcome,
    StageSpec,
    Workflow,
    fingerprint,
)
from minegs.e2e.stages import (
    dataset_identity,
    dataset_spec,
    default_specs,
    depth_spec,
    ingest_spec,
    staging_digest,
    train_spec,
)
from minegs.eval.volume import compare_to_reference
from minegs.train.runner.base import RunStatus

from e57_fakes import make_scan, pose_node
from test_depth_render import StandInRenderer, make_run
from test_section_volume import _series


def blank_state(**over) -> WorkflowState:
    stages = {s: StageRecord(stage=s) for s in STAGE_ORDER}
    stages.update(over.pop("stages", {}))
    return WorkflowState(
        workflow_id="wf_test",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        stages=stages,
        provenance=ProvenanceRecord(),
        **over,
    )


def test_the_stage_graph_is_declared_once_and_is_the_survey_order():
    assert [s.value for s in STAGE_ORDER] == [
        "ingest",
        "dataset",
        "train",
        "depth",
        "surface",
        "geometry",
        "sections_volume",
        "report",
    ]
    assert tuple(Stage) == STAGE_ORDER


def test_a_ledger_missing_a_stage_cannot_say_whether_it_ran():
    with pytest.raises(ValueError, match="missing stage records"):
        WorkflowState(
            workflow_id="wf",
            created_at="t",
            updated_at="t",
            stages={Stage.INGEST: StageRecord(stage=Stage.INGEST)},
            provenance=ProvenanceRecord(),
        )


def test_a_stage_record_filed_under_the_wrong_key_is_refused():
    stages = {s: StageRecord(stage=s) for s in STAGE_ORDER}
    stages[Stage.DEPTH] = StageRecord(stage=Stage.SURFACE)
    with pytest.raises(ValueError, match="wrong key"):
        blank_state(stages=stages)


def test_reused_is_usable_and_distinct_from_succeeded():
    """The report has to be able to say what this workflow actually ran."""
    assert StageRecord(stage=Stage.TRAIN, status=StageStatus.REUSED).usable
    assert StageRecord(stage=Stage.TRAIN, status=StageStatus.SUCCEEDED).usable
    assert not StageRecord(stage=Stage.TRAIN, status=StageStatus.FAILED).usable
    assert not StageRecord(stage=Stage.TRAIN, status=StageStatus.PENDING).usable
    assert StageStatus.REUSED != StageStatus.SUCCEEDED


def test_done_through_stops_at_the_first_stage_that_is_not_usable():
    state = blank_state()
    for s in (Stage.INGEST, Stage.DATASET, Stage.TRAIN):
        state.stages[s].status = StageStatus.SUCCEEDED
    state.stages[Stage.DEPTH].status = StageStatus.FAILED
    state.stages[Stage.SURFACE].status = StageStatus.SUCCEEDED  # cannot be reached
    assert state.done_through() == [Stage.INGEST, Stage.DATASET, Stage.TRAIN]


def test_the_state_round_trips_through_its_file(tmp_path):
    state = blank_state(source_sha256="a" * 64)
    state.stages[Stage.INGEST].status = StageStatus.SUCCEEDED
    state.stages[Stage.INGEST].input_fingerprint = "b" * 64
    back = WorkflowState.load(state.save(tmp_path / "workflow.json"))
    assert back.source_sha256 == "a" * 64
    assert back.stages[Stage.INGEST].status is StageStatus.SUCCEEDED
    assert back.stages[Stage.INGEST].input_fingerprint == "b" * 64
    assert back.schema_version == WorkflowState.SCHEMA_VERSION


def test_a_report_defaults_to_claiming_nothing():
    """Every summary is optional and every maturity field starts at its weakest value."""
    rep = Phase2Report(
        report_id="rep", workflow_id="wf", generated_at="t", provenance=ProvenanceRecord()
    )
    assert rep.maturity.structural_status == "incomplete"
    assert rep.maturity.real_data_validation_status == "not_validated"
    assert rep.maturity.human_visual_review_status == "pending"
    assert rep.training.real_gpu_execution is False
    assert rep.reconstruction.real_renderer_execution is False
    assert rep.volume.predicted_volume_m3 is None and rep.geometry.chamfer_m is None


def test_validated_is_not_something_a_run_can_say_about_itself():
    """G2 is not passed by numbers alone: a human has to have looked (§Phase 2 §16)."""
    with pytest.raises(ValueError, match="human_visual_review_status=pass"):
        MaturitySummary(real_data_validation_status="validated")
    ok = MaturitySummary(real_data_validation_status="validated", human_visual_review_status="pass")
    assert ok.real_data_validation_status == "validated"


def test_the_report_carries_both_geometry_directions():
    """One-directional accuracy hides a hole; the schema does not let the report print one."""
    fields = set(Phase2Report.model_fields["geometry"].annotation.model_fields)
    assert {"accuracy_median_m", "accuracy_p95_m"} <= fields
    assert {"completeness_median_m", "completeness_p95_m"} <= fields
    assert {"chamfer_m", "f_score"} <= fields


# ---------------------------------------------------------------- C1: the ledger and the loop


def spec_for(stage: Stage, inputs, ran: list[Stage], outputs=None) -> StageSpec:
    def _run(ctx: StageContext) -> StageOutcome:
        ran.append(stage)
        return StageOutcome(outputs=dict(outputs or {"id": f"{stage.value}_out"}))

    return StageSpec(stage=stage, inputs=inputs, run=_run)


def chain_specs(ran: list[Stage], world: dict) -> dict[Stage, StageSpec]:
    """A stage graph shaped like the real one: each stage's inputs come from the one before.

    ``world`` stands in for what is on disk, so a test can change it under a completed stage.
    """
    specs: dict[Stage, StageSpec] = {}
    for i, stage in enumerate(STAGE_ORDER):
        if i == 0:
            specs[stage] = spec_for(stage, lambda ctx: {"source": world["source"]}, ran)
        else:
            prev = STAGE_ORDER[i - 1]
            specs[stage] = spec_for(
                stage, lambda ctx, prev=prev: {"upstream": ctx.upstream(prev)}, ran
            )
    return specs


def test_a_workflow_runs_its_stages_in_order_and_records_each_one(tmp_path):
    ran: list[Stage] = []
    world = {"source": "e57_a"}
    wf = Workflow(tmp_path / "wf", E2EConfig(source_e57="a.e57"))
    state = wf.execute(chain_specs(ran, world))

    assert ran == list(STAGE_ORDER)
    for stage in STAGE_ORDER:
        rec = state.stages[stage]
        assert rec.status is StageStatus.SUCCEEDED
        assert rec.input_fingerprint and rec.started_at and rec.completed_at
        assert rec.elapsed_seconds is not None and rec.elapsed_seconds >= 0
        assert rec.minegs_version and rec.tool_versions
        assert rec.outputs == {"id": f"{stage.value}_out"}


def test_a_second_run_reuses_every_stage_and_says_so(tmp_path):
    ran: list[Stage] = []
    world = {"source": "e57_a"}
    wf = Workflow(tmp_path / "wf", E2EConfig(source_e57="a.e57"))
    wf.execute(chain_specs(ran, world))
    ran.clear()

    again = Workflow(tmp_path / "wf").execute(chain_specs(ran, world))
    assert ran == []  # nothing re-ran
    assert all(again.stages[s].status is StageStatus.REUSED for s in STAGE_ORDER)


def test_a_changed_source_refuses_to_reuse_the_stage_it_changed(tmp_path):
    """The whole point of the fingerprint: a green tick on evidence about another survey."""
    ran: list[Stage] = []
    world = {"source": "e57_a"}
    wf = Workflow(tmp_path / "wf", E2EConfig(source_e57="a.e57"))
    wf.execute(chain_specs(ran, world))

    world["source"] = "e57_b"
    with pytest.raises(ContractError, match="completed against different inputs"):
        Workflow(tmp_path / "wf").execute(chain_specs([], world))


def test_a_stale_stage_is_visible_before_anything_is_attempted(tmp_path):
    ran: list[Stage] = []
    world = {"source": "e57_a"}
    wf = Workflow(tmp_path / "wf", E2EConfig(source_e57="a.e57"))
    wf.execute(chain_specs(ran, world))
    assert Workflow(tmp_path / "wf").stale_stages(chain_specs([], world)) == []

    # ...and it cascades: every stage after it was built on what changed
    world["source"] = "e57_b"
    assert Workflow(tmp_path / "wf").stale_stages(chain_specs([], world)) == list(STAGE_ORDER)


def test_rebuild_from_is_the_explicit_answer_and_cascades(tmp_path):
    ran: list[Stage] = []
    world = {"source": "e57_a"}
    Workflow(tmp_path / "wf", E2EConfig(source_e57="a.e57")).execute(chain_specs(ran, world))
    ran.clear()

    world["source"] = "e57_b"
    state = Workflow(tmp_path / "wf").execute(chain_specs(ran, world), rebuild_from=Stage.INGEST)
    assert ran == list(STAGE_ORDER)  # everything after the changed stage too
    assert all(state.stages[s].status is StageStatus.SUCCEEDED for s in STAGE_ORDER)


def test_rebuilding_one_stage_leaves_the_ones_before_it_reused(tmp_path):
    ran: list[Stage] = []
    world = {"source": "e57_a"}
    Workflow(tmp_path / "wf", E2EConfig(source_e57="a.e57")).execute(chain_specs(ran, world))
    ran.clear()

    state = Workflow(tmp_path / "wf").execute(chain_specs(ran, world), rebuild_from=Stage.DEPTH)
    assert ran == [Stage.DEPTH, Stage.SURFACE, Stage.GEOMETRY, Stage.SECTIONS_VOLUME, Stage.REPORT]
    assert state.stages[Stage.TRAIN].status is StageStatus.REUSED
    assert state.stages[Stage.DEPTH].status is StageStatus.SUCCEEDED


def test_a_downstream_stage_refuses_to_run_on_an_upstream_that_did_not(tmp_path):
    ran: list[Stage] = []
    world = {"source": "e57_a"}
    specs = chain_specs(ran, world)

    def boom(ctx: StageContext) -> StageOutcome:
        raise ContractError("the extractor said no")

    specs[Stage.INGEST] = StageSpec(
        stage=Stage.INGEST, inputs=lambda ctx: {"source": world["source"]}, run=boom
    )
    wf = Workflow(tmp_path / "wf", E2EConfig(source_e57="a.e57"))
    with pytest.raises(ContractError, match="the extractor said no"):
        wf.execute(specs)

    # the ledger survives the crash and says where it stopped
    state = WorkflowState.load(tmp_path / "wf" / "workflow.json")
    assert state.stages[Stage.INGEST].status is StageStatus.FAILED
    assert "the extractor said no" in (state.stages[Stage.INGEST].failure_reason or "")
    assert state.stages[Stage.INGEST].elapsed_seconds is not None
    assert state.stages[Stage.DATASET].status is StageStatus.PENDING

    # ...and a later stage will not build on the hole it left
    ctx = StageContext(
        stage=Stage.DATASET,
        config=E2EConfig(source_e57="a.e57"),
        work_dir=tmp_path / "wf",
        state=state,
    )
    with pytest.raises(ContractError, match="needs ingest, which is failed"):
        ctx.upstream(Stage.INGEST)


def test_through_stops_where_it_is_told(tmp_path):
    ran: list[Stage] = []
    world = {"source": "e57_a"}
    state = Workflow(tmp_path / "wf", E2EConfig(source_e57="a.e57")).execute(
        chain_specs(ran, world), through=Stage.SURFACE
    )
    assert ran == [Stage.INGEST, Stage.DATASET, Stage.TRAIN, Stage.DEPTH, Stage.SURFACE]
    assert state.done_through() == ran
    assert state.stages[Stage.GEOMETRY].status is StageStatus.PENDING


def test_reopening_a_workflow_with_a_different_config_is_a_different_workflow(tmp_path):
    Workflow(tmp_path / "wf", E2EConfig(source_e57="a.e57", profile="light"))
    with pytest.raises(ContractError, match="different workflow config"):
        Workflow(tmp_path / "wf", E2EConfig(source_e57="a.e57", profile="heavy"))
    # ...and reopening with the same config, or with none, is fine
    assert Workflow(tmp_path / "wf", E2EConfig(source_e57="a.e57", profile="light"))
    assert Workflow(tmp_path / "wf").config.profile == "light"


def test_a_config_that_does_not_name_what_a_stage_needs_says_all_of_it_at_once():
    cfg = E2EConfig(source_e57="a.e57")
    assert cfg.require("source_e57") == ("a.e57",)
    with pytest.raises(ContractError, match=r"\['staging_dir', 'dataset_dir'\]"):
        cfg.require("source_e57", "staging_dir", "dataset_dir")


def test_the_fingerprint_does_not_depend_on_key_order():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


# ---------------------------------------------------------------- C2: ingest and dataset


def e2e_config(**over) -> E2EConfig:
    return E2EConfig(**over)


def _fake_scan():
    """A registered scan with coordinates — the smallest thing `extract` can stage."""
    import numpy as np

    n = 6
    return make_scan(
        guid="{scan-a}",
        pose=pose_node((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        point_data={
            "cartesianX": np.arange(n, dtype=np.float64),
            "cartesianY": np.zeros(n),
            "cartesianZ": np.zeros(n),
        },
    )


def write_build_config(tmp_path: Path, build_config_small, convention: Path) -> Path:
    """The operator's build config, with the convention file wherever the test wants it."""
    spec = build_config_small.model_dump(mode="json")
    spec["camera"] = {"mode": "e57_pinhole", "convention_file": str(convention)}
    path = tmp_path / "build_config.json"
    path.write_text(json.dumps(spec, indent=2))
    return path


def test_ingest_extracts_an_e57_and_carries_its_identity(fake_e57, tmp_path):
    """The real inventory and the real extractor, through the stage that will call them."""
    path, _ = fake_e57([_fake_scan()])
    wf = Workflow(
        tmp_path / "wf",
        e2e_config(source_e57=str(path), staging_dir=str(tmp_path / "staging")),
    )
    state = wf.execute({Stage.INGEST: ingest_spec()}, through=Stage.INGEST)

    out = state.stages[Stage.INGEST].outputs
    assert out["mode"] == "extract"
    assert out["source_sha256"] == sha256_file(path)
    assert out["source_file_name"] == path.name
    assert out["n_scans_extracted"] == 1
    assert out["registration"] == "registered" and out["output_frame"] == "SOURCE"
    assert (tmp_path / "staging" / "extraction_manifest.json").is_file()
    assert out["staging_digest"] == staging_digest(tmp_path / "staging")
    assert state.stages[Stage.INGEST].command["call"].endswith("extract.extract")


def test_a_replaced_e57_will_not_be_carried_by_an_old_ingest_record(fake_e57, tmp_path):
    """T1: source identity changes, so everything recorded about it is about another survey."""
    path, _ = fake_e57([_fake_scan()])
    cfg = e2e_config(source_e57=str(path), staging_dir=str(tmp_path / "staging"))
    Workflow(tmp_path / "wf", cfg).execute({Stage.INGEST: ingest_spec()}, through=Stage.INGEST)

    path.write_bytes(path.read_bytes() + b"a different survey")
    wf = Workflow(tmp_path / "wf")
    assert wf.stale_stages({Stage.INGEST: ingest_spec()}) == [Stage.INGEST]
    with pytest.raises(ContractError, match="source_sha256"):
        wf.execute({Stage.INGEST: ingest_spec()}, through=Stage.INGEST)


def test_ingest_adopts_a_staging_tree_it_did_not_make_and_records_that(staging_small, tmp_path):
    """Extracting on the machine with the disk for it is legitimate — and is not the same thing."""
    wf = Workflow(tmp_path / "wf", e2e_config(staging_dir=str(staging_small.staging_dir)))
    state = wf.execute({Stage.INGEST: ingest_spec()}, through=Stage.INGEST)

    out = state.stages[Stage.INGEST].outputs
    assert out["mode"] == "adopted_staging" and out["adopted"] is True
    assert out["source_sha256"] and out["n_scans_extracted"] > 0
    assert out["staging_digest"] == staging_digest(staging_small.staging_dir)


@pytest.fixture(scope="module")
def dataset_stage(staging_small, build_config_small, tmp_path_factory):
    """One real Phase 0C build, driven by the DATASET stage, shared by the tests below."""
    root = tmp_path_factory.mktemp("c2")
    convention = root / "camera_convention.json"
    cfg = e2e_config(
        staging_dir=str(staging_small.staging_dir),
        build_config=str(write_build_config(root, build_config_small, convention)),
        dataset_dir=str(root / "dataset"),
    )
    wf = Workflow(root / "wf", cfg)
    state = wf.execute(
        {Stage.INGEST: ingest_spec(), Stage.DATASET: dataset_spec()}, through=Stage.DATASET
    )
    return SimpleNamespace(root=root, state=state, config=cfg, convention=convention)


def test_the_dataset_stage_builds_validates_judges_and_gates(dataset_stage):
    out = dataset_stage.state.stages[Stage.DATASET].outputs
    ident = dataset_identity(out["dataset_dir"])

    assert out["dataset_id"] == ident["dataset_id"]
    assert out["dataset_hash"] == ident["dataset_hash"]
    assert out["n_stations"] > 0 and out["n_images"] > 0 and out["init_points"] > 0
    assert "geometry_holdout" in out["protocols"]
    assert "volume_accuracy" in out["claims"]
    assert out["golden_gate_passed"] is True and out["golden_gate_result"] == "pass"
    from minegs.dataset.golden_gate import REPORT_FILE

    assert Path(out["golden_gate_dir"], REPORT_FILE).is_file()


def test_the_camera_convention_is_measured_and_recorded_never_assumed(dataset_stage):
    """§12: the convention is evidence. A default here would be a silent hard-code."""
    out = dataset_stage.state.stages[Stage.DATASET].outputs
    assert out["camera_convention_status"] == "measured:selected"
    assert dataset_stage.convention.is_file()
    assert out["camera_convention_sha256"] == sha256_file(dataset_stage.convention)
    assert out["camera_convention"]["source"]


def test_the_runner_does_not_invent_a_holdout(dataset_stage):
    """The manifest and judge() are the source of truth; the workflow only carries them."""
    from minegs.core.manifest import Manifest
    from minegs.eval.protocol import judge

    out = dataset_stage.state.stages[Stage.DATASET].outputs
    manifest = Manifest.load_dataset(out["dataset_dir"], strict_layout=False)
    declared = [list(r) for r in judge(manifest).holdout_ranges_m]
    assert out["geometry_holdout_ranges_m"] == declared
    assert declared == [list(r) for r in manifest.split.geometry_holdout.chainage_ranges_m]


def test_an_edited_staging_tree_makes_the_dataset_stage_stale(dataset_stage, tmp_path):
    """T2 upstream: the build's input is re-read from disk, not copied from the ingest record."""
    import shutil

    staging = Path(dataset_stage.state.stages[Stage.INGEST].outputs["staging_dir"])
    moved = tmp_path / "staging_copy"
    shutil.copytree(staging, moved)
    cfg = dataset_stage.config.model_dump(mode="json")
    cfg["staging_dir"] = str(moved)
    cfg["dataset_dir"] = str(tmp_path / "dataset")
    wf = Workflow(tmp_path / "wf", E2EConfig.from_dict(cfg))
    specs = {Stage.INGEST: ingest_spec(), Stage.DATASET: dataset_spec()}
    wf.execute(specs, through=Stage.DATASET)

    inv = moved / "inventory.json"
    inv.write_text(inv.read_text().replace('"scan_count"', '"scan_count" '))
    again = Workflow(tmp_path / "wf")
    assert again.stale_stages(specs) == [Stage.INGEST, Stage.DATASET]
    with pytest.raises(ContractError, match="completed against different inputs"):
        again.execute(specs, through=Stage.DATASET)


def test_a_supplied_camera_convention_is_left_exactly_as_it_is(
    staging_small, build_config_small, calibration_small, tmp_path
):
    """An existing convention is the operator's reviewed evidence, not a cache to refresh."""
    _, conv_path = calibration_small
    mine = tmp_path / "camera_convention.json"
    mine.write_text(conv_path.read_text())
    before = sha256_file(mine)

    cfg = e2e_config(
        staging_dir=str(staging_small.staging_dir),
        build_config=str(write_build_config(tmp_path, build_config_small, mine)),
        dataset_dir=str(tmp_path / "dataset"),
    )
    state = Workflow(tmp_path / "wf", cfg).execute(
        {Stage.INGEST: ingest_spec(), Stage.DATASET: dataset_spec()}, through=Stage.DATASET
    )
    out = state.stages[Stage.DATASET].outputs
    assert out["camera_convention_status"] == "supplied"
    assert sha256_file(mine) == before


# ---------------------------------------------------------------- C3: train, depth, surface


def stand_in_trainer(run_cfg, runner_cfg):
    """A run the real validators accept, minted without a GPU.

    The seam is the *execution*, not the contract: what comes out still has to pass check_run,
    still has to carry this dataset's hash, and still has to leave a checkpoint the depth
    manifest can be verified against.
    """
    run_dir = Path(run_cfg.run_dir)
    make_run(Path(run_cfg.dataset_dir), run_dir, run_id=run_dir.name)
    return run_dir


@pytest.fixture(scope="module")
def tls_reference(staging_small, tmp_path_factory):
    """The held-out TLS, in TLS_GLOBAL.

    The synthetic survey's SOURCE frame is adopted as TLS_GLOBAL by the build config
    (``explicit_identity``), so the survey cloud *is* the reference — the true tunnel the
    reconstruction is measured against, and never an input to it.
    """
    from minegs.core.pointcloud import PointCloud, write_ply

    src = staging_small.points_source
    out = tmp_path_factory.mktemp("tls") / "tls_full.ply"
    # float64: this survey's SOURCE frame carries a UTM-scale offset, and float32 would quantise
    # it to half a metre. A real TLS_GLOBAL reference has the same problem and the same answer.
    return write_ply(PointCloud(src.xyz, rgb=src.rgb, frame="TLS_GLOBAL"), out, xyz_dtype="f8")


@pytest.fixture(scope="module")
def chain(staging_small, build_config_small, tls_reference, tmp_path_factory):
    """The whole chain, driven by the workflow, with both hardware seams substituted."""
    root = tmp_path_factory.mktemp("c3")
    cfg = e2e_config(
        staging_dir=str(staging_small.staging_dir),
        build_config=str(write_build_config(root, build_config_small, root / "conv.json")),
        dataset_dir=str(root / "dataset"),
        tls_reference_ply=str(tls_reference),
        stride=2,
        interval_m=1.0,
        thickness_m=0.5,
        angle_bins=72,
        max_dist_m=1.0,
    )
    wf = Workflow(root / "wf", cfg)
    specs = default_specs()
    state = wf.execute(
        specs,
        through=Stage.REPORT,
        renderer=StandInRenderer(depth_m=7.0),
        trainer=stand_in_trainer,
    )
    return SimpleNamespace(root=root, wf=wf, state=state, config=cfg, specs=specs)


def test_the_train_stage_validates_the_run_it_launched(chain):
    """check_run is the same gate `eval surface-depth` applies; launching it is not evidence."""
    out = chain.state.stages[Stage.TRAIN].outputs
    dataset = chain.state.stages[Stage.DATASET].outputs

    assert out["status"] == "succeeded"
    assert out["dataset_id"] == dataset["dataset_id"]
    assert out["dataset_hash"] == dataset["dataset_hash"]
    assert out["frame_of_outputs"] == "LOCAL_METRIC"
    assert out["final_checkpoint"] and out["checkpoint_sha256"]
    assert out["real_gpu_execution"] is False  # substituted, and said so


def test_a_substituted_trainer_claims_no_gpu_it_did_not_use(chain):
    """The host having a GPU would say nothing about where these weights came from."""
    env = chain.state.stages[Stage.TRAIN].runtime_env
    assert env["gpu_model"] is None and env["cuda_version"] is None
    assert "substituted" in env["reason"]


def test_the_depth_stage_produces_a_manifest_tied_to_that_run(chain):
    out = chain.state.stages[Stage.DEPTH].outputs
    train = chain.state.stages[Stage.TRAIN].outputs

    assert out["run_id"] == train["run_id"]
    assert out["depth_manifest_id"] and out["depth_manifest_sha256"]
    assert out["depth_map_count"] > 0 and 0.0 < out["mean_valid_ratio"] <= 1.0
    assert out["real_renderer_execution"] is False


def test_the_surface_stage_re_derives_its_own_promotion(chain):
    """Having built the surface is not evidence about it: it is re-read and re-derived."""
    out = chain.state.stages[Stage.SURFACE].outputs

    assert out["depth_source"] == "minegs_render"
    assert out["point_count"] > 0 and out["verified_point_count"] == out["point_count"]
    assert out["point_sha256"] and out["surface_id"]
    assert out["run_id"] == chain.state.stages[Stage.TRAIN].outputs["run_id"]


def test_a_surface_that_does_not_promote_stops_the_stage(chain, tmp_path):
    """T4: the stage calls rederive_depth_source, so a demoted surface is a stage failure."""
    from minegs.eval.surface.depth import rederive_depth_source
    from minegs.eval.surface.models import load_surface

    surface_dir = Path(chain.state.stages[Stage.SURFACE].outputs["surface_dir"])
    dataset_dir = chain.state.stages[Stage.DATASET].outputs["dataset_dir"]
    surface, _ = load_surface(surface_dir)
    assert rederive_depth_source(surface, dataset_dir) is None

    gone = dict(surface.parameters)
    gone["depth_dir"] = str(tmp_path / "not_here")
    surface.parameters = gone
    assert "cannot be re-derived" in (rederive_depth_source(surface, dataset_dir) or "")


def test_a_failed_train_stage_is_not_a_run_the_depth_stage_will_touch(
    staging_small, build_config_small, tmp_path
):
    """T3: DEPTH needs TRAIN usable, and a trainer that does not finish leaves it failed."""
    cfg = e2e_config(
        staging_dir=str(staging_small.staging_dir),
        build_config=str(write_build_config(tmp_path, build_config_small, tmp_path / "c.json")),
        dataset_dir=str(tmp_path / "dataset"),
    )

    def broken(run_cfg, runner_cfg):
        run_dir = Path(run_cfg.run_dir)
        make_run(Path(run_cfg.dataset_dir), run_dir, run_id=run_dir.name, status=RunStatus.FAILED)
        return run_dir

    specs = {
        Stage.INGEST: ingest_spec(),
        Stage.DATASET: dataset_spec(),
        Stage.TRAIN: train_spec(),
        Stage.DEPTH: depth_spec(),
    }
    wf = Workflow(tmp_path / "wf", cfg)
    with pytest.raises(ContractError, match="not succeeded"):
        wf.execute(specs, through=Stage.DEPTH, trainer=broken, renderer=StandInRenderer())

    state = WorkflowState.load(tmp_path / "wf" / "workflow.json")
    assert state.stages[Stage.TRAIN].status is StageStatus.FAILED
    assert state.stages[Stage.DEPTH].status is StageStatus.PENDING


def test_a_retrained_run_makes_the_depth_and_surface_stages_stale(chain):
    """T2: depth hangs off the run id and the checkpoint digest, both re-read from disk."""
    run_dir = Path(chain.state.stages[Stage.TRAIN].outputs["run_dir"])
    checkpoint = run_dir / chain.state.stages[Stage.TRAIN].outputs["final_checkpoint"]
    keep = checkpoint.read_bytes()
    try:
        checkpoint.write_bytes(keep + b"another run's weights")
        wf = Workflow(chain.root / "wf")
        # ...and it cascades: the surface, the geometry and the comparison all rest on it
        assert wf.stale_stages(chain.specs) == [
            Stage.DEPTH,
            Stage.SURFACE,
            Stage.GEOMETRY,
            Stage.SECTIONS_VOLUME,
            Stage.REPORT,
        ]
        with pytest.raises(ContractError, match="checkpoint_sha256"):
            wf.execute(chain.specs, through=Stage.SURFACE)
    finally:
        checkpoint.write_bytes(keep)


# ---------------------------------------------------------------- C4: paired TLS validation


def paired_record(areas, *, section_id, kind="raw_cloud", interval_m=1.0, angle_bins=4, **over):
    """A section artifact on a fixed grid, for comparing two of them directly."""
    from minegs.core.provenance import ProvenanceRecord
    from minegs.eval.sections import SectionRecord, SectionSource

    source = (
        SectionSource(kind="raw_cloud", point_sha256="a" * 64, point_path="/tmp/ref.ply")
        if kind == "raw_cloud"
        else SectionSource(
            kind="surface",
            surface_id="surface_x",
            run_id="run_x",
            depth_source="minegs_render",
            point_sha256="b" * 64,
            point_path="/tmp/surface",
        )
    )
    fields = {
        "dataset_id": "ds",
        "dataset_hash": "h" * 64,
        "reference_axis": "centerline:design:centerline.csv",
        "reference_axis_sha256": "c" * 64,
        **over,
    }
    return SectionRecord(
        section_id=section_id,
        source=source,
        series=_series(areas, interval_m=interval_m, angle_bins=angle_bins),
        parameters={"interval_m": interval_m, "start_m": None, "end_m": None},
        provenance=ProvenanceRecord(),
        **fields,
    )


def test_two_series_cut_on_different_grids_are_not_comparable():
    """T6: 'the area at 22 m' means different things on two grids."""
    pred = paired_record([10.0] * 5, section_id="p", kind="surface")
    for field, value in (
        ("reference_axis", "centerline:extracted:other.csv"),
        ("dataset_hash", "d" * 64),
    ):
        ref = paired_record([10.0] * 5, section_id="r", **{field: value})
        with pytest.raises(ContractError, match=f"disagree about {field}"):
            compare_to_reference(pred, ref)

    coarse = paired_record([10.0] * 3, section_id="r", interval_m=2.0)
    with pytest.raises(ContractError, match="disagree about interval_m"):
        compare_to_reference(pred, coarse)

    short = paired_record([10.0] * 4, section_id="r")
    with pytest.raises(ContractError, match="different chainages"):
        compare_to_reference(pred, short)


def test_section_errors_are_paired_station_by_station():
    pred = paired_record([12.0, 11.0, 10.0, 9.0, 8.0], section_id="p", kind="surface")
    ref = paired_record([10.0, 10.0, 10.0, 10.0, 10.0], section_id="r")
    v = compare_to_reference(pred, ref)

    s = v.sections
    assert s.station_count == 5 and s.paired_valid_count == 5
    assert [st.signed_error_m2 for st in s.stations] == [2.0, 1.0, 0.0, -1.0, -2.0]
    assert s.mean_absolute_error_m2 == pytest.approx(1.2)
    assert s.median_absolute_error_m2 == pytest.approx(1.0)
    assert s.mean_signed_error_m2 == pytest.approx(0.0)
    assert s.mean_relative_error == pytest.approx(0.12)


def test_a_zero_reference_area_is_reported_not_divided_by():
    pred = paired_record([10.0, 10.0, 10.0], section_id="p", kind="surface")
    ref = paired_record([10.0, 0.0, 10.0], section_id="r")
    v = compare_to_reference(pred, ref)

    zero = v.sections.stations[1]
    assert zero.absolute_error_m2 == pytest.approx(10.0)
    assert zero.relative_error is None
    assert "ratio is undefined" in zero.relative_error_reason
    assert v.sections.relative_error_station_count == 2  # the other two


def test_a_station_only_one_side_observed_is_not_paired():
    pred = paired_record([10.0, None, 10.0, 10.0], section_id="p", kind="surface")
    ref = paired_record([10.0, 10.0, None, 10.0], section_id="r")
    v = compare_to_reference(pred, ref)

    assert v.sections.paired_valid_count == 2
    assert v.sections.missing_prediction_count == 1
    assert v.sections.missing_reference_count == 1
    assert v.sections.stations[1].absolute_error_m2 is None
    assert v.sections.stations[1].ref_area_m2 == pytest.approx(10.0)


def test_t7_neither_volume_is_integrated_across_a_missing_station():
    """areas 10,10,-,10,10 on one side: two segments, and the reference follows the same cut."""
    pred = paired_record([10.0, 10.0, None, 10.0, 10.0], section_id="p", kind="surface")
    ref = paired_record([12.0, 12.0, 12.0, 12.0, 12.0], section_id="r")
    v = compare_to_reference(pred, ref)

    assert v.volume.integrated_intervals_m == [(0.0, 1.0), (3.0, 4.0)]
    assert v.volume.missing_intervals_m == [(1.0, 3.0)]
    assert v.volume.predicted_volume_m3 == pytest.approx(20.0)
    assert v.volume.reference_volume_m3 == pytest.approx(24.0)  # over the same 2 m, not 4 m
    assert v.volume.absolute_error_m3 == pytest.approx(4.0)
    assert v.volume.coverage_fraction == pytest.approx(0.5)


def test_t8_the_two_volumes_are_compared_over_the_common_domain_only():
    """The rule the module exists for: differing coverages are not an error term.

    The prediction reaches further than the reference. Integrating each over its own span and
    subtracting would report 10 m3 of 'error' that is really 1 m of tunnel only one side saw.
    """
    pred = paired_record([10.0] * 5, section_id="p", kind="surface")
    ref = paired_record([10.0, 10.0, 10.0, 10.0, None], section_id="r")
    v = compare_to_reference(pred, ref)

    assert v.volume.integrated_intervals_m == [(0.0, 3.0)]
    assert v.volume.predicted_volume_m3 == pytest.approx(30.0)
    assert v.volume.reference_volume_m3 == pytest.approx(30.0)
    assert v.volume.absolute_error_m3 == pytest.approx(0.0)  # not 10.0
    assert v.volume.prediction_only_intervals_m == [(3.0, 4.0)]
    assert v.volume.reference_only_intervals_m == []
    assert "common domain only" in " ".join(v.volume.notes)


def test_coverage_is_measured_against_the_requested_holdout():
    """A reconstruction that reaches half the holdout has half the coverage."""
    pred = paired_record([10.0] * 11, section_id="p", kind="surface")
    ref = paired_record([10.0] * 5 + [None] * 6, section_id="r")
    v = compare_to_reference(pred, ref, ranges=[(0.0, 8.0)])

    assert v.volume.requested_length_m == pytest.approx(8.0)
    assert v.volume.common_covered_length_m == pytest.approx(4.0)
    assert v.volume.coverage_fraction == pytest.approx(0.5)
    assert v.volume.missing_intervals_m == [(4.0, 8.0)]


def test_nothing_observed_by_both_sides_is_not_a_zero_error():
    pred = paired_record([10.0, None, 10.0], section_id="p", kind="surface")
    ref = paired_record([None, 10.0, None], section_id="r")
    v = compare_to_reference(pred, ref)

    assert v.volume.predicted_volume_m3 is None and v.volume.reference_volume_m3 is None
    assert v.volume.absolute_error_m3 is None and v.volume.relative_error is None
    assert "nothing the two volumes could be compared over" in " ".join(v.volume.notes)


def test_a_zero_reference_volume_is_reported_not_divided_by():
    pred = paired_record([10.0, 10.0], section_id="p", kind="surface")
    ref = paired_record([0.0, 0.0], section_id="r")
    v = compare_to_reference(pred, ref)

    assert v.volume.reference_volume_m3 == pytest.approx(0.0)
    assert v.volume.absolute_error_m3 == pytest.approx(10.0)
    assert v.volume.relative_error is None
    assert "ratio is undefined" in v.volume.relative_error_reason


def test_the_sections_volume_stage_compares_the_run_against_held_out_tls(chain):
    out = chain.state.stages[Stage.SECTIONS_VOLUME].outputs
    holdout = chain.state.stages[Stage.DATASET].outputs["geometry_holdout_ranges_m"]

    assert out["requested_ranges_m"] == holdout  # judge()'s, not the runner's
    assert out["grid"]["interval_m"] == 1.0 and out["grid"]["angle_bins"] == 72
    assert out["paired_valid_count"] > 0
    assert out["predicted_volume_m3"] is not None and out["reference_volume_m3"] is not None
    assert out["absolute_error_m3"] == pytest.approx(
        abs(out["predicted_volume_m3"] - out["reference_volume_m3"])
    )
    for key in ("sections_predicted_path", "sections_reference_path", "paired_validation_path"):
        assert Path(out[key]).is_file()


def test_the_reference_sections_are_recorded_as_the_raw_cloud_they_are(chain):
    """Dressing the TLS reference up as a reconstruction would make its record say something
    false about where it came from."""
    from minegs.eval.sections import SectionRecord

    out = chain.state.stages[Stage.SECTIONS_VOLUME].outputs
    ref = SectionRecord.load(out["sections_reference_path"])
    pred = SectionRecord.load(out["sections_predicted_path"])

    assert ref.source.kind == "raw_cloud" and not ref.supports_accuracy_claim
    assert pred.source.kind == "surface" and pred.source.depth_source == "minegs_render"
    assert pred.source.run_id == chain.state.stages[Stage.TRAIN].outputs["run_id"]


def test_the_geometry_stage_carries_both_directions_and_the_real_claim(chain):
    out = chain.state.stages[Stage.GEOMETRY].outputs
    holdout = chain.state.stages[Stage.DATASET].outputs["geometry_holdout_ranges_m"]

    assert out["claim"] == "geometry_accuracy"
    assert out["chainage_range_m"] == [holdout[0][0], holdout[-1][1]]
    for key in (
        "accuracy_median_m",
        "accuracy_p95_m",
        "completeness_median_m",
        "completeness_p95_m",
        "chamfer_m",
    ):
        assert out[key] is not None
    assert out["f_score"] and Path(out["report_path"]).is_file()


def test_t5_the_geometry_numbers_are_measured_inside_the_holdout_only(chain):
    """The stage does not offer --no-holdout-only: a claim path does not get to choose."""
    import json as _json

    rep = _json.loads(Path(chain.state.stages[Stage.GEOMETRY].outputs["report_path"]).read_text())
    holdout = chain.state.stages[Stage.DATASET].outputs["geometry_holdout_ranges_m"]
    assert rep["chainage_range_m"] == [holdout[0][0], holdout[-1][1]]
    assert rep["claim"] == "geometry_accuracy"
    command = chain.state.stages[Stage.GEOMETRY].command
    assert command["holdout_only"] is True and command["diagnostic"] is False


# ---------------------------------------------------------------- C5: the report


def test_the_report_stage_writes_both_documents(chain):
    out = chain.state.stages[Stage.REPORT].outputs
    assert Path(out["report_json_path"]).is_file()
    assert Path(out["report_md_path"]).is_file()
    assert Path(out["report_json_path"]).name == "phase2_report.json"
    assert Path(out["report_md_path"]).name == "phase2_report.md"
    report = Phase2Report.load(out["report_json_path"])
    assert report.report_id == out["report_id"]
    assert report.workflow_id == chain.state.workflow_id


def test_the_report_copies_its_numbers_and_recomputes_none(chain):
    """A report that recomputes is a second implementation, and a second answer."""
    sv = chain.state.stages[Stage.SECTIONS_VOLUME].outputs
    paired = json.loads(Path(sv["paired_validation_path"]).read_text())
    report = Phase2Report.load(chain.state.stages[Stage.REPORT].outputs["report_json_path"])

    for field in (
        "predicted_volume_m3",
        "reference_volume_m3",
        "signed_error_m3",
        "absolute_error_m3",
        "relative_error",
        "requested_length_m",
        "common_covered_length_m",
        "coverage_fraction",
    ):
        # Equality, not approximate equality: these are copies, so any difference at all means
        # something along the way did arithmetic of its own.
        assert getattr(report.volume, field) == paired["volume"][field], field
    for field in (
        "station_count",
        "paired_valid_count",
        "missing_prediction_count",
        "missing_reference_count",
        "mean_absolute_error_m2",
        "median_absolute_error_m2",
        "p95_absolute_error_m2",
        "mean_signed_error_m2",
        "mean_relative_error",
    ):
        assert getattr(report.sections, field) == paired["sections"][field], field

    geometry = chain.state.stages[Stage.GEOMETRY].outputs
    assert report.geometry.chamfer_m == geometry["chamfer_m"]
    assert report.geometry.accuracy_p95_m == geometry["accuracy_p95_m"]
    assert report.geometry.claim == geometry["claim"] == "geometry_accuracy"


def test_the_markdown_is_a_rendering_of_the_json_and_not_a_second_one(chain):
    from minegs.e2e.report import render_markdown

    out = chain.state.stages[Stage.REPORT].outputs
    report = Phase2Report.load(out["report_json_path"])
    assert Path(out["report_md_path"]).read_text() == render_markdown(report)


def test_the_report_never_upgrades_a_maturity_status(chain):
    """Whatever the numbers look like, the workflow does not get to validate itself."""
    from minegs.e2e.report import MATURITY_PENDING

    report = Phase2Report.load(chain.state.stages[Stage.REPORT].outputs["report_json_path"])
    assert report.maturity.structural_status == "implemented_and_structurally_tested"
    assert report.maturity.real_data_validation_status == "not_validated"
    assert report.maturity.human_visual_review_status == "pending"
    assert report.maturity.statement == MATURITY_PENDING
    assert "NOT VALIDATED" in report.maturity.statement
    assert "PENDING" in report.maturity.statement


def test_the_report_records_both_substitutions_rather_than_the_appearance_of_a_run(chain):
    report = Phase2Report.load(chain.state.stages[Stage.REPORT].outputs["report_json_path"])
    md = Path(chain.state.stages[Stage.REPORT].outputs["report_md_path"]).read_text()

    assert report.training.real_gpu_execution is False
    assert report.reconstruction.real_renderer_execution is False
    assert report.training.gpu_model is None and report.training.cuda_version is None
    assert any("substituted" in n for n in report.training.notes)
    assert any("GsplatDepthRenderer.render" in n for n in report.reconstruction.notes)
    assert any("GsplatDepthRenderer.render" in n for n in report.maturity.notes)
    # The reader sees it beside the maturity statement, not buried in a JSON field.
    assert "real GPU training executed: **False**" in md
    assert "real depth renderer executed: **False**" in md


def test_the_report_states_the_identities_its_numbers_rest_on(chain):
    report = Phase2Report.load(chain.state.stages[Stage.REPORT].outputs["report_json_path"])
    dataset = chain.state.stages[Stage.DATASET].outputs
    train = chain.state.stages[Stage.TRAIN].outputs
    surface = chain.state.stages[Stage.SURFACE].outputs

    assert report.dataset.dataset_id == dataset["dataset_id"]
    assert report.dataset.dataset_hash == dataset["dataset_hash"]
    assert report.dataset.golden_gate_passed is True
    assert report.dataset.centerline_length_m == dataset["centerline_length_m"] > 0
    assert report.dataset.geometry_holdout_ranges_m == [
        tuple(r) for r in dataset["geometry_holdout_ranges_m"]
    ]
    assert report.training.run_id == train["run_id"]
    assert report.reconstruction.surface_id == surface["surface_id"]
    assert report.reconstruction.depth_source == "minegs_render"
    assert report.source.sha256 or report.source.notes  # adopted staging says so in notes


def test_a_value_the_report_cannot_find_is_null_with_a_reason():
    """An invented number travels further from a report than from anywhere else."""
    from minegs.e2e.report import build_report, render_markdown

    report = build_report(blank_state())

    assert report.volume.predicted_volume_m3 is None
    assert report.sections.paired_valid_count is None
    assert report.geometry.chamfer_m is None
    assert report.dataset.dataset_id is None and report.runtime.total_seconds is None
    assert report.maturity.structural_status == "incomplete"
    assert any("ingest" in n for n in report.maturity.notes)
    assert any("unidentified" in n for n in report.source.notes)
    # And the rendering says so rather than printing a zero.
    md = render_markdown(report)
    assert "- predicted: —" in md and "Chamfer: —" in md


def test_the_report_names_every_stage_that_did_not_complete():
    from minegs.e2e.report import build_report

    state = blank_state(
        stages={
            Stage.INGEST: StageRecord(stage=Stage.INGEST, status=StageStatus.SUCCEEDED),
            Stage.DATASET: StageRecord(
                stage=Stage.DATASET,
                status=StageStatus.FAILED,
                failure_reason="ContractError: golden gate",
            ),
        }
    )
    report = build_report(state)

    note = "\n".join(report.maturity.notes)
    assert "dataset=failed" in note and "train=pending" in note
    assert "ingest=" not in note  # it completed; only the missing ones are named
    rows = {row["stage"]: row for row in report.stages}
    assert rows["dataset"]["failure_reason"] == "ContractError: golden gate"
    assert [row["stage"] for row in report.stages] == [s.value for s in STAGE_ORDER]


def test_regenerating_the_report_runs_no_stage(chain, tmp_path):
    """Rewriting a heading must not cost an E57 extraction or a training run."""
    from minegs.e2e.report import report_from_workflow

    before = json.loads((chain.root / "wf" / "workflow.json").read_text())
    path = report_from_workflow(chain.root / "wf", tmp_path / "again")
    after = json.loads((chain.root / "wf" / "workflow.json").read_text())

    assert path.is_file() and (tmp_path / "again" / "phase2_report.md").is_file()
    # The ledger is untouched: nothing ran, nothing was marked reused, no time was spent.
    assert before["stages"] == after["stages"]
    regenerated = Phase2Report.load(path)
    original = Phase2Report.load(chain.state.stages[Stage.REPORT].outputs["report_json_path"])
    assert regenerated.volume.model_dump() == original.volume.model_dump()
    assert regenerated.sections.model_dump() == original.sections.model_dump()
    assert regenerated.report_id != original.report_id  # a new document, not a forged copy


def test_there_is_nothing_to_report_on_without_a_workflow(tmp_path):
    from minegs.e2e.report import report_from_workflow

    with pytest.raises(ContractError, match="no workflow there"):
        report_from_workflow(tmp_path)


def test_an_edited_paired_validation_makes_the_report_stale(chain, tmp_path):
    """The report reads that file, so a change to it is a change to the report's inputs."""
    from minegs.e2e.report import build_report

    root = tmp_path / "restale"
    root.mkdir()
    state_path = root / "workflow.json"
    state_path.write_text((chain.root / "wf" / "workflow.json").read_text())
    wf = Workflow(root)
    paired = Path(chain.state.stages[Stage.SECTIONS_VOLUME].outputs["paired_validation_path"])
    original = paired.read_text()
    try:
        edited = json.loads(original)
        edited["volume"]["predicted_volume_m3"] = 1.0
        paired.write_text(json.dumps(edited))
        assert Stage.REPORT in wf.stale_stages(default_specs())
        with pytest.raises(ContractError, match="stage report completed against different"):
            wf.execute(
                default_specs(),
                renderer=StandInRenderer(depth_m=7.0),
                trainer=stand_in_trainer,
            )
        # And the report the edit was trying to reach refuses it outright. This assertion used
        # to say the opposite -- that `build_report` read the edited 1.0 back -- which is
        # exactly the hole an independent review found: `minegs e2e report` runs no stage, so
        # it reached this call without passing the staleness gate above, and published a Phase
        # 2 report carrying the edited volume. The report now checks the file against the
        # digest the stage recorded for it.
        with pytest.raises(ContractError, match="not the numbers this workflow produced"):
            build_report(wf.state)
    finally:
        paired.write_text(original)


def test_the_report_stage_refuses_a_paired_validation_that_is_not_there(chain, tmp_path):
    from minegs.e2e.stages import report_spec

    ctx = StageContext(
        stage=Stage.REPORT,
        config=chain.config,
        work_dir=tmp_path,
        state=chain.state,
    )
    paired = Path(chain.state.stages[Stage.SECTIONS_VOLUME].outputs["paired_validation_path"])
    original = paired.read_text()
    try:
        paired.unlink()
        with pytest.raises(ContractError, match="is not there"):
            report_spec().inputs(ctx)
    finally:
        paired.write_text(original)


def test_the_runtime_table_sums_only_the_stages_this_workflow_ran(chain):
    """Every stage that ran, the report stage included.

    It used to be excluded because it could not be known: the document was written from inside
    the stage, before the stage had an elapsed time. The finalising write happens after the
    record is closed, so the total is now the whole workflow rather than the whole workflow
    minus the part that was writing the number down.
    """
    report = Phase2Report.load(chain.state.stages[Stage.REPORT].outputs["report_json_path"])
    measured = [
        chain.state.stages[s].elapsed_seconds
        for s in STAGE_ORDER
        if chain.state.stages[s].elapsed_seconds is not None
    ]
    assert chain.state.stages[Stage.REPORT].elapsed_seconds is not None
    assert report.runtime.total_seconds == pytest.approx(sum(measured), abs=1e-6)
    assert report.runtime.dataset_seconds == chain.state.stages[Stage.DATASET].elapsed_seconds


def test_a_partial_workflow_is_still_reported_not_refused(chain, tmp_path):
    """The new currency gate is about artifacts that moved, not about stages that never ran.

    A partial report is what an operator wants to look at after a failure, and refusing to
    build one would make the command useless exactly when it matters.
    """
    from minegs.e2e.report import report_from_workflow

    root = tmp_path / "partial"
    root.mkdir()
    state = WorkflowState.load(chain.root / "wf" / "workflow.json")
    for stage in (Stage.GEOMETRY, Stage.SECTIONS_VOLUME, Stage.REPORT):
        state.stages[stage] = StageRecord(stage=stage)
    state.save(root / "workflow.json")

    report = Phase2Report.load(report_from_workflow(root, tmp_path / "out"))

    assert report.maturity.structural_status == "incomplete"
    assert any("sections_volume=pending" in n for n in report.maturity.notes)
    assert report.volume.predicted_volume_m3 is None
    assert report.reconstruction.surface_id  # what did complete is still reported


def test_a_report_that_cannot_be_finalised_fails_the_stage(chain, tmp_path):
    """The finalising write is part of the stage, so its failure is the stage's failure.

    A ledger claiming a stage succeeded when the artifact that stage exists to produce was
    never finished is the one outcome worse than a stage that says it stopped.
    """
    from minegs.e2e.stages import report_spec

    root = tmp_path / "wf"
    root.mkdir()
    (root / "workflow.json").write_text((chain.root / "wf" / "workflow.json").read_text())
    wf = Workflow(root)
    wf.state.stages[Stage.REPORT] = StageRecord(stage=Stage.REPORT)
    wf.save()

    def explode(ctx, outcome):
        raise ContractError("the disk filled up")

    specs = dict(default_specs())
    spec = report_spec()
    specs[Stage.REPORT] = StageSpec(
        stage=spec.stage, inputs=spec.inputs, run=spec.run, finalise=explode
    )
    with pytest.raises(ContractError, match="the disk filled up"):
        wf.execute(specs, renderer=StandInRenderer(depth_m=7.0), trainer=stand_in_trainer)

    reloaded = WorkflowState.load(root / "workflow.json")
    assert reloaded.stages[Stage.REPORT].status is StageStatus.FAILED
    assert "the disk filled up" in reloaded.stages[Stage.REPORT].failure_reason
