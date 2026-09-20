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
    ingest_spec,
    staging_digest,
)

from e57_fakes import make_scan, pose_node


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

    world["source"] = "e57_b"
    assert Workflow(tmp_path / "wf").stale_stages(chain_specs([], world)) == [Stage.INGEST]


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
