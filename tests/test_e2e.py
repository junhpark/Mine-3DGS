"""Phase 2 — the end-to-end contract (§Phase 2).

C0 freezes the shape before anything executes: the stage graph, the ledger that makes resuming
safe, and the report that may not say more than its artifacts do. The orchestration that fills
them in follows in later checkpoints; what is asserted here is the contract they have to keep.
"""

from __future__ import annotations

import pytest
from minegs.core.provenance import ProvenanceRecord
from minegs.e2e.models import (
    STAGE_ORDER,
    MaturitySummary,
    Phase2Report,
    Stage,
    StageRecord,
    StageStatus,
    WorkflowState,
)


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
