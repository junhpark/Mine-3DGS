"""The Phase 2 end-to-end contract: workflow state and the scientific report (§Phase 2).

Phase 0A–1C each closed one boundary. Nothing so far carries a *survey* from the E57 on disk to
a report a reviewer can audit: the operator runs eight commands by hand, remembers which output
feeds which input, and nothing records that the surface being evaluated came from the run that
was trained on the dataset that was built from that E57.

Two artifacts fix that, and neither of them is allowed to contain any science of its own.

``WorkflowState`` is the ledger. One record per stage: what it consumed, what it produced, how
long it took, and on which commit of which minegs. It is what makes the workflow resumable —
and, more importantly, what makes resuming *safe*. A stage may be reused only when the
fingerprint of its inputs still matches the world, because a SUCCEEDED stage whose inputs have
since changed is not a shortcut, it is stale evidence with a green tick on it.

``Phase2Report`` is the aggregation. Every number in it is copied from an artifact that already
verified itself; the report recomputes nothing and may not upgrade anything. A field it cannot
fill is ``None`` with a reason, never a plausible value — a report is the one place where an
invented number would travel furthest.

What neither of them does: grant a claim. ``Claim`` (``minegs/eval/protocol.py``) is unchanged
by Phase 2, and the evidence chain stays exactly as Phase 1C left it —

    SUCCEEDED run → verified minegs_render depth → verified SurfaceRecord
      → verified SectionRecord → gap-safe integration → volume_accuracy

— with the orchestrator calling the same validators as the CLI rather than vouching for its own
output. "This stage produced it, so it is trusted" is the one thing an orchestrator must never
be allowed to say.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from minegs.core.config import VersionedModel
from minegs.core.provenance import ProvenanceRecord

__all__ = [
    "STAGE_ORDER",
    "DatasetSummary",
    "GeometrySummary",
    "MaturitySummary",
    "Phase2Report",
    "ReconstructionSummary",
    "RuntimeSummary",
    "SectionSummary",
    "SourceSummary",
    "Stage",
    "StageRecord",
    "StageStatus",
    "TrainingSummary",
    "VolumeSummary",
    "WorkflowState",
]

WORKFLOW_STATE_FILE = "workflow.json"
REPORT_JSON_FILE = "phase2_report.json"
REPORT_MD_FILE = "phase2_report.md"


class Stage(str, Enum):
    """The eight steps of the survey, in the only order they can happen in."""

    INGEST = "ingest"
    DATASET = "dataset"
    TRAIN = "train"
    DEPTH = "depth"
    SURFACE = "surface"
    GEOMETRY = "geometry"
    SECTIONS_VOLUME = "sections_volume"
    REPORT = "report"


#: Declared once so the runner, the state and the docs cannot drift apart.
STAGE_ORDER: tuple[Stage, ...] = tuple(Stage)


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    #: Reused from a previous execution whose input fingerprint still matches. Distinct from
    #: SUCCEEDED so the report can say what this workflow actually ran.
    REUSED = "reused"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StageRecord(_Strict):
    """One stage's execution, and the fingerprint that decides whether it may be reused.

    ``input_fingerprint`` is the whole safety argument. It is computed from the identities the
    stage actually depends on — the E57 digest, the dataset hash, the run id, the surface's
    point digest — and recomputed from the world before any reuse. Same fingerprint means the
    stage would do the same work again, so skipping it changes nothing. A different one means
    something upstream moved, and the recorded outputs describe a survey that no longer exists.
    """

    stage: Stage
    status: StageStatus = StageStatus.PENDING
    started_at: str | None = None
    completed_at: str | None = None
    elapsed_seconds: float | None = None
    #: sha256 over the identities this stage consumed; see ``minegs/e2e/runner.py``.
    input_fingerprint: str | None = None
    #: The identities the fingerprint was built from, in the open, so a mismatch can be read.
    inputs: dict[str, Any] = Field(default_factory=dict)
    #: Artifact ids, digests and paths this stage produced. What downstream stages consume.
    outputs: dict[str, Any] = Field(default_factory=dict)
    #: The call that produced it — enough to run the stage again by hand.
    command: dict[str, Any] = Field(default_factory=dict)
    git_commit: str = "unknown"
    minegs_version: str = ""
    tool_versions: dict[str, str] = Field(default_factory=dict)
    #: Best effort, and null with a reason rather than invented (§P2-C3).
    runtime_env: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None

    @property
    def usable(self) -> bool:
        """Whether a downstream stage may build on this one's outputs."""
        return self.status in (StageStatus.SUCCEEDED, StageStatus.REUSED)


class WorkflowState(VersionedModel):
    """``<workflow_dir>/workflow.json`` — the ledger, and the only thing resume reads."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    workflow_id: str = Field(min_length=1)
    #: The survey this workflow is of. Set by the INGEST stage and never rewritten: a workflow
    #: is about one E57, and pointing an existing ledger at another file is a new workflow.
    source_sha256: str | None = None
    created_at: str
    updated_at: str
    config: dict[str, Any] = Field(default_factory=dict)
    stages: dict[Stage, StageRecord] = Field(default_factory=dict)
    provenance: ProvenanceRecord

    @model_validator(mode="after")
    def _every_stage_present(self) -> WorkflowState:
        """A ledger with a stage missing cannot say whether it ran, so it names all of them."""
        missing = [s.value for s in STAGE_ORDER if s not in self.stages]
        if missing:
            raise ValueError(f"workflow state is missing stage records for {missing}")
        wrong = [s.value for s, rec in self.stages.items() if rec.stage != s]
        if wrong:
            raise ValueError(f"stage records filed under the wrong key: {wrong}")
        return self

    def record(self, stage: Stage) -> StageRecord:
        return self.stages[stage]

    def done_through(self) -> list[Stage]:
        """The leading run of stages that may be built on, in order."""
        out: list[Stage] = []
        for s in STAGE_ORDER:
            if not self.stages[s].usable:
                break
            out.append(s)
        return out


# ---------------------------------------------------------------- the report


class SourceSummary(_Strict):
    """The survey the whole chain rests on."""

    file_name: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    #: What the E57 says about itself — writer, coordinate metadata, scan count.
    capture_metadata: dict[str, Any] = Field(default_factory=dict)
    #: Why a field above is null, when it is. Never a guess in its place.
    notes: list[str] = Field(default_factory=list)


class DatasetSummary(_Strict):
    dataset_id: str | None = None
    dataset_hash: str | None = None
    station_count: int | None = None
    image_count: int | None = None
    init_point_count: int | None = None
    #: The evaluated length is the holdout, not the drift: it is what the claim is about.
    evaluated_length_m: float | None = None
    centerline_length_m: float | None = None
    geometry_holdout_ranges_m: list[tuple[float, float]] = Field(default_factory=list)
    #: The measured convention, copied from the calibration evidence. Never a default.
    camera_convention: dict[str, Any] = Field(default_factory=dict)
    protocol: list[str] = Field(default_factory=list)
    claims_allowed: list[str] = Field(default_factory=list)
    golden_gate_passed: bool | None = None
    notes: list[str] = Field(default_factory=list)


class TrainingSummary(_Strict):
    run_id: str | None = None
    backend: str | None = None
    backend_version: str | None = None
    profile: str | None = None
    steps: int | None = None
    checkpoint_step: int | None = None
    git_commit: str | None = None
    #: null with a reason when the environment did not say; see ``notes``.
    gpu_model: str | None = None
    cuda_version: str | None = None
    runner: str | None = None
    #: True only when a real CUDA device ran a real backend. A substituted backend says False.
    real_gpu_execution: bool = False
    notes: list[str] = Field(default_factory=list)


class ReconstructionSummary(_Strict):
    depth_manifest_id: str | None = None
    renderer: str | None = None
    renderer_version: str | None = None
    depth_map_count: int | None = None
    mean_depth_valid_ratio: float | None = None
    surface_id: str | None = None
    surface_point_count: int | None = None
    depth_source: str | None = None
    #: True only when ``GsplatDepthRenderer.render`` itself produced the depth.
    real_renderer_execution: bool = False
    notes: list[str] = Field(default_factory=list)


class GeometrySummary(_Strict):
    """Copied from ``GeometryReport``. Both directions, because one of them hides holes."""

    claim: str | None = None
    chainage_range_m: tuple[float, float] | None = None
    max_dist_m: float | None = None
    accuracy_median_m: float | None = None
    accuracy_p95_m: float | None = None
    completeness_median_m: float | None = None
    completeness_p95_m: float | None = None
    chamfer_m: float | None = None
    f_score: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class SectionSummary(_Strict):
    """Predicted sections against held-out TLS sections, station by station."""

    interval_m: float | None = None
    thickness_m: float | None = None
    angle_bins: int | None = None
    requested_ranges_m: list[tuple[float, float]] = Field(default_factory=list)
    station_count: int | None = None
    paired_valid_count: int | None = None
    missing_prediction_count: int | None = None
    missing_reference_count: int | None = None
    mean_absolute_error_m2: float | None = None
    median_absolute_error_m2: float | None = None
    p95_absolute_error_m2: float | None = None
    mean_signed_error_m2: float | None = None
    mean_relative_error: float | None = None
    notes: list[str] = Field(default_factory=list)


class VolumeSummary(_Strict):
    """Predicted against reference over the *common* integration domain, never otherwise."""

    predicted_volume_m3: float | None = None
    reference_volume_m3: float | None = None
    signed_error_m3: float | None = None
    absolute_error_m3: float | None = None
    relative_error: float | None = None
    requested_length_m: float | None = None
    common_covered_length_m: float | None = None
    coverage_fraction: float | None = None
    integrated_intervals_m: list[tuple[float, float]] = Field(default_factory=list)
    missing_intervals_m: list[tuple[float, float]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RuntimeSummary(_Strict):
    ingest_seconds: float | None = None
    dataset_seconds: float | None = None
    training_seconds: float | None = None
    depth_render_seconds: float | None = None
    surface_build_seconds: float | None = None
    evaluation_seconds: float | None = None
    total_seconds: float | None = None


class MaturitySummary(_Strict):
    """The three statements a reader needs before believing any number above them.

    ``structural_status`` is about the *code*: did the contract path run end to end.
    ``real_data_validation_status`` is about the *science*: was this a real survey on a real GPU.
    ``human_visual_review_status`` is about a person: nobody but an operator may set it to pass,
    and nothing in minegs writes anything but ``pending`` (§P2 §16).
    """

    structural_status: Literal["implemented_and_structurally_tested", "incomplete"] = "incomplete"
    real_data_validation_status: Literal["not_validated", "validated"] = "not_validated"
    human_visual_review_status: Literal["pending", "pass", "fail"] = "pending"
    statement: str = ""
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validated_needs_real_execution(self) -> MaturitySummary:
        """``validated`` is not a field a workflow may set about itself.

        The report builder never passes it; it exists so that a hand-edited report claiming
        validation at least has to also claim a human looked, which is the condition the phase
        contract puts on it (§P2 §17).
        """
        if self.real_data_validation_status == "validated" and (
            self.human_visual_review_status != "pass"
        ):
            raise ValueError(
                "real_data_validation_status=validated requires human_visual_review_status=pass; "
                "G2 is not passed by numbers alone (§Phase 2)"
            )
        return self


class Phase2Report(VersionedModel):
    """``<out>/phase2_report.json`` — the machine-readable source of truth.

    The Markdown beside it is a rendering of *this*, never a second computation: two documents
    that each work out the volume error are two documents that can disagree about it.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    report_id: str = Field(min_length=1)
    workflow_id: str = Field(min_length=1)
    generated_at: str
    source: SourceSummary = Field(default_factory=SourceSummary)
    dataset: DatasetSummary = Field(default_factory=DatasetSummary)
    training: TrainingSummary = Field(default_factory=TrainingSummary)
    reconstruction: ReconstructionSummary = Field(default_factory=ReconstructionSummary)
    geometry: GeometrySummary = Field(default_factory=GeometrySummary)
    sections: SectionSummary = Field(default_factory=SectionSummary)
    volume: VolumeSummary = Field(default_factory=VolumeSummary)
    runtime: RuntimeSummary = Field(default_factory=RuntimeSummary)
    maturity: MaturitySummary = Field(default_factory=MaturitySummary)
    #: One entry per stage: status, timing and the identities it carried. The report's own
    #: provenance, so a reader can tell a reused stage from one this workflow ran.
    stages: list[dict[str, Any]] = Field(default_factory=list)
    provenance: ProvenanceRecord
