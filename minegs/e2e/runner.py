"""Ordering, state, artifact identity and timing — and nothing else (§Phase 2 C1).

The runner is deliberately thin. It does not know how to extract an E57, build a dataset, train
a model or fuse a surface; it knows the order those happen in, what each one consumed, and how
to refuse to skip one whose inputs have moved. Every stage handler calls the same domain
function the CLI calls, so there is exactly one implementation of each step in the repository
and the orchestrated path cannot drift from the hand-run one.

**Why a fingerprint and not a timestamp.** Resuming is only useful if it is also safe, and the
unsafe version is easy to write by accident: see a SUCCEEDED stage, skip it, carry on. That
reuses evidence about a survey that may no longer exist — a re-extracted E57, an edited build
config, a retrained run. So a stage records a digest of the identities it actually depended on,
and the runner recomputes that digest from the world before reusing anything. Matching means
running the stage again would do the same work; differing means the recorded outputs describe
something else, and the runner stops rather than quietly building on them.

The fingerprints chain, which is what makes one changed input invalidate everything after it:
``train``'s is built from the dataset hash, ``depth``'s from the run id and checkpoint digest,
``surface``'s from the depth manifest id. Change the E57 and every stage downstream of ingest
comes out stale on its own terms, without the runner having to know why.

This is not training resume. It means an already-succeeded TRAIN stage is not run twice; the
trainer's own ability to continue from a checkpoint is a different problem, still unimplemented
and still failing closed (Phase 0D.3).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from minegs.core.config import VersionedModel, config_hash
from minegs.core.errors import ContractError
from minegs.core.provenance import git_commit, make_id, stamp, tool_versions
from minegs.e2e.models import (
    STAGE_ORDER,
    WORKFLOW_STATE_FILE,
    Stage,
    StageRecord,
    StageStatus,
    WorkflowState,
)

__all__ = [
    "E2EConfig",
    "StageContext",
    "StageOutcome",
    "StageSpec",
    "Workflow",
    "fingerprint",
    "load_e2e_config",
    "now_iso",
    "stage_from_name",
    "stages_through",
]

import minegs


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(inputs: dict[str, Any]) -> str:
    """A stage's inputs as one digest. Canonical JSON, so key order cannot change it."""
    return config_hash(inputs)


class E2EConfig(VersionedModel):
    """Everything the workflow needs, in one file, so a rerun is a rerun of the same thing.

    Paths are the operator's; nothing here is defaulted to a location on any particular machine.
    A field left unset means the stage that needs it will say so when it runs, rather than the
    runner inventing a path early and failing somewhere less legible.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    #: The survey. Absent when starting from a staging tree that was extracted elsewhere.
    source_e57: str | None = None
    staging_dir: str | None = None
    dataset_dir: str | None = None
    runs_dir: str | None = None
    #: Dataset build config (YAML/JSON), as `minegs dataset from-e57 --config` takes it.
    build_config: str | None = None
    #: The held-out TLS cloud, in TLS_GLOBAL. The evaluation reference — never training input.
    tls_reference_ply: str | None = None

    # ---- ingest
    voxel_m: float | None = None
    mapping: str | None = None
    vendor_manifest: str | None = None
    images_dir: str | None = None
    max_scan_points: int | None = None
    scan_ids: list[str] = Field(default_factory=list)

    # ---- training
    profile: str = "light"
    backend: str = "gsplat"
    runner: str = "local"
    native: bool = False

    # ---- depth / surface
    min_alpha: float | None = None
    stride: int = 2
    max_depth_m: float | None = None

    # ---- evaluation
    max_dist_m: float = 1.0
    interval_m: float = 1.0
    thickness_m: float = 0.5
    angle_bins: int = 180

    #: Everything above that names a file or a directory. A config is a document an operator
    #: keeps beside its inputs, so its relative paths mean "next to me", not "next to wherever
    #: this was invoked from".
    PATH_FIELDS: ClassVar[tuple[str, ...]] = (
        "source_e57",
        "staging_dir",
        "dataset_dir",
        "runs_dir",
        "build_config",
        "tls_reference_ply",
        "mapping",
        "vendor_manifest",
        "images_dir",
    )

    def resolve_paths(self, base: str | Path) -> E2EConfig:
        """Make relative file references absolute against the config file's directory."""
        base = Path(base)
        cfg = self.model_copy(deep=True)
        for name in self.PATH_FIELDS:
            value = getattr(cfg, name)
            if not value:
                continue
            q = Path(value)
            setattr(cfg, name, str(q if q.is_absolute() else (base / q).resolve()))
        return cfg

    def require(self, *names: str) -> tuple[Any, ...]:
        """Fetch config values a stage cannot run without, naming all the missing ones at once."""
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            raise ContractError(
                f"the workflow config does not set {missing}; this stage cannot run without them"
            )
        return tuple(getattr(self, n) for n in names)


@dataclass
class StageOutcome:
    """What a handler produced. ``outputs`` is what downstream fingerprints are built from."""

    outputs: dict[str, Any] = field(default_factory=dict)
    command: dict[str, Any] = field(default_factory=dict)
    runtime_env: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageContext:
    """What a handler is given: the config, the workflow directory, and what came before."""

    stage: Stage
    config: E2EConfig
    work_dir: Path
    state: WorkflowState
    #: Injected seams. Both exist because the hardware does not, and both are refused entry to
    #: the evidence path: a substituted renderer mints a manifest naming a renderer this build
    #: does not ship, and a substituted trainer still has to leave a run the real validators
    #: accept. Neither is allowed to mean "trust this because we made it".
    renderer: Any = None
    trainer: Any = None
    #: True only when this execution's ``--rebuild-from`` covers this stage. A stage that
    #: publishes into a fixed directory reads it to decide whether it may replace what is
    #: there: an ordinary run never may, because an artifact nobody asked to destroy is
    #: evidence. Stating the intent is what makes the destruction visible.
    rebuilding: bool = False

    def upstream(self, stage: Stage) -> dict[str, Any]:
        """The outputs of an earlier stage, refusing if it has not produced any."""
        rec = self.state.stages[stage]
        if not rec.usable:
            raise ContractError(
                f"stage {self.stage.value} needs {stage.value}, which is {rec.status.value}"
            )
        return dict(rec.outputs)


@dataclass(frozen=True)
class StageSpec:
    """A stage, split into the question and the work.

    ``inputs`` is answered *before* deciding whether to run: it is the identity of everything the
    stage depends on, read from the world as it is now. ``run`` is only called when that identity
    differs from the recorded one, or when there is no recorded one.
    """

    stage: Stage
    inputs: Callable[[StageContext], dict[str, Any]]
    run: Callable[[StageContext], StageOutcome]
    #: Called once the record is final, for a stage whose artifact describes the ledger it is
    #: in. Only the report needs it: written from inside ``run`` it would have to guess its own
    #: outcome, and a document that predicts its result is exactly what this project does not
    #: write. Raising here fails the stage rather than leaving a success nobody completed.
    finalise: Callable[[StageContext, StageOutcome], None] | None = None


class Workflow:
    """The ledger and the loop over it."""

    def __init__(self, work_dir: str | Path, config: E2EConfig | None = None) -> None:
        self.work_dir = Path(work_dir)
        self.path = self.work_dir / WORKFLOW_STATE_FILE
        if self.path.is_file():
            self.state = WorkflowState.load(self.path)
            if config is not None and config.model_dump(mode="json") != self.state.config:
                # Silently adopting a new config would make every recorded fingerprint a
                # statement about settings that are no longer in force.
                raise ContractError(
                    f"{self.path} was created with a different workflow config. Re-running with "
                    "changed settings is a new workflow: point --work-dir at a new directory, or "
                    "rebuild the affected stages explicitly."
                )
        else:
            if config is None:
                raise ContractError(f"{self.path}: no workflow there, and no config to start one")
            self.state = _new_state(config)
            self.work_dir.mkdir(parents=True, exist_ok=True)
            self.save()

    @property
    def config(self) -> E2EConfig:
        return E2EConfig.from_dict(dict(self.state.config))

    def save(self) -> Path:
        self.state.updated_at = now_iso()
        return self.state.save(self.path)

    # ---------------------------------------------------------------- execution

    def _context(
        self,
        stage: Stage,
        renderer: Any = None,
        trainer: Any = None,
        rebuilding: bool = False,
    ) -> StageContext:
        return StageContext(
            stage=stage,
            config=self.config,
            work_dir=self.work_dir,
            state=self.state,
            renderer=renderer,
            trainer=trainer,
            rebuilding=rebuilding,
        )

    def resolve_inputs(
        self,
        stage: Stage,
        specs: dict[Stage, StageSpec],
        cache: dict[Stage, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """A stage's own inputs, plus the fingerprint of the stage before it.

        The chain link is what makes one changed identity invalidate *everything* after it. A
        stage only reads what it directly touches — ``surface`` does not re-hash the checkpoint —
        so without it a retrained run would leave the depth stage stale and the surface built
        from that depth looking fresh. The upstream fingerprint is recomputed from the world,
        never read back from the ledger, and memoised per call so a multi-gigabyte E57 is
        digested once however many stages hang off it.
        """
        cache = {} if cache is None else cache
        if stage in cache:
            return cache[stage]
        own = dict(specs[stage].inputs(self._context(stage)))
        idx = STAGE_ORDER.index(stage)
        if idx > 0 and STAGE_ORDER[idx - 1] in specs:
            prev = STAGE_ORDER[idx - 1]
            own["_upstream"] = fingerprint(self.resolve_inputs(prev, specs, cache))
        cache[stage] = own
        return own

    def execute(
        self,
        specs: dict[Stage, StageSpec],
        through: Stage = Stage.REPORT,
        rebuild_from: Stage | None = None,
        renderer: Any = None,
        trainer: Any = None,
        on_stage: Callable[[str, StageRecord], None] | None = None,
    ) -> WorkflowState:
        """Run the stages up to *through*, reusing what is still valid.

        *rebuild_from* is the explicit answer to a stale stage: it clears that stage and
        everything after it, so what gets rebuilt is visible in the ledger rather than implied.

        *on_stage* is called ``("start", rec)`` and ``("end", rec)`` around each stage. It exists
        so the CLI can say what is happening during a run measured in hours; it is told, never
        asked, and returning from it cannot change what the workflow does.
        """
        # The stages this execution was told to destroy and rebuild. A stage that publishes
        # into a fixed directory -- the staging tree, the dataset -- refuses to write over one
        # that is already there, which is correct on every run except the one that asked for
        # it. Without this, `--rebuild-from ingest` cleared the ledger and then failed on the
        # staging tree it had just declared void, so the remedy the workflow prints for a stale
        # E57 did not work.
        rebuilding: set[Stage] = set()
        if rebuild_from is not None:
            self._clear_from(rebuild_from)
            rebuilding = set(STAGE_ORDER[STAGE_ORDER.index(rebuild_from) :])
        wanted = [s for s in STAGE_ORDER if STAGE_ORDER.index(s) <= STAGE_ORDER.index(through)]
        cache: dict[Stage, dict[str, Any]] = {}
        for stage in wanted:
            if stage not in specs:
                raise ContractError(f"no handler registered for stage {stage.value}")
            if on_stage is not None:
                on_stage("start", self.state.stages[stage])
            # Dropped after each stage runs: the stage that just executed changed the world its
            # successors read, so a cached answer from before it would describe the old one.
            rec = self._one(
                specs,
                stage,
                cache,
                renderer=renderer,
                trainer=trainer,
                rebuilding=stage in rebuilding,
            )
            if on_stage is not None:
                on_stage("end", rec)
            cache.clear()
        return self.state

    def _one(
        self,
        specs: dict[Stage, StageSpec],
        stage: Stage,
        cache: dict[Stage, dict[str, Any]],
        renderer: Any,
        trainer: Any,
        rebuilding: bool = False,
    ) -> StageRecord:
        spec = specs[stage]
        rec = self.state.stages[stage]
        ctx = self._context(stage, renderer=renderer, trainer=trainer, rebuilding=rebuilding)
        inputs = self.resolve_inputs(stage, specs, cache)
        fp = fingerprint(inputs)

        if rec.usable:
            if rec.input_fingerprint == fp:
                if rec.status is StageStatus.SUCCEEDED:
                    rec.status = StageStatus.REUSED
                self.save()
                return rec
            raise ContractError(_stale(stage, rec, inputs))

        rec.status = StageStatus.RUNNING
        rec.started_at = now_iso()
        rec.input_fingerprint = fp
        rec.inputs = inputs
        rec.git_commit = git_commit()
        rec.minegs_version = minegs.__version__
        rec.tool_versions = tool_versions()
        rec.failure_reason = None
        self.save()

        t0 = time.monotonic()
        try:
            outcome = spec.run(ctx)
        except BaseException as e:
            rec.status = StageStatus.FAILED
            rec.completed_at = now_iso()
            rec.elapsed_seconds = round(time.monotonic() - t0, 3)
            rec.failure_reason = f"{type(e).__name__}: {e}"
            self.save()
            raise
        rec.status = StageStatus.SUCCEEDED
        rec.completed_at = now_iso()
        rec.elapsed_seconds = round(time.monotonic() - t0, 3)
        rec.outputs = dict(outcome.outputs)
        rec.command = dict(outcome.command)
        rec.runtime_env = dict(outcome.runtime_env)
        if stage is Stage.INGEST:
            # The survey's identity is the workflow's, not one stage's: `WorkflowState` declares
            # it (§P2 §3) and everything downstream is evidence about it. Lifted here rather
            # than left only in the stage's outputs, where the contract said it would not be.
            self.state.source_sha256 = outcome.outputs.get("source_sha256")
        self.save()
        if spec.finalise is not None:
            # After the record is final, so an artifact that describes this ledger describes
            # the finished one. A failure here is this stage's failure: a success nobody
            # completed is worse than a stage that says it stopped.
            try:
                spec.finalise(ctx, outcome)
            except BaseException as e:
                rec.status = StageStatus.FAILED
                rec.failure_reason = f"{type(e).__name__}: {e}"
                self.save()
                raise
        return rec

    def _clear_from(self, stage: Stage) -> None:
        start = STAGE_ORDER.index(stage)
        for s in STAGE_ORDER[start:]:
            self.state.stages[s] = StageRecord(stage=s)
        self.save()

    # ---------------------------------------------------------------- reading

    def stale_stages(self, specs: dict[Stage, StageSpec]) -> list[Stage]:
        """Which completed stages would be refused if the workflow ran now.

        Read-only: ``status`` asks the question without doing anything about it, which is what
        an operator wants before deciding what to rebuild.
        """
        out: list[Stage] = []
        cache: dict[Stage, dict[str, Any]] = {}
        for stage in STAGE_ORDER:
            rec = self.state.stages[stage]
            if not rec.usable or stage not in specs:
                continue
            try:
                fp = fingerprint(self.resolve_inputs(stage, specs, cache))
            except Exception:  # an input that cannot even be read is not a match
                out.append(stage)
                continue
            if fp != rec.input_fingerprint:
                out.append(stage)
        return out


def load_e2e_config(path: str | Path) -> E2EConfig:
    """Read a workflow config, with its relative paths taken against its own directory."""
    p = Path(path)
    if not p.is_file():
        raise ContractError(f"workflow config {p} not found")
    return E2EConfig.load(p).resolve_paths(p.parent)


def _stale(stage: Stage, rec: StageRecord, now: dict[str, Any]) -> str:
    changed = sorted(k for k in set(now) | set(rec.inputs) if now.get(k) != rec.inputs.get(k)) or [
        "(nothing named — the recorded inputs are from an older shape)"
    ]
    return (
        f"stage {stage.value} completed against different inputs: {changed} changed since it "
        f"ran. Its outputs describe a survey that is no longer on disk, so reusing them would "
        "carry stale evidence into everything downstream. Re-run this stage and the ones after "
        f"it explicitly (`--rebuild-from {stage.value}`), or point the workflow at the inputs "
        "it was built from."
    )


def _new_state(config: E2EConfig) -> WorkflowState:
    stamped = now_iso()
    return WorkflowState(
        workflow_id=make_id("e2e"),
        created_at=stamped,
        updated_at=stamped,
        config=config.model_dump(mode="json"),
        stages={s: StageRecord(stage=s) for s in STAGE_ORDER},
        provenance=stamp(config.model_dump(mode="json")),
    )


def stage_from_name(name: str) -> Stage:
    """Parse a stage name, listing the real ones rather than raising a bare ValueError."""
    try:
        return Stage(name)
    except ValueError:
        raise ContractError(
            f"{name!r} is not a workflow stage; the stages are {[s.value for s in STAGE_ORDER]}"
        ) from None


def stages_through(through: Stage) -> Iterable[Stage]:
    return (s for s in STAGE_ORDER if STAGE_ORDER.index(s) <= STAGE_ORDER.index(through))
