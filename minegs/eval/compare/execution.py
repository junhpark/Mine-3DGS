"""Local ↔ RunPod execution comparison (Phase 6 §11).

First the experiment fingerprint: the same dataset bytes, code, image digest, backend, profile,
trainer request, step budget, depth artifact and chunk. If any differs the two runs are not a
reproducibility pair, and nothing else is put side by side — numbers from two experiments do not
measure the provider. If they are a pair, only evidence both runs already recorded is laid next
to each other. No tolerance is applied (that is fixed in live G3 commissioning), bitwise identity
is not expected, and nothing here is a G3 verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from minegs.core.config import VersionedModel, canonical_json
from minegs.core.errors import ContractError
from minegs.core.provenance import ProvenanceRecord, stamp

EXECUTION_COMPARISON_FILE = "execution_comparison.json"
MATURITY = (
    "Configuration equivalence and recorded evidence of two executions. No tolerance is frozen, "
    "no reproducibility is claimed; Phase 6 G3: PENDING."
)


class ExecutionComparison(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    comparison_id: str
    reproducibility_pair: bool
    differences: list[str] = Field(default_factory=list)
    fingerprint: dict[str, Any] = Field(default_factory=dict)
    sides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    g3_status: str = "PENDING"
    maturity_statement: str = MATURITY
    notes: list[str] = Field(default_factory=list)
    provenance: ProvenanceRecord


def fingerprint(rec) -> dict[str, Any]:
    """What makes two runs the same experiment, from the run's own record."""
    chunk = rec.chunk or {}
    return {
        "dataset_hash": rec.dataset_hash,
        "code": rec.provenance.git_commit,
        "docker_digest": rec.docker_digest,
        "backend": canonical_json(rec.backend),
        "profile": canonical_json(rec.profile),
        "expected_trainer_config": canonical_json(rec.expected_trainer_config),
        "max_steps": rec.max_steps,
        "depth_supervision": (rec.depth_supervision or {}).get("artifact_sha256"),
        "chunk": None if not chunk else [chunk.get("plan_digest"), chunk.get("chunk_id")],
    }


def _side(rec, run_dir: Path) -> dict[str, Any]:
    from minegs.train.runner.base import real_gpu_evidence

    remote = run_dir / "remote.json"
    verified = False
    if remote.is_file():
        verified = bool(json.loads(remote.read_text()).get("artifact_sync_verified"))
    rt = dict(rec.runtime or {})
    return {
        "run_id": rec.run_id,
        "runner": rec.runner,
        "status": rec.status.value,
        "real_gpu_training": real_gpu_evidence(rec),
        "remote_provider_execution": rec.runner == "runpod" and rec.remote_execution is not None,
        "artifact_sync_verified": verified,
        "gpu_model": rt.get("gpu_model"),
        "gaussian_count": rec.gaussian_count,
        "observed_final_step": rec.observed_final_step,
        "scene_scale": (rec.chunk or {}).get("scene_scale"),
        "train_seconds": rec.train_seconds,
        "duration_s": rec.duration_s,
        "peak_gpu_memory_gb": rec.peak_gpu_memory_gb,
    }


def compare_execution(
    run_a: str | Path, run_b: str | Path, *, comparison_id: str = "execution-comparison"
) -> ExecutionComparison:
    from minegs.train.runner.base import RunStatus, load_record

    a_dir, b_dir = Path(run_a), Path(run_b)
    a, b = load_record(a_dir), load_record(b_dir)
    if a.run_id == b.run_id:
        raise ContractError(
            f"both inputs are run {a.run_id}; an execution is not compared to itself"
        )
    fa, fb = fingerprint(a), fingerprint(b)
    diffs = [k for k in fa if fa[k] != fb[k]]
    notes = []
    for rec in (a, b):
        code = rec.provenance.git_commit
        if code == "unknown" or code.endswith("-dirty"):
            diffs.append(f"code of {rec.run_id} is {code!r}: which code ran cannot be shown")
        if rec.docker_digest is None:
            diffs.append(f"{rec.run_id} records no image digest (a --native run)")
        if rec.status is not RunStatus.SUCCEEDED:
            diffs.append(f"{rec.run_id} is {rec.status.value}, not succeeded")
    pair = not diffs
    sides = {}
    if pair:
        sides = {"a": _side(a, a_dir), "b": _side(b, b_dir)}
    else:
        notes.append(
            "not a reproducibility pair: the runs differ in what they computed, so their "
            "numbers are not laid side by side"
        )
    notes.append("no tolerance is applied; bitwise identity across GPUs is not expected")
    return ExecutionComparison(
        comparison_id=comparison_id,
        reproducibility_pair=pair,
        differences=diffs,
        fingerprint={"a": fa, "b": fb},
        sides=sides,
        notes=notes,
        provenance=stamp({"comparison_id": comparison_id}, parents=[a.run_id, b.run_id]),
    )


__all__ = ["EXECUTION_COMPARISON_FILE", "MATURITY", "ExecutionComparison", "compare_execution"]
