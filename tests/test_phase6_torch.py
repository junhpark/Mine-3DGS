"""Phase 6 — a depth-supervised heavy chunk, end to end through RunPod (fake provider, CPU torch).

The pod side runs the Phase 4 adapter around the stand-in upstream (``tests/fake_upstream``) with
the global depth artifact and the Phase 5 plan it received by identity. Shown: the wiring and the
evidence. Real GPU, real RunPod: NOT PERFORMED.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the adapter executes torch code")

from minegs.chunks.plan import build_chunk_plan  # noqa: E402
from minegs.train.runner.base import RunStatus, load_record  # noqa: E402
from minegs.train.supervision.build import build_tls_projection  # noqa: E402

from phase5_scene import CORE_M, OVERLAP_M, long_tunnel  # noqa: E402
from phase6_fakes import submit  # noqa: E402

FAKE = Path(__file__).parent / "fake_upstream" / "simple_trainer.py"


def test_a_heavy_depth_chunk_runs_through_runpod(pod_env, monkeypatch):
    monkeypatch.setenv("MINEGS_GSPLAT_TRAINER", str(FAKE))
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}:{os.environ['PATH']}")
    sc = long_tunnel(pod_env.tmp / "t")
    plan, plan_path = build_chunk_plan(sc.dataset_dir, CORE_M, OVERLAP_M)
    sup = build_tls_projection(sc.dataset_dir, sc.cloud, pod_env.tmp / "dsup")
    h = submit(
        None,
        pod_env,
        dataset_dir=sc.dataset_dir,
        profile="heavy",
        overrides={"max_steps": 6},
        chunk_id="K001",
        chunk_plan=str(plan_path),
        depth_supervision=str(sup.path),
    )
    assert h.wait(poll_s=0.05) is RunStatus.SUCCEEDED, load_record(h.run_dir).failure_reason
    rec = load_record(h.run_dir)
    assert rec.runner == "runpod" and rec.chunk["plan_digest"] == plan.plan_digest
    assert rec.depth_supervision["artifact_sha256"] == sup.artifact_sha256
    assert rec.remote_sync["depth_artifact_sha256"] == sup.artifact_sha256
    assert rec.depth_supervision["trainer"]["steps_with_depth_term"] > 0
    assert rec.chunk["staged_images"] == len(plan.chunk("K001").images)
    assert rec.chunk["scene_scale_source"] == "adapter" and rec.chunk["scene_scale"] > 0
