"""Phase 5 C2 — a depth-supervised heavy chunk, end to end through the Phase 4 adapter.

Runs the stand-in upstream (``tests/fake_upstream``) with CPU torch, as the Phase 4 torch tests
do. What is shown is the wiring: the global depth artifact reused byte for byte, the adapter
reading only the chunk's samples, and the chunk binding in the evidence. Real GPU: NOT PERFORMED.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the adapter executes torch code")

from minegs.chunks.plan import build_chunk_plan  # noqa: E402
from minegs.train.runner import RunConfig, get_runner  # noqa: E402
from minegs.train.runner import local as runner_local  # noqa: E402
from minegs.train.runner.base import RunnerConfig, RunStatus, load_record  # noqa: E402
from minegs.train.supervision.build import build_tls_projection  # noqa: E402

from phase5_scene import CORE_M, OVERLAP_M, long_tunnel  # noqa: E402

FAKE = Path(__file__).parent / "fake_upstream" / "simple_trainer.py"


@pytest.fixture
def upstream(monkeypatch):
    monkeypatch.setenv("MINEGS_GSPLAT_TRAINER", str(FAKE))
    monkeypatch.setattr(runner_local, "cuda_available", lambda: True)
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}:{os.environ['PATH']}")


def test_a_heavy_chunk_trains_with_the_global_depth_artifact(tmp_path, upstream):
    sc = long_tunnel(tmp_path / "t")
    plan, plan_path = build_chunk_plan(sc.dataset_dir, CORE_M, OVERLAP_M)
    sup = build_tls_projection(sc.dataset_dir, sc.cloud, tmp_path / "dsup")
    h = get_runner("local", RunnerConfig(runner="local", native=True)).submit(
        RunConfig(
            dataset_dir=str(sc.dataset_dir),
            profile="heavy",
            run_dir=str(tmp_path / "run"),
            chunk_id="K001",
            chunk_plan=str(plan_path),
            depth_supervision=str(sup.path),
            overrides={"max_steps": 6},
        )
    )
    assert h.wait(poll_s=0.05) is RunStatus.SUCCEEDED, load_record(h.run_dir).failure_reason
    rec = load_record(h.run_dir)
    chunk = plan.chunk("K001")
    assert rec.chunk["chunk_id"] == "K001" and rec.chunk["plan_digest"] == plan.plan_digest
    # the artifact is the global one, unchanged; the trainer used the chunk's images only
    assert rec.depth_supervision["artifact_sha256"] == sup.artifact_sha256
    trainer = rec.depth_supervision["trainer"]
    assert trainer["steps_with_depth_term"] > 0
    assert 0 < trainer["images_with_samples"] <= len(chunk.images)
    assert rec.chunk["depth_images_with_samples"] == len(sup.images_with_samples(chunk.images))
    assert rec.chunk["depth_samples_for_images"] < sup.record.n_samples_in_loss
    # scene_scale is the chunk's own, recorded and not compensated
    assert rec.chunk["scene_scale"] and rec.chunk["scene_scale"] > 0
