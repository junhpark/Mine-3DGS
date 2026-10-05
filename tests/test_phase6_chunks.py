"""Phase 6 — Phase 5 chunk runs and Phase 4 depth runs, executed through RunPod.

``train chunks`` keeps its sequential loop and first-failure stop; each chunk is one RunPod
submission (one pod, one run). The pod receives the plan and the depth artifact by identity and
verifies both again. Fake provider, temp volume; real GPU: NOT PERFORMED.
"""

from __future__ import annotations

import shutil

import pytest
from minegs.chunks.plan import build_chunk_plan, load_chunk_plan
from minegs.chunks.run import train_chunks
from minegs.core.manifest import Manifest
from minegs.train.remote.bundle import RunInputBundle
from minegs.train.remote.worker import verify_pod_inputs
from minegs.train.runner.base import RunStatus, load_record
from minegs.train.runner.runpod import RunPodRunner
from minegs.train.staging import stage_dataset

from phase5_scene import CORE_M, OVERLAP_M, long_tunnel
from phase6_fakes import layout_of, runpod_config, submit


@pytest.fixture
def tunnel(tmp_path):
    sc = long_tunnel(tmp_path / "t")
    sc.plan, sc.plan_path = build_chunk_plan(sc.dataset_dir, CORE_M, OVERLAP_M)
    return sc


def test_train_chunks_runs_each_chunk_on_its_own_pod(tunnel, pod_env, tmp_path):
    pod_env.trainer(span=50.0)
    done = train_chunks(
        tunnel.dataset_dir,
        tunnel.plan_path,
        "light",
        RunPodRunner(runpod_config(pod_env.volume)),
        runs_dir=tmp_path / "runs",
        overrides={"max_steps": 10},
        poll_s=0.01,
    )
    assert [d["chunk_id"] for d in done] == ["K000", "K001", "K002"]
    assert all(d["status"] == "succeeded" for d in done)
    assert len(pod_env.client.created) == 3 and pod_env.client.terminated == [
        "pod001",
        "pod002",
        "pod003",
    ]
    for d in done:
        rec = load_record(d["run_dir"])
        c = tunnel.plan.chunk(d["chunk_id"])
        assert rec.runner == "runpod" and rec.chunk["plan_digest"] == tunnel.plan.plan_digest
        assert rec.chunk["staged_images"] == rec.chunk["planned_images"] == len(c.images)
        assert rec.remote_sync["chunk_plan_digest"] == tunnel.plan.plan_digest
        assert tuple(rec.chunk["core_range_m"]) == c.core_range_m
    # one plan upload, at its digest; the pod recorded where it read it, not an identity
    lay = layout_of(pod_env, Manifest.load_dataset(tunnel.dataset_dir).dataset_id)
    plan_dir = pod_env.volume / lay.chunk_plan(tunnel.plan.plan_digest)
    assert load_record(done[1]["run_dir"]).chunk["plan_path"] == str(plan_dir / "chunk_plan.json")

    # a pulled chunk run is rendered from the plan found by identity once the pod path is gone
    from minegs.eval.surface.depth import run_views
    from minegs.ingest.common.colmap_io import read_model

    shutil.rmtree(plan_dir)
    rec = load_record(done[1]["run_dir"])
    model = read_model(tunnel.dataset_dir / "sparse" / "0")
    views = run_views(rec, tunnel.dataset_dir, model.images)
    assert {im.name for im in views.values()} == set(tunnel.plan.chunk("K001").views)


def test_a_failed_remote_chunk_stops_the_set(tunnel, pod_env, tmp_path):
    pod_env.trainer(span=50.0, exit_code=3)
    done = train_chunks(
        tunnel.dataset_dir,
        tunnel.plan_path,
        "light",
        RunPodRunner(runpod_config(pod_env.volume)),
        runs_dir=tmp_path / "runs",
        overrides={"max_steps": 10},
        poll_s=0.01,
    )
    assert [(d["chunk_id"], d["status"]) for d in done] == [("K000", "failed")]
    assert len(pod_env.client.created) == 1


def test_a_depth_supervised_chunk_reaches_the_pod_whole(tunnel, pod_env):
    """RunPod + chunk + heavy-depth + DepthSupervisionRecord: same dataset, plan, artifact and
    profile on the pod, and the pod stages every planned image of the chunk."""
    from minegs.train.supervision.build import build_tls_projection

    sup = build_tls_projection(tunnel.dataset_dir, tunnel.cloud, pod_env.tmp / "dsup")
    pod_env.mode("running")
    h = submit(
        None,
        pod_env,
        dataset_dir=tunnel.dataset_dir,
        profile="heavy-depth",
        overrides={"max_steps": 10},
        chunk_id="K001",
        chunk_plan=str(tunnel.plan_path),
        depth_supervision=str(sup.path),
    )
    rec = load_record(h.run_dir)
    job = pod_env.volume / layout_of(pod_env, rec).job(rec.run_id)
    bundle = RunInputBundle.load(job / "inputs.json")
    bundle.check_digest()
    assert bundle.profile == "heavy-depth" and bundle.dataset_hash == rec.dataset_hash
    assert bundle.chunk.plan_digest == tunnel.plan.plan_digest and bundle.chunk.chunk_id == "K001"
    assert bundle.depth_supervision.artifact_sha256 == sup.artifact_sha256
    pod_hash, _gpu = verify_pod_inputs(bundle)
    assert pod_hash == rec.dataset_hash
    remote_ds = bundle.paths["dataset"]
    plan = load_chunk_plan(bundle.chunk.path)
    st = stage_dataset(
        remote_ds,
        pod_env.tmp / "pod_stage",
        Manifest.load_dataset(remote_ds),
        max_images=None,
        chunk=plan.chunk("K001"),
    )
    assert sorted(st.images) == tunnel.plan.chunk("K001").images
    assert h.status() is RunStatus.PENDING
