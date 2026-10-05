"""Phase 6 — what reaches the pod is the dataset being claimed, and nothing else.

The volume is a temp directory; tampering with it stands in for anything that could change the
bytes between this machine and the trainer. The pod re-derives every identity before training.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from minegs.core.errors import ContractError
from minegs.core.provenance import sha256_tree
from minegs.train.remote.bundle import dataset_file_table, dataset_hash_of
from minegs.train.remote.layout import RemoteLayout
from minegs.train.remote.status import JobStatus
from minegs.train.remote.worker import verify_pod_inputs
from minegs.train.runner import sync as rsync
from minegs.train.runner.base import DATASET_HASH_PATTERNS, RunStatus, load_record
from minegs.train.runner.runpod import RemoteExecutionRecord

from phase6_fakes import layout_of, runpod_config, submit, volume_files


@pytest.fixture
def dataset_copy(synthetic, tmp_path):
    """A private copy of the synthetic dataset with Phase 3-style evidence under provenance/."""
    ds = tmp_path / "proj" / "dataset"
    shutil.copytree(synthetic.dataset_dir, ds)
    (ds / "provenance" / "sfm").mkdir(parents=True)
    (ds / "provenance" / "sfm" / "cameras.txt").write_text("# evidence\n")
    (ds / "provenance" / "registration.json").write_text('{"scale": 1.0}')
    (ds / "chunks").mkdir(exist_ok=True)
    (ds / "stray_notes.txt").write_text("not part of the dataset")
    return ds


def test_the_upload_is_exactly_the_bytes_the_dataset_hash_covers(dataset_copy):
    table = dataset_file_table(dataset_copy)
    assert "provenance/sfm/cameras.txt" in table and "provenance/registration.json" in table
    assert "stray_notes.txt" not in table and not any(f.startswith("chunks/") for f in table)
    assert dataset_hash_of(table) == sha256_tree(dataset_copy, DATASET_HASH_PATTERNS)
    assert rsync.dataset_files(dataset_copy) == sorted(table)
    argv = rsync.push(dataset_copy, "remote:x", dry_run=True)  # dry run: no rclone, no mutation
    assert argv[:2] == ["rclone", "copy"] and "--files-from" in argv


def test_survey_sources_never_go_up(synthetic, dataset_copy, pod_env):
    (dataset_copy / "images" / "scan.e57").write_bytes(b"E57 raw")
    for target, match in (
        (dataset_copy, "survey source files"),
        (synthetic.root / "raw", "raw"),
        (synthetic.root, "not a dataset"),
    ):
        with pytest.raises(ContractError, match=match):
            rsync.push_command(target, "remote:x")
    with pytest.raises(ContractError, match="survey source files"):
        submit(synthetic, pod_env, dataset_dir=dataset_copy)
    assert pod_env.client.created == [] and volume_files(pod_env) == []


@pytest.mark.parametrize("link", ["provenance_dir", "image_outside", "raw_behind_png"])
def test_nothing_goes_up_through_a_symlink(synthetic, dataset_copy, pod_env, tmp_path, link):
    """A link can point anywhere on this machine; its target never reaches the volume."""
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "secret_notes.txt").write_text("not a dataset file")
    (outside / "scan.e57").write_bytes(b"E57 raw")
    if link == "provenance_dir":
        shutil.rmtree(dataset_copy / "provenance")
        (dataset_copy / "provenance").symlink_to(outside, target_is_directory=True)
    else:
        img = sorted((dataset_copy / "images").rglob("*.*"))[0]
        img.unlink()
        target = outside / ("secret_notes.txt" if link == "image_outside" else "scan.e57")
        img.symlink_to(target)
    with pytest.raises(ContractError, match="symlink"):
        dataset_file_table(dataset_copy)
    with pytest.raises(ContractError, match="symlink"):
        submit(synthetic, pod_env, dataset_dir=dataset_copy)
    assert pod_env.client.created == [] and volume_files(pod_env) == []


def test_a_claimed_dataset_path_is_never_written_again(synthetic, dataset_copy, pod_env):
    """Content-addressed and write-once: a second submission of the same bytes uploads nothing,
    and a volume copy that went bad is refused by the pod, not silently repaired or reused."""
    h1 = submit(synthetic, pod_env, name="v1", dataset_dir=dataset_copy)
    assert h1.wait(poll_s=0.01) is RunStatus.SUCCEEDED
    rec1 = load_record(h1.run_dir)
    remote_ds = pod_env.volume / layout_of(pod_env, rec1).dataset(rec1.dataset_hash)
    before = {p: p.stat().st_mtime_ns for p in remote_ds.rglob("*") if p.is_file()}
    img = sorted((remote_ds / "images").rglob("*.*"))[0]
    img.write_bytes(img.read_bytes() + b"\0")
    bad = img.read_bytes()
    h2 = submit(synthetic, pod_env, name="v2", dataset_dir=dataset_copy)
    assert h2.wait(poll_s=0.01) is RunStatus.FAILED
    assert RemoteExecutionRecord.load(h2.run_dir / "remote.json").failure_stage == (
        "input_verification"
    )
    assert img.read_bytes() == bad  # not overwritten
    after = {p: p.stat().st_mtime_ns for p in remote_ds.rglob("*") if p.is_file()}
    assert {p: t for p, t in after.items() if p != img} == {
        p: t for p, t in before.items() if p != img
    }


def test_the_trainer_reads_only_verified_bytes(synthetic, dataset_copy, pod_env, monkeypatch):
    """A staged image that is not the verified dataset's file fails the job after the run."""
    from minegs.train.runner import local as runner_local

    real = runner_local.stage_dataset

    def swapping(dataset_dir, staged_dir, *a, **kw):
        st = real(dataset_dir, staged_dir, *a, **kw)
        f = sorted((Path(staged_dir) / "images").rglob("*.*"))[0]
        data = f.read_bytes()
        f.unlink()  # staging hard-links: replace the link, never write through it
        f.write_bytes(data + b"\0")
        return st

    monkeypatch.setattr(runner_local, "stage_dataset", swapping)
    h = submit(synthetic, pod_env, dataset_dir=dataset_copy)
    assert h.wait(poll_s=0.01) is RunStatus.FAILED
    rec = load_record(h.run_dir)
    js = JobStatus.model_validate(RemoteExecutionRecord.load(h.run_dir / "remote.json").job_status)
    assert js.failure_stage == "input_verification" and js.exit_code == 1
    assert rec.status is RunStatus.FAILED and "not the verified dataset" in rec.failure_reason


@pytest.mark.parametrize("swap", ["sparse_cameras", "init_points"])
def test_a_dataset_changed_during_staging_never_reaches_the_trainer(
    synthetic, dataset_copy, pod_env, monkeypatch, swap
):
    """Staging reads sparse/0 and init_points.ply after prepare hashed the volume. A change in that
    window is caught by the re-hash after staging, before any trainer process exists."""
    import subprocess

    from minegs.core.pointcloud import read_ply, write_ply
    from minegs.train.runner import local as runner_local

    real = runner_local.stage_dataset
    started = []

    def changing(dataset_dir, staged_dir, *a, **kw):
        ds = Path(dataset_dir)
        if swap == "sparse_cameras":
            f = ds / "sparse" / "0" / "cameras.txt"
            f.write_text(f.read_text() + "# edited on the volume\n")
        else:
            pc = read_ply(ds / "init_points.ply")
            write_ply(pc.select(np.arange(len(pc) - 1)), ds / "init_points.ply")
        return real(dataset_dir, staged_dir, *a, **kw)  # reads the changed bytes

    real_popen = subprocess.Popen

    def popen(*a, **kw):
        if "simple_trainer" in " ".join(map(str, a[0] if a else kw.get("args", []))):
            started.append(a)
        return real_popen(*a, **kw)

    monkeypatch.setattr(runner_local, "stage_dataset", changing)
    monkeypatch.setattr(runner_local.subprocess, "Popen", popen)
    h = submit(synthetic, pod_env, dataset_dir=dataset_copy)
    assert h.wait(poll_s=0.01) is RunStatus.FAILED
    assert started == []  # no trainer process was ever launched
    rem = RemoteExecutionRecord.load(h.run_dir / "remote.json")
    js = JobStatus.model_validate(rem.job_status)
    assert js.failure_stage == "input_verification" and js.exit_code == 2
    assert js.trainer_exit_code is None and "changed while it was being staged" in js.message
    assert rem.failure_stage == "input_verification" and not rem.artifact_sync_verified
    rec = load_record(h.run_dir)
    assert rec.status is RunStatus.FAILED and "changed while it was being staged" in (
        rec.failure_reason
    )
    pod_run = pod_env.volume / layout_of(pod_env, rec).run(rec.run_id)
    assert not (pod_run / "run.json").exists() and not (pod_run / "point_cloud").exists()


def test_a_symlinked_dataset_root_is_refused(synthetic, dataset_copy, pod_env, tmp_path):
    link = tmp_path / "dataset_link"
    link.symlink_to(dataset_copy, target_is_directory=True)
    with pytest.raises(ContractError, match="symlink"):
        submit(synthetic, pod_env, dataset_dir=link)
    assert pod_env.client.created == [] and volume_files(pod_env) == []


def _staged_job(synthetic, env, ds, name="r1", **over):
    """Submit with the container not started yet; return the handle and the pod-side paths."""
    env.mode("running")
    h = submit(synthetic, env, name=name, dataset_dir=ds, **over)
    rec = load_record(h.run_dir)
    layout = layout_of(env, rec)
    return (
        h,
        rec,
        env.volume / layout.dataset(rec.dataset_hash),
        env.volume / layout.job(rec.run_id),
    )


def _run_container(env) -> int:
    return env.client.start_container(env.client.created[-1])


@pytest.mark.parametrize(
    "tamper",
    ["image_bytes", "extra_image", "missing_provenance", "changed_centerline"],
)
def test_the_pod_refuses_a_dataset_that_is_not_the_one_submitted(
    synthetic, dataset_copy, pod_env, tamper
):
    h, rec, remote_ds, job = _staged_job(synthetic, pod_env, dataset_copy)
    if tamper == "image_bytes":
        img = sorted((remote_ds / "images").rglob("*.*"))[0]
        img.write_bytes(img.read_bytes() + b"\0")
    elif tamper == "extra_image":
        (remote_ds / "images" / "extra.png").write_bytes(b"not in the dataset")
    elif tamper == "missing_provenance":
        (remote_ds / "provenance" / "registration.json").unlink()
    else:
        cl = remote_ds / "centerline.csv"
        cl.write_text(cl.read_text() + "\n")
    assert _run_container(pod_env) == 2
    js = JobStatus.load(job / "status.json")
    assert (
        js.failure_stage == "input_verification"
        and "not the dataset that was submitted" in js.message
    )
    assert h.status() is RunStatus.FAILED
    assert not (pod_env.volume / layout_of(pod_env, rec).run(rec.run_id)).exists()


def test_an_edited_or_unpublished_input_bundle_is_refused(synthetic, dataset_copy, pod_env):
    h, _rec, _ds, job = _staged_job(synthetic, pod_env, dataset_copy)
    inputs = json.loads((job / "inputs.json").read_text())
    inputs["overrides"]["max_steps"] = 99999
    (job / "inputs.json").write_text(json.dumps(inputs))
    assert _run_container(pod_env) == 2
    assert JobStatus.load(job / "status.json").failure_stage == "input_bundle"
    assert h.status() is RunStatus.FAILED
    # only a partial bundle on the volume: the worker does not start from it
    _h2, _rec2, _ds, job2 = _staged_job(synthetic, pod_env, dataset_copy, name="r2")
    (job2 / "inputs.json").rename(job2 / "inputs.json.partial")
    assert _run_container(pod_env) == 2
    assert "partial input set" in JobStatus.load(job2 / "status.json").message


def test_a_dataset_path_that_claims_another_dataset_is_never_overwritten(
    synthetic, dataset_copy, pod_env
):
    table = dataset_file_table(dataset_copy)
    from minegs.core.manifest import Manifest

    ds_id = Manifest.load_dataset(dataset_copy).dataset_id
    path = pod_env.volume / RemoteLayout(str(pod_env.volume), ds_id).dataset(dataset_hash_of(table))
    path.mkdir(parents=True)
    (path / "dataset.json").write_text(json.dumps({"dataset_id": ds_id, "dataset_hash": "0" * 64}))
    with pytest.raises(ContractError, match="holds something else"):
        submit(synthetic, pod_env, dataset_dir=dataset_copy)
    assert sorted(p.name for p in path.iterdir()) == ["dataset.json"]  # nothing written over it
    assert pod_env.client.created == []
    rem = RemoteExecutionRecord.load(pod_env.tmp / "runs" / "r1" / "remote.json")
    assert rem.failure_stage == "input_sync"


def test_two_versions_of_one_dataset_id_never_share_bytes(synthetic, dataset_copy, pod_env):
    h1 = submit(synthetic, pod_env, name="v1", dataset_dir=dataset_copy)
    assert h1.wait(poll_s=0.01) is RunStatus.SUCCEEDED
    rec1 = load_record(h1.run_dir)
    first = pod_env.volume / layout_of(pod_env, rec1).dataset(rec1.dataset_hash)
    (dataset_copy / "provenance" / "registration.json").write_text('{"scale": 2.0}')
    h2 = submit(synthetic, pod_env, name="v2", dataset_dir=dataset_copy)
    assert h2.wait(poll_s=0.01) is RunStatus.SUCCEEDED
    rec2 = load_record(h2.run_dir)
    assert rec2.dataset_id == rec1.dataset_id and rec2.dataset_hash != rec1.dataset_hash
    assert sha256_tree(first, DATASET_HASH_PATTERNS) == rec1.dataset_hash  # untouched
    # the same bytes again: same path, nothing to change, still verified on the pod
    h3 = submit(synthetic, pod_env, name="v3", dataset_dir=dataset_copy)
    assert h3.wait(poll_s=0.01) is RunStatus.SUCCEEDED
    datasets = pod_env.volume / layout_of(pod_env, rec1).base / "datasets"
    assert len(list(datasets.iterdir())) == 2


def test_a_remote_run_id_is_written_once(synthetic, pod_env):
    h = submit(synthetic, pod_env, name="a", run_id="fixed_run_001")
    assert h.wait(poll_s=0.01) is RunStatus.SUCCEEDED
    with pytest.raises(ContractError, match="written once"):
        submit(synthetic, pod_env, name="b", run_id="fixed_run_001")
    assert len(pod_env.client.created) == 1
    with pytest.raises(ContractError, match="cannot name a remote path"):
        submit(synthetic, pod_env, name="c", run_id="bad id; rm -rf")


@pytest.fixture
def depth_artifact(synthetic, dataset_copy, tmp_path):
    from minegs.train.supervision.build import build_tls_projection

    return build_tls_projection(
        dataset_copy, synthetic.root / "raw" / "tls_full.ply", tmp_path / "dsup"
    )


def test_the_depth_artifact_is_reverified_on_the_pod(
    synthetic, dataset_copy, depth_artifact, pod_env
):
    h, rec, _remote_ds, job = _staged_job(
        synthetic,
        pod_env,
        dataset_copy,
        profile="heavy-depth",
        depth_supervision=str(depth_artifact.path),
    )
    inputs = json.loads((job / "inputs.json").read_text())
    from minegs.train.remote.bundle import RunInputBundle

    bundle = RunInputBundle.model_validate(inputs)
    assert bundle.depth_supervision.artifact_sha256 == depth_artifact.artifact_sha256
    pod_hash, _gpu, _table = verify_pod_inputs(bundle)  # the uploaded artifact verifies as it is
    assert pod_hash == rec.dataset_hash
    samples = Path(bundle.depth_supervision.path) / "samples.npy"
    samples.write_bytes(samples.read_bytes()[:-8] + b"\x00" * 8)
    assert _run_container(pod_env) == 2
    js = JobStatus.load(job / "status.json")
    assert js.failure_stage == "input_verification"
    assert h.status() is RunStatus.FAILED


def test_the_round_trip_through_an_rclone_remote(synthetic, pod_env, monkeypatch):
    """The same run with the volume reached through rclone (a stand-in binary over the same
    directory): transport changes, every integrity check is the same code."""
    import os

    monkeypatch.setenv("FAKE_RCLONE_ROOT", str(pod_env.volume.parent))
    shim = Path(__file__).parent / "fake_rclone"
    monkeypatch.setenv("PATH", f"{shim}:{os.environ['PATH']}")
    cfg = runpod_config(pod_env.volume, sync={"remote": f"rs3:{pod_env.volume.name}"})
    h = submit(synthetic, pod_env, cfg=cfg)
    assert h.wait(poll_s=0.01) is RunStatus.SUCCEEDED
    rem = RemoteExecutionRecord.load(h.run_dir / "remote.json")
    assert rem.storage == f"rs3:{pod_env.volume.name}" and rem.artifact_sync_verified
    assert h.fetch_artifacts()
