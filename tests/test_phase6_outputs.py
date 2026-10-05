"""Phase 6 — outputs are published only when every byte matches what the pod recorded.

Each test lets the pod finish, then changes what comes back: the manifest, one file, the bytes in
transit, or a run.json a capable forger made consistent with its manifest. Nothing partial is
ever published as a run.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from minegs.core.provenance import sha256_file
from minegs.train.remote import store as remote_store
from minegs.train.remote.layout import OUTPUT_MANIFEST_FILE
from minegs.train.remote.status import (
    JobStatus,
    RemoteOutputRecord,
    output_entries,
    tree_digest,
)
from minegs.train.runner.base import RunStatus, load_record
from minegs.train.runner.runpod import RemoteExecutionRecord

from phase6_fakes import layout_of, submit


def _finished(synthetic, env, name="r1"):
    """A job the pod has finished, not yet looked at from this side."""
    env.mode("running")
    h = submit(synthetic, env, name=name)
    assert env.client.start_container(env.client.created[-1]) == 0
    rec = load_record(h.run_dir)
    lay = layout_of(env, rec)
    return h, env.volume / lay.run(rec.run_id), env.volume / lay.job(rec.run_id)


def _assert_not_published(h, stage="pull_verification"):
    assert h.status() is RunStatus.FAILED
    rem = RemoteExecutionRecord.load(h.run_dir / "remote.json")
    assert rem.failure_stage == stage and not rem.artifact_sync_verified
    assert sorted(p.name for p in h.run_dir.iterdir()) == ["remote.json", "run.json"]
    assert load_record(h.run_dir).status is RunStatus.FAILED
    assert not list(h.run_dir.parent.glob(f".{h.run_dir.name}.pull"))
    return rem


def test_a_tampered_output_manifest_is_refused(synthetic, pod_env):
    h, run, _job = _finished(synthetic, pod_env)
    m = json.loads((run / OUTPUT_MANIFEST_FILE).read_text())
    m["entries"][0]["size"] += 1
    (run / OUTPUT_MANIFEST_FILE).write_text(json.dumps(m))
    rem = _assert_not_published(h)
    assert "not the one the job status recorded" in rem.failure_message


def test_a_file_lost_or_changed_in_transit_is_refused(synthetic, pod_env, monkeypatch):
    real = remote_store.LocalDirStore.download_tree

    def lossy(self, src_rel, dest, exclude=()):
        real(self, src_rel, dest, exclude)
        next(p for p in sorted(dest.rglob("*.ply"))).unlink()

    h, _run, _job = _finished(synthetic, pod_env)
    monkeypatch.setattr(remote_store.LocalDirStore, "download_tree", lossy)
    assert "missing" in _assert_not_published(h).failure_message

    def corrupting(self, src_rel, dest, exclude=()):
        real(self, src_rel, dest, exclude)
        f = next(p for p in sorted(dest.rglob("*.ply")))
        data = bytearray(f.read_bytes())
        data[-1] ^= 0xFF  # a real change, whatever the byte was
        f.write_bytes(bytes(data))

    h, _run, _job = _finished(synthetic, pod_env, name="r2")
    monkeypatch.setattr(remote_store.LocalDirStore, "download_tree", corrupting)
    assert "differ from the manifest" in _assert_not_published(h).failure_message


def _forge(run, job, edit) -> None:
    """Change run.json and re-seal manifest and status consistently, as a capable forger would."""
    rj = run / "run.json"
    d = json.loads(rj.read_text())
    edit(d)
    rj.write_text(json.dumps(d))
    m = RemoteOutputRecord.load(run / OUTPUT_MANIFEST_FILE)
    entries = output_entries(run, OUTPUT_MANIFEST_FILE)
    m = m.model_copy(
        update={
            "entries": entries,
            "output_tree_digest": tree_digest(entries),
            "run_json_sha256": sha256_file(rj),
        }
    )
    (run / OUTPUT_MANIFEST_FILE).write_text(m.model_dump_json(indent=2))
    js = JobStatus.load(job / "status.json")
    js = js.model_copy(
        update={
            "output_manifest_sha256": hashlib.sha256(
                (run / OUTPUT_MANIFEST_FILE).read_bytes()
            ).hexdigest()
        }
    )
    js.save(job / "status.json")


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda d: d.update(run_id="someone_elses_run"), "run_id someone_elses_run"),
        (lambda d: d.update(dataset_hash="0" * 64), "dataset_hash 000000000000"),
        (
            lambda d: d.update(
                image="ghcr.io/x/y@sha256:" + "cd" * 32, docker_digest="sha256:" + "cd" * 32
            ),
            "image ghcr.io/x/y",
        ),
        (lambda d: d.update(runner="local"), "runner local"),
        (lambda d: d["remote_sync"].update(input_bundle_digest="f" * 64), "input bundle"),
    ],
)
def test_a_run_record_that_is_not_the_submitted_run_is_refused(synthetic, pod_env, edit, match):
    h, run, job = _finished(synthetic, pod_env)
    _forge(run, job, edit)
    rem = _assert_not_published(h)
    assert match in rem.failure_message


def test_outputs_never_land_on_foreign_files(synthetic, pod_env):
    h, _run, _job = _finished(synthetic, pod_env)
    (h.run_dir / "notes.txt").write_text("someone else's file")
    assert h.status() is RunStatus.FAILED
    rem = RemoteExecutionRecord.load(h.run_dir / "remote.json")
    assert "this submission did not write" in rem.failure_message
    assert (h.run_dir / "notes.txt").read_text() == "someone else's file"
    assert not (h.run_dir / "point_cloud").exists()


def test_a_verified_pull_matches_the_pod_byte_for_byte(synthetic, pod_env):
    h, run, _job = _finished(synthetic, pod_env)
    assert h.status() is RunStatus.SUCCEEDED
    m = RemoteOutputRecord.load(run / OUTPUT_MANIFEST_FILE)
    for e in m.entries:
        assert sha256_file(h.run_dir / e.path) == e.sha256
    assert not (h.run_dir / "backend_out").exists()  # scratch stays on the volume
    rem = RemoteExecutionRecord.load(h.run_dir / "remote.json")
    assert rem.pulled_output_sha256 == m.output_tree_digest


# ================================================================ execution comparison


def test_compare_execution_pairs_only_the_same_experiment(
    synthetic, pod_env, monkeypatch, tmp_path
):
    from minegs.core import provenance
    from minegs.eval.compare.execution import compare_execution

    monkeypatch.setattr(provenance, "git_commit", lambda *a, **k: "a" * 40)
    a = submit(synthetic, pod_env, name="a")
    b = submit(synthetic, pod_env, name="b")
    assert a.wait(poll_s=0.01) is RunStatus.SUCCEEDED and b.wait(poll_s=0.01) is RunStatus.SUCCEEDED
    rep = compare_execution(a.run_dir, b.run_dir)
    assert rep.reproducibility_pair and rep.differences == [] and rep.g3_status == "PENDING"
    assert rep.sides["a"]["remote_provider_execution"] and rep.sides["a"]["artifact_sync_verified"]
    assert rep.sides["a"]["real_gpu_training"] is False  # a stand-in trainer is not a GPU run
    # a different step budget is a different experiment: nothing is laid side by side
    c = submit(synthetic, pod_env, name="c", overrides={"max_steps": 20, "max_images": 4})
    assert c.wait(poll_s=0.01) is RunStatus.SUCCEEDED
    rep = compare_execution(a.run_dir, c.run_dir)
    assert not rep.reproducibility_pair and "max_steps" in rep.differences and rep.sides == {}
    # code that cannot be identified is not the same code
    monkeypatch.setattr(provenance, "git_commit", lambda *a, **k: "unknown")
    d = submit(synthetic, pod_env, name="d")
    assert d.wait(poll_s=0.01) is RunStatus.SUCCEEDED
    rep = compare_execution(a.run_dir, d.run_dir)
    assert not rep.reproducibility_pair and any("'unknown'" in x for x in rep.differences)
