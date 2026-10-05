"""Phase 6 — RunPod execution, structurally: fake provider, temp volume, real everything else.

No billable call: the provider is ``FakeRunPodClient`` and the network volume is a temp
directory. Live RunPod execution: NOT PERFORMED.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from minegs.core.errors import ContractError
from minegs.core.provenance import sha256_file
from minegs.train.backends import get_backend
from minegs.train.remote import store as remote_store
from minegs.train.remote import worker as remote_worker
from minegs.train.remote.provider import ProviderAllocationError, ProviderError
from minegs.train.remote.secrets import redact
from minegs.train.remote.status import JobState, JobStatus
from minegs.train.runner import RunConfig
from minegs.train.runner import local as runner_local
from minegs.train.runner.base import (
    IMAGE_CUDA_ARCHS,
    RunnerConfig,
    RunStatus,
    get_runner,
    load_record,
)
from minegs.train.runner.runpod import RemoteExecutionRecord, RunPodRunner

from phase6_fakes import (
    IMAGE,
    SECRET,
    FakeRunPodClient,
    all_text,
    layout_of,
    read_json,
    run_cfg,
    runpod_config,
    submit,
    volume_files,
)
from test_gpu_baseline import FAKE_TRAINER

# ================================================================ the round trip


def test_a_runpod_run_round_trips_with_verified_evidence(synthetic, pod_env):
    h = submit(synthetic, pod_env)
    assert h.wait(poll_s=0.01) is RunStatus.SUCCEEDED
    rec = load_record(h.run_dir)
    assert rec.runner == "runpod" and rec.status is RunStatus.SUCCEEDED
    assert rec.image == IMAGE and rec.docker_digest == IMAGE.split("@")[1]
    assert rec.schema_version == "1.4"
    assert rec.remote_sync["pod_dataset_hash"] == rec.remote_sync["local_dataset_hash"]
    assert rec.remote_sync["pod_dataset_hash"] == rec.dataset_hash
    assert rec.remote_execution["provider"] == "runpod"
    assert rec.runtime["source"] == "runpod_pod"
    rem = RemoteExecutionRecord.load(h.run_dir / "remote.json")
    assert rem.artifact_sync_verified and rem.pulled_output_sha256
    assert rem.state == "succeeded" and rem.pod_id == "pod001"
    assert rem.chosen_gpu_type == "NVIDIA RTX A5000" and rem.remote_provider_execution
    assert rem.job_status["exit_code"] == 0 and rem.job_status["trainer_exit_code"] == 0
    assert rem.credential_source == "RUNPOD_API_KEY"
    # outputs arrived and the pod was terminated once its evidence was durable
    assert h.fetch_artifacts() and pod_env.client.terminated == ["pod001"]
    spec = pod_env.client.created[0]
    assert spec.image == IMAGE and spec.gpu_count == 1 and spec.network_volume_id == "vol123"
    assert spec.env == {} and SECRET not in spec.model_dump_json()
    assert spec.docker_args.startswith("train remote-worker ") and '"' not in spec.docker_args
    # the volume keeps the truth after the pod is gone
    layout = layout_of(pod_env, rec)
    assert (pod_env.volume / layout.job(rec.run_id) / "status.json").is_file()
    assert (pod_env.volume / layout.run(rec.run_id) / "output_manifest.json").is_file()
    assert SECRET not in all_text(pod_env.volume, h.run_dir)


def test_the_pod_computes_what_a_local_run_computes(synthetic, pod_env, tmp_path):
    """S8: same staging, command shape, trainer request, frame — only the provider differs."""
    h = submit(synthetic, pod_env)
    assert h.wait(poll_s=0.01) is RunStatus.SUCCEEDED
    remote = load_record(h.run_dir)
    local_dir = tmp_path / "runs" / "local"
    lh = get_runner("local", RunnerConfig(runner="local", native=True)).submit(
        run_cfg(synthetic, pod_env, "local")
    )
    assert lh.wait(poll_s=0.01) is RunStatus.SUCCEEDED
    local = load_record(local_dir)
    assert local.runner == "local" and local.remote_execution is None and local.remote_sync is None
    assert remote.dataset_hash == local.dataset_hash
    assert remote.staged["sha256"] == local.staged["sha256"]
    assert remote.staged["n_images"] == local.staged["n_images"]
    assert remote.expected_trainer_config == local.expected_trainer_config
    assert remote.profile == local.profile and remote.backend == local.backend
    assert remote.T_local_from_internal == local.T_local_from_internal
    assert remote.T_tls_from_local == local.T_tls_from_local

    def shape(rec, run_dir):
        return [a.replace(str(run_dir), "<RUN>") for a in rec.command]

    pod_run_dir = pod_env.volume / layout_of(pod_env, remote).run(remote.run_id)
    assert shape(remote, pod_run_dir) == shape(local, local_dir)


def test_a_local_run_is_unchanged_by_phase_6(synthetic, tmp_path, monkeypatch):
    """The LocalRunner command is still exactly the backend's command."""
    script = tmp_path / "simple_trainer.py"
    script.write_text(FAKE_TRAINER)
    script.with_suffix(".cfg.json").write_text("{}")
    monkeypatch.setenv("MINEGS_GSPLAT_TRAINER", str(script))
    monkeypatch.setattr(runner_local, "cuda_available", lambda: True)
    run_dir = tmp_path / "runs" / "plain"
    h = get_runner("local", RunnerConfig(runner="local", native=True, image=IMAGE)).submit(
        RunConfig(
            dataset_dir=str(synthetic.dataset_dir),
            profile="light",
            run_dir=str(run_dir),
            overrides={"max_steps": 10, "max_images": 4},
        )
    )
    assert h.wait(poll_s=0.01) is RunStatus.SUCCEEDED
    rec = load_record(run_dir)
    from minegs.train.profiles import load_profile

    prof = load_profile("light")
    prof.max_steps, prof.max_images = 10, 4
    cmd = get_backend("gsplat").build_command(run_dir / "staged", run_dir / "backend_out", prof)
    assert rec.command == cmd.argv and rec.runner == "local"
    assert rec.remote_execution is None and rec.remote_sync is None
    assert not (run_dir / "remote.json").exists()
    # a bare-environment run ran in no image, so it records none (the pod's worker does)
    assert rec.image is None and rec.docker_digest is None


# ================================================================ refused before any cost


@pytest.mark.parametrize(
    "why,over,match",
    [
        ("unpinned image", {"image": "ghcr.io/example/minegs:gpu"}, "pinned by digest"),
        ("short digest", {"image": "ghcr.io/example/minegs@sha256:abc"}, "pinned by digest"),
        ("not sha256", {"image": "ghcr.io/example/minegs@md5:" + "a" * 32}, "pinned by digest"),
        ("no volume", {"network_volume_id": None}, "network_volume_id"),
        ("two GPUs", {"gpu_count": 2}, "exactly one GPU"),
        ("relative mount", {"volume_mount": "data"}, "volume_mount"),
        ("root mount", {"volume_mount": "/"}, "volume_mount"),
        ("dotdot mount", {"volume_mount": "/data/../etc"}, "volume_mount"),
        ("no gpu types", {"gpu_types": []}, "gpu_types"),
        ("no remote", {"sync": {}}, "sync.remote"),
        ("bad remote", {"sync": {"remote": "not a remote"}}, "rclone remote"),
    ],
)
def test_an_incomplete_runner_config_costs_nothing(synthetic, pod_env, why, over, match):
    with pytest.raises(ContractError, match=match):
        submit(synthetic, pod_env, cfg=runpod_config(pod_env.volume, **over))
    assert pod_env.client.created == [] and volume_files(pod_env) == []
    assert not (pod_env.tmp / "runs" / "r1").exists()


def test_a_missing_credential_or_rclone_costs_nothing(synthetic, pod_env, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY")
    with pytest.raises(ContractError, match="RUNPOD_API_KEY is not set"):
        submit(synthetic, pod_env)
    monkeypatch.setenv("RUNPOD_API_KEY", SECRET)
    monkeypatch.setattr(remote_store.shutil, "which", lambda _name: None)
    with pytest.raises(ContractError, match="rclone is not on PATH"):
        submit(synthetic, pod_env, cfg=runpod_config(pod_env.volume, sync={"remote": "rs3:vol"}))
    assert pod_env.client.created == [] and volume_files(pod_env) == []


def test_bad_runs_are_refused_before_upload_or_pod(synthetic, pod_env, tmp_path):
    # a profile the pod cannot load, a profile needing depth without it, an unknown chunk
    prof = tmp_path / "mine.yaml"
    prof.write_text("schema_version: '1.0'\nname: mine\nbackend: gsplat\n")
    cases = [
        ({"profile": str(prof)}, "shipped profile"),
        ({"profile": "heavy"}, "DepthSupervisionRecord"),
        ({"chunk_id": "K000"}, "verified chunk plan"),
        ({"dataset_dir": synthetic.root / "raw"}, "raw"),
        ({"dataset_dir": synthetic.root}, "not a dataset"),
    ]
    for i, (over, match) in enumerate(cases):
        with pytest.raises(ContractError, match=match):
            submit(synthetic, pod_env, name=f"bad{i}", **over)
    assert pod_env.client.created == [] and volume_files(pod_env) == []


def test_the_default_runner_arch_list_is_the_images(tmp_path):
    text = (Path(__file__).resolve().parents[1] / "docker" / "Dockerfile.gpu").read_text()
    line = next(x for x in text.splitlines() if x.startswith("ARG TORCH_CUDA_ARCH_LIST="))
    assert tuple(line.split("=", 1)[1].strip('"').split(";")) == IMAGE_CUDA_ARCHS
    assert RunnerConfig().cuda_archs == list(IMAGE_CUDA_ARCHS)


def test_an_unallocatable_gpu_falls_through_then_fails_truthfully(synthetic, pod_env):
    pod_env.mode("run", alloc_fail=("NVIDIA RTX A5000",))
    h = submit(synthetic, pod_env)
    rem = RemoteExecutionRecord.load(h.run_dir / "remote.json")
    assert rem.chosen_gpu_type == "NVIDIA GeForce RTX 3090"
    assert [a["gpu_type"] for a in rem.allocation_attempts] == ["NVIDIA RTX A5000"]
    assert h.wait(poll_s=0.01) is RunStatus.SUCCEEDED

    pod_env.mode("alloc_fail")
    with pytest.raises(ProviderAllocationError, match="no pod could be allocated"):
        submit(synthetic, pod_env, name="r2")
    run_dir = pod_env.tmp / "runs" / "r2"
    rec, rem = load_record(run_dir), RemoteExecutionRecord.load(run_dir / "remote.json")
    assert rec.status is RunStatus.FAILED and "pod_allocation" in rec.failure_reason
    assert rem.failure_stage == "pod_allocation" and not rem.remote_provider_execution


# ================================================================ lifecycle is not success


def test_an_ended_pod_without_a_final_status_is_not_success(synthetic, pod_env):
    for i, m in enumerate(("terminated", "vanish")):
        pod_env.mode(m)
        h = submit(synthetic, pod_env, name=f"r{i}")
        assert h.status() is RunStatus.FAILED
        rec, rem = load_record(h.run_dir), RemoteExecutionRecord.load(h.run_dir / "remote.json")
        assert (
            rec.status is RunStatus.FAILED and rem.failure_stage == "pod_ended_without_final_status"
        )
        assert not rem.artifact_sync_verified and h.fetch_artifacts() == []


def test_a_running_pod_is_running_and_a_timeout_changes_nothing(synthetic, pod_env):
    client = pod_env.mode("running")
    h = submit(synthetic, pod_env)
    assert h.status() is RunStatus.PENDING
    rec = load_record(h.run_dir)
    job = pod_env.volume / layout_of(pod_env, rec).job(rec.run_id)
    JobStatus(run_id=rec.run_id, state=JobState.RUNNING).save(job / "status.json")
    assert h.wait(poll_s=0.01, timeout_s=0.05) is RunStatus.RUNNING
    assert client.terminated == [] and load_record(h.run_dir).status is RunStatus.PENDING
    client.mode = "timeout"
    assert h.status() is RunStatus.UNKNOWN
    assert client.terminated == [] and load_record(h.run_dir).status is RunStatus.PENDING


def test_a_cancelled_run_is_cancelled_and_a_failed_one_stays_failed(synthetic, pod_env):
    client = pod_env.mode("running")
    h = submit(synthetic, pod_env)
    RunPodRunner(runpod_config(pod_env.volume)).terminate(h)
    assert h.status() is RunStatus.CANCELLED and client.terminated == ["pod001"]
    rec = load_record(h.run_dir)
    assert rec.status is RunStatus.CANCELLED
    assert (
        pod_env.volume / layout_of(pod_env, rec).job(rec.run_id) / "cancel_requested.json"
    ).exists()

    pod_env.mode("run")
    pod_env.trainer(exit_code=3)
    h2 = submit(synthetic, pod_env, name="r2")
    assert h2.status() is RunStatus.FAILED
    h2.cancel()
    assert h2.status() is RunStatus.FAILED and load_record(h2.run_dir).status is RunStatus.FAILED


def test_a_cancel_whose_terminate_failed_is_not_cancelled(synthetic, pod_env):
    """The pod may still be running (and billing): the error is raised, the run is not marked
    cancelled, and a second cancel terminates it."""
    client = pod_env.mode("running")
    h = submit(synthetic, pod_env)
    real = client.terminate_pod

    def failing(pod_id):
        raise ProviderError("RunPod terminate_pod: 503")

    client.terminate_pod = failing
    with pytest.raises(ProviderError, match="503"):
        h.cancel()
    assert h.status() is RunStatus.PENDING and load_record(h.run_dir).status is RunStatus.PENDING
    rem = RemoteExecutionRecord.load(h.run_dir / "remote.json")
    assert rem.cancel_requested_at and rem.terminated_at is None
    assert any("terminate failed" in n for n in rem.notes)
    client.terminate_pod = real
    h.cancel()
    assert h.status() is RunStatus.CANCELLED and client.terminated == ["pod001"]


def test_a_run_is_fetched_with_the_image_it_was_submitted_with(synthetic, pod_env):
    pod_env.mode("running")
    h = submit(synthetic, pod_env)
    other = "ghcr.io/example/minegs:gpu@sha256:" + "cd" * 32
    with pytest.raises(ContractError, match="submitted with image"):
        RunPodRunner(runpod_config(pod_env.volume, image=other)).attach(h.run_dir)
    assert pod_env.client.start_container(pod_env.client.created[-1]) == 0
    assert RunPodRunner(runpod_config(pod_env.volume)).attach(h.run_dir).status() is (
        RunStatus.SUCCEEDED
    )


def test_a_trainer_that_exits_non_zero_fails_with_its_evidence(synthetic, pod_env):
    pod_env.trainer(exit_code=3)
    h = submit(synthetic, pod_env)
    assert h.wait(poll_s=0.01) is RunStatus.FAILED
    rem = RemoteExecutionRecord.load(h.run_dir / "remote.json")
    assert rem.job_status["trainer_exit_code"] == 3 and rem.job_status["exit_code"] == 1
    assert rem.failure_stage == "trainer_exit"
    # the failed run's evidence came back verified, unedited
    assert rem.artifact_sync_verified and load_record(h.run_dir).status is RunStatus.FAILED
    assert (h.run_dir / "log" / "train.log").is_file()


def test_exit_zero_without_outputs_is_a_failure(synthetic, pod_env):
    pod_env.trainer(write_ply=False)
    h = submit(synthetic, pod_env)
    assert h.wait(poll_s=0.01) is RunStatus.FAILED
    rem = RemoteExecutionRecord.load(h.run_dir / "remote.json")
    assert rem.job_status["trainer_exit_code"] == 0 and rem.failure_stage == "output_verification"


def _forge_status(env, rec, **update) -> None:
    p = env.volume / layout_of(env, rec).job(rec.run_id) / "status.json"
    js = JobStatus.load(p).model_copy(update=update)
    js.save(p)


def test_a_forged_success_status_does_not_make_a_success(synthetic, pod_env):
    # SUCCEEDED with a non-zero exit code
    pod_env.mode("running")
    h = submit(synthetic, pod_env)
    rec = load_record(h.run_dir)
    pod_env.client.start_container(pod_env.client.created[0])
    _forge_status(pod_env, rec, exit_code=1)
    assert h.status() is RunStatus.FAILED
    # the trainer exited 0 but the run failed its postconditions; the status is forged to success
    pod_env.mode("running")
    pod_env.trainer(write_ply=False)
    h = submit(synthetic, pod_env, name="r2")
    rec = load_record(h.run_dir)
    pod_env.client.start_container(pod_env.client.created[0])
    _forge_status(pod_env, rec, state=JobState.SUCCEEDED, exit_code=0, failure_stage=None)
    assert h.status() is RunStatus.FAILED
    rem = RemoteExecutionRecord.load(h.run_dir / "remote.json")
    assert (
        rem.failure_stage == "pull_verification" and "run.json says failed" in rem.failure_message
    )


def test_the_worker_never_trains_twice(synthetic, pod_env):
    pod_env.mode("running")
    h = submit(synthetic, pod_env)
    rec = load_record(h.run_dir)
    spec = pod_env.client.created[0]
    assert pod_env.client.start_container(spec) == 0
    run_dir = pod_env.volume / layout_of(pod_env, rec).run(rec.run_id)
    before = sha256_file(run_dir / "run.json")
    # the container restarts after finishing: nothing runs again
    assert pod_env.client.start_container(spec) == 0
    assert sha256_file(run_dir / "run.json") == before
    # a restart before the job finished records it and does not retry
    pod_env.mode("running")
    h2 = submit(synthetic, pod_env, name="r2")
    rec2 = load_record(h2.run_dir)
    job2 = pod_env.volume / layout_of(pod_env, rec2).job(rec2.run_id)
    (job2 / "claim").write_text("earlier attempt")
    assert pod_env.client.start_container(pod_env.client.created[0]) == 3
    js = JobStatus.load(job2 / "status.json")
    assert js.state.value == "failed" and js.failure_stage == "worker_restarted"
    assert h2.status() is RunStatus.FAILED


@pytest.mark.parametrize(
    "gpu,match",
    [
        ({"count": 1, "names": ["Old"], "capabilities": ["6.1"]}, "compute capability 6.1"),
        ({"count": 2, "names": ["a", "b"], "capabilities": ["8.6", "8.6"]}, "2 GPU"),
        ({"count": 0, "names": [], "capabilities": []}, "0 GPU"),
    ],
)
def test_the_pod_refuses_a_gpu_the_image_cannot_use(synthetic, pod_env, monkeypatch, gpu, match):
    monkeypatch.setattr(remote_worker, "visible_gpus", lambda: gpu)
    h = submit(synthetic, pod_env)
    assert h.wait(poll_s=0.01) is RunStatus.FAILED
    rem = RemoteExecutionRecord.load(h.run_dir / "remote.json")
    assert rem.job_status["failure_stage"] == "gpu_unsupported"
    assert match in rem.job_status["message"]
    rec = load_record(h.run_dir)
    assert not (pod_env.volume / layout_of(pod_env, rec).run(rec.run_id)).exists()  # never trained


# ================================================================ secrets


def test_credentials_never_reach_records_logs_or_errors(synthetic, pod_env, monkeypatch, capsys):
    monkeypatch.setenv("RCLONE_CONFIG_PASS", "rclone-pass-value-123")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret-value-456")
    secrets = (SECRET, "rclone-pass-value-123", "aws-secret-value-456")
    h = submit(synthetic, pod_env)
    assert h.wait(poll_s=0.01) is RunStatus.SUCCEEDED
    pod_env.trainer(exit_code=3)
    assert submit(synthetic, pod_env, name="r2").wait(poll_s=0.01) is RunStatus.FAILED

    class Leaky(FakeRunPodClient):
        def create_pod(self, spec):
            raise ProviderError(redact(f"RunPod create_pod: 401 for key {SECRET}"))

    pod_env.client = Leaky()
    with pytest.raises(ProviderError) as ei:
        submit(synthetic, pod_env, name="r3")
    text = all_text(pod_env.volume, pod_env.tmp / "runs") + str(ei.value) + capsys.readouterr().out
    for s in secrets:
        assert s not in text
    assert "RUNPOD_API_KEY" in text  # the name is recorded, the value is not
    assert redact("Authorization: Bearer abc.def") == "Authorization: ***"


@pytest.mark.parametrize(
    "text, leaked",
    [
        ("Authorization: Basic dXNlcjpwYXNzd29yZA==", "dXNlcjpwYXNzd29yZA"),
        ("authorization: Token tok-0123456789", "tok-0123456789"),
        (
            "Authorization: AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE/20260101/us/s3/aws4_request, "
            "SignedHeaders=host, Signature=feedfacecafe\nnext",
            "feedfacecafe",
        ),
        ('{"Authorization": "Bearer sk-live-abc", "x": 1}', "sk-live-abc"),
        ("{'authorization': 'Basic zzzzzz', 'a': 2}", "zzzzzz"),
        ("retrying with bearer abc.def-123", "abc.def-123"),
    ],
)
def test_any_authorization_header_is_redacted(text, leaked):
    out = redact(text, {})
    assert leaked not in out and "***" in out


def test_rclone_errors_are_redacted_before_they_are_shortened(tmp_path, monkeypatch):
    """A secret straddling the cut would survive a truncate-then-redact as an unmatched half."""
    from minegs.train.remote.store import RcloneStore

    secret = "s3-secret-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcd"
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)
    exe = tmp_path / "rclone"
    exe.write_text(f"#!/bin/sh\necho 'secret {secret}' >&2\nprintf '%0780d' 0 >&2\nexit 1\n")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    with pytest.raises(ContractError) as ei:
        RcloneStore("vol:x").exists("a")
    assert secret[-12:] not in str(ei.value) and "***" in str(ei.value)


# ================================================================ CLI


def _config_file(env, **over) -> Path:
    import yaml

    cfg = runpod_config(env.volume, **over)
    p = env.tmp / "runpod.yaml"
    p.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return p


def test_cli_dry_run_shows_the_request_and_costs_nothing(synthetic, pod_env):
    from minegs.cli.main import app
    from typer.testing import CliRunner

    cfg = _config_file(pod_env)
    argv = ["train", "run", str(synthetic.dataset_dir), "--runner", "runpod", "--config", str(cfg)]
    res = CliRunner().invoke(app, [*argv, "--dry-run"])
    assert res.exit_code == 0, res.output
    out = json.loads(res.output[res.output.index("{") :])
    assert out["pod_requests"][0]["image"] == IMAGE and out["inputs"]["bundle_digest"]
    assert out["credential_source"] == "RUNPOD_API_KEY" and SECRET not in res.output
    assert pod_env.client.created == [] and volume_files(pod_env) == []
    res = CliRunner().invoke(app, ["train", "run", str(synthetic.dataset_dir), "--dry-run"])
    assert res.exit_code == 2 and "train command" in res.output


def test_cli_fetch_cancel_and_the_worker_exit_code(synthetic, pod_env):
    from minegs.cli.main import app
    from typer.testing import CliRunner

    cfg = _config_file(pod_env)
    pod_env.mode("running")
    h = submit(synthetic, pod_env)
    res = CliRunner().invoke(app, ["train", "fetch", str(h.run_dir), "--config", str(cfg)])
    assert res.exit_code == 0 and "pending" in res.output
    # the container: minegs train remote-worker <inputs.json>; its exit code is the job's
    spec = pod_env.client.created[0]
    res = CliRunner().invoke(app, spec.docker_args.split())
    assert res.exit_code == 0, res.output
    res = CliRunner().invoke(app, ["train", "fetch", str(h.run_dir), "--config", str(cfg)])
    assert res.exit_code == 0 and "succeeded" in res.output and ".ply" in res.output
    # a second job, cancelled from the CLI
    h2 = submit(synthetic, pod_env, name="r2")
    res = CliRunner().invoke(app, ["train", "cancel", str(h2.run_dir), "--config", str(cfg)])
    assert res.exit_code == 0 and "cancelled" in res.output
    assert read_json(h2.run_dir / "run.json")["status"] == "cancelled"
    shutil.rmtree(pod_env.tmp / "runs")
