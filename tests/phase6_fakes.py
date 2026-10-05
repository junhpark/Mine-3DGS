"""Phase 6 test doubles: the RunPod API is the only thing substituted.

``FakeRunPodClient`` stands behind the same three calls as the SDK client. When a scenario says
the container runs, it does what the image's ENTRYPOINT would: ``minegs <docker_args>`` — here
``run_worker`` on the ``inputs.json`` path the pod request names — against a temp directory
standing in for the network volume. Uploads, hashing, the worker, the trainer stand-in, the
status file, the output manifest and the verified pull are all the production code on real bytes.
No billable call is possible: nothing here imports the SDK.
"""

from __future__ import annotations

import json
from pathlib import Path

from minegs.train.remote import worker as remote_worker
from minegs.train.remote.layout import RemoteLayout
from minegs.train.remote.provider import (
    PodInfo,
    PodSpec,
    ProviderAllocationError,
    ProviderUnavailableError,
    RunPodClient,
)
from minegs.train.runner import RunConfig
from minegs.train.runner import local as runner_local
from minegs.train.runner.base import RunnerConfig
from minegs.train.runner.runpod import RunPodRunner

DIGEST = "sha256:" + "ab" * 32
IMAGE = f"ghcr.io/example/minegs:gpu@{DIGEST}"
SECRET = "super-secret-test-value"
FAKE_GPU = {"count": 1, "names": ["Fake RTX A5000"], "capabilities": ["8.6"]}


class FakeRunPodClient(RunPodClient):
    """Scenarios: run (container runs the worker to completion, then EXITED), running (pod up,
    worker not yet done), vanish (pod disappears, nothing written), terminated (TERMINATED,
    nothing written), alloc_fail, timeout (every get_pod raises)."""

    def __init__(self, mode: str = "run", *, alloc_fail: tuple[str, ...] = ()) -> None:
        self.mode, self.alloc_fail = mode, alloc_fail
        self.created: list[PodSpec] = []
        self.pods: dict[str, PodInfo | None] = {}
        self.terminated: list[str] = []
        self.exit_codes: dict[str, int] = {}

    def create_pod(self, spec: PodSpec) -> PodInfo:
        if self.mode == "alloc_fail" or spec.gpu_type_id in self.alloc_fail:
            raise ProviderAllocationError(f"no instances available for {spec.gpu_type_id}")
        self.created.append(spec)
        pod_id = f"pod{len(self.created):03d}"
        info = PodInfo(
            pod_id=pod_id,
            desired_status="RUNNING",
            gpu_display_name="RTX A5000",
            gpu_count=spec.gpu_count,
            image=spec.image,
            cost_per_hr=0.29,
        )
        self.pods[pod_id] = info
        if self.mode == "run":
            self.exit_codes[pod_id] = self.start_container(spec)
            self.pods[pod_id] = info.model_copy(update={"desired_status": "EXITED"})
        elif self.mode == "vanish":
            self.pods[pod_id] = None
        elif self.mode == "terminated":
            self.pods[pod_id] = info.model_copy(update={"desired_status": "TERMINATED"})
        return info

    @staticmethod
    def start_container(spec: PodSpec) -> int:
        from minegs.train.remote.worker import run_worker

        args = spec.docker_args.split()
        assert args[:2] == ["train", "remote-worker"], args
        return run_worker(args[2], poll_s=0.01)

    def get_pod(self, pod_id: str) -> PodInfo | None:
        if self.mode == "timeout":
            raise ProviderUnavailableError("RunPod get_pod: timed out")
        return self.pods.get(pod_id)

    def terminate_pod(self, pod_id: str) -> None:
        self.terminated.append(pod_id)
        if self.pods.get(pod_id) is not None:
            self.pods[pod_id] = self.pods[pod_id].model_copy(
                update={"desired_status": "TERMINATED"}
            )


def runpod_config(volume: Path, **over) -> RunnerConfig:
    return RunnerConfig(
        runner="runpod",
        image=over.pop("image", IMAGE),
        gpu_types=over.pop("gpu_types", ["NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090"]),
        network_volume_id=over.pop("network_volume_id", "vol123"),
        volume_mount=over.pop("volume_mount", str(volume)),
        container_disk_gb=40,
        sync=over.pop("sync", {"remote": str(volume)}),
        **over,
    )


def all_text(*roots: Path) -> str:
    """Every byte under the given directories, as text, for secret scanning."""
    out = []
    for root in roots:
        for p in Path(root).rglob("*"):
            if p.is_file():
                out.append(p.read_bytes().decode(errors="replace"))
    return "\n".join(out)


def read_json(p: Path) -> dict:
    return json.loads(Path(p).read_text())


class Env:
    pass


def make_pod_env(tmp_path, monkeypatch):
    """A volume, a trainer stand-in, a GPU as far as the worker can tell, and a credential."""
    from test_gpu_baseline import FAKE_TRAINER

    volume = tmp_path / "volume"
    volume.mkdir()
    script = tmp_path / "simple_trainer.py"
    script.write_text(FAKE_TRAINER)
    script.with_suffix(".cfg.json").write_text("{}")
    monkeypatch.setenv("MINEGS_GSPLAT_TRAINER", str(script))
    monkeypatch.setenv("RUNPOD_API_KEY", SECRET)
    monkeypatch.setattr(runner_local, "cuda_available", lambda: True)
    monkeypatch.setattr(remote_worker, "visible_gpus", lambda: dict(FAKE_GPU))
    env = Env()
    env.volume, env.tmp = volume, tmp_path
    env.client = FakeRunPodClient("run")
    monkeypatch.setattr(RunPodRunner, "client_factory", staticmethod(lambda: env.client))

    def mode(m: str, **kw):
        env.client = FakeRunPodClient(m, **kw)
        return env.client

    def trainer(**cfg):
        script.with_suffix(".cfg.json").write_text(json.dumps(cfg))

    env.mode, env.trainer = mode, trainer
    return env


def run_cfg(synthetic, env, name="r1", **over):
    return RunConfig(
        dataset_dir=str(over.pop("dataset_dir", None) or synthetic.dataset_dir),
        profile=over.pop("profile", "light"),
        run_dir=str(env.tmp / "runs" / name),
        overrides=over.pop("overrides", {"max_steps": 10, "max_images": 4}),
        **over,
    )


def submit(synthetic, env, name="r1", cfg=None, **over):
    return RunPodRunner(cfg or runpod_config(env.volume)).submit(
        run_cfg(synthetic, env, name, **over)
    )


def layout_of(env, rec_or_ds):
    return RemoteLayout(str(env.volume), getattr(rec_or_ds, "dataset_id", rec_or_ds))


def volume_files(env) -> list[Path]:
    return sorted(p for p in env.volume.rglob("*") if p.is_file())
