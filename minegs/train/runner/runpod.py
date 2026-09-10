"""RunPodRunner (§8.2): create pod (network volume) -> ``sync.push`` (dataset only) ->
entrypoint -> poll -> ``sync.pull`` (ply, log) -> terminate. Checkpoints stay on the
volume; ``--resume`` picks them up. Same GPU image digest as LocalRunner.

Phase 1. The control flow is laid out here; API calls go through the ``runpod`` SDK
(``pip install 'minegs[runpod]'``, ``RUNPOD_API_KEY`` env).
"""

from __future__ import annotations

import os
from pathlib import Path

from minegs.core.errors import ContractError, MissingDependencyError, NotYetImplementedError
from minegs.train.backends import get_backend
from minegs.train.runner import sync
from minegs.train.runner.base import RunConfig, RunHandle, Runner, RunStatus


class RunPodHandle(RunHandle):
    def __init__(self, run_id: str, run_dir: Path, pod_id: str, remote: str) -> None:
        super().__init__(run_id, run_dir)
        self.pod_id = pod_id
        self.remote = remote

    def status(self) -> RunStatus:
        rp = _runpod()
        pod = rp.get_pod(self.pod_id)
        st = (pod or {}).get("desiredStatus", "")
        return {
            "RUNNING": RunStatus.RUNNING,
            "EXITED": RunStatus.SUCCEEDED,
            "TERMINATED": RunStatus.SUCCEEDED,
        }.get(st, RunStatus.UNKNOWN)

    def logs(self, tail: int | None = None) -> str:
        sync.pull(f"{self.remote}/runs/{self.run_id}/log", self.run_dir / "log")
        p = self.run_dir / "log" / "train.log"
        lines = p.read_text(errors="replace").splitlines() if p.exists() else []
        return "\n".join(lines[-tail:] if tail else lines)

    def fetch_artifacts(self) -> list[Path]:
        sync.pull(
            f"{self.remote}/runs/{self.run_id}",
            self.run_dir,
            include=("point_cloud/**", "log/**", "run.json"),
        )
        return sorted((self.run_dir / "point_cloud").glob("*.ply"))


class RunPodRunner(Runner):
    name = "runpod"

    def submit(self, run: RunConfig) -> RunHandle:
        run, manifest, profile, record = self.prepare(run)
        if not self.config.image_digest():
            raise ContractError("runner image must be pinned by digest (image@sha256:...) (§8.2)")
        if not os.environ.get("RUNPOD_API_KEY"):
            raise ContractError("RUNPOD_API_KEY is not set")
        remote = str(self.config.sync.get("remote", ""))
        if not remote:
            raise ContractError("runner.sync.remote (rclone remote) is required")
        backend = get_backend(run.backend or profile.backend)
        cmd = backend.build_command(
            Path(f"/data/datasets/{manifest.dataset_id}"),
            Path(f"/data/runs/{run.run_id}/backend_out"),
            profile,
            resume=run.resume,
        )
        record.command = cmd.argv
        record.T_local_from_internal = cmd.T_local_from_internal.to_list()
        self.write_record(record, Path(run.run_dir))
        # 1. dataset only (§1.4) — raw never leaves the workstation
        sync.push(Path(run.dataset_dir), f"{remote}/datasets/{manifest.dataset_id}")
        # 2. pod
        rp = _runpod()
        pod = rp.create_pod(
            name=f"minegs-{run.run_id}",
            image_name=self.config.image,
            gpu_type_id=self.config.gpu_types[0] if self.config.gpu_types else "NVIDIA RTX A5000",
            network_volume_id=self.config.network_volume_id,
            volume_mount_path=self.config.volume_mount,
            container_disk_in_gb=self.config.container_disk_gb,
            docker_args=" ".join(cmd.argv[1:])
            if cmd.argv and cmd.argv[0] == "minegs"
            else " ".join(cmd.argv),
            env={"MINEGS_RUN_ID": run.run_id, **cmd.env},
        )
        pod_id = pod["id"] if isinstance(pod, dict) else str(pod)
        record.status = RunStatus.RUNNING
        self.write_record(record, Path(run.run_dir))
        return RunPodHandle(run.run_id, Path(run.run_dir), pod_id, remote)

    def terminate(self, handle: RunPodHandle) -> None:
        _runpod().terminate_pod(handle.pod_id)


def _runpod():
    try:
        import runpod
    except ImportError as e:
        raise MissingDependencyError("runpod", "runpod", "RunPodRunner") from e
    runpod.api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not hasattr(runpod, "create_pod"):  # pragma: no cover
        raise NotYetImplementedError("runpod SDK surface changed; adapter update", "1")
    return runpod
