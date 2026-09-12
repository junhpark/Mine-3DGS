"""LocalRunner (§8.2): ``docker run --gpus all <image@digest> train run ...``.
Refuses without CUDA and suggests RunPod. ``native=True`` runs the backend command in the
current environment (developer mode, no digest recorded)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from minegs.core.errors import ContractError, NoGpuError
from minegs.core.provenance import sha256_tree
from minegs.train.backends import get_backend
from minegs.train.runner.base import (
    RunConfig,
    RunHandle,
    Runner,
    RunStatus,
    cuda_available,
    docker_available,
    load_record,
)
from minegs.train.runner.resume import (
    check_staged_compatibility,
    container_checkpoint,
    docker_resume_mount,
    guard_resume_argv,
)
from minegs.train.staging import stage_dataset


class LocalHandle(RunHandle):
    def __init__(self, run_id: str, run_dir: Path, proc: subprocess.Popen | None) -> None:
        super().__init__(run_id, run_dir)
        self._proc = proc

    def status(self) -> RunStatus:
        if self._proc is None:
            try:
                return load_record(self.run_dir).status
            except ContractError:
                return RunStatus.UNKNOWN
        rc = self._proc.poll()
        if rc is None:
            return RunStatus.RUNNING
        return RunStatus.SUCCEEDED if rc == 0 else RunStatus.FAILED

    def logs(self, tail: int | None = None) -> str:
        p = self.run_dir / "log" / "train.log"
        if not p.exists():
            return ""
        lines = p.read_text(errors="replace").splitlines()
        return "\n".join(lines[-tail:] if tail else lines)

    def fetch_artifacts(self) -> list[Path]:
        return sorted((self.run_dir / "point_cloud").glob("*.ply"))


class LocalRunner(Runner):
    name = "local"

    def submit(self, run: RunConfig) -> RunHandle:
        run, manifest, profile, record = self.prepare(run)
        run_dir = Path(run.run_dir)
        dataset_dir = Path(run.dataset_dir).resolve()
        backend = get_backend(run.backend or profile.backend)
        work = run_dir / "backend_out"
        (run_dir / "log").mkdir(parents=True, exist_ok=True)

        if not cuda_available():
            raise NoGpuError(
                "No CUDA device found. Training needs a GPU. Run on a CUDA host (docker with "
                "--gpus all), or use --native for a developer run in the current environment. "
                "Cloud routing (RunPod) is Phase 6 and not implemented yet — see docs/ROADMAP.md."
            )

        staged = stage_dataset(
            dataset_dir,
            run_dir / "staged",
            manifest,
            max_images=profile.max_images,
            chunk_id=run.chunk_id,
        )
        record.staged = {
            "path": "staged",
            "n_images": len(staged.images),
            "n_train_available": staged.n_train_available,
            "subset": staged.subset,
            "init_source": staged.init_source,
            "init_points": staged.init_points,
            "sha256": sha256_tree(staged.path, ("sparse/0/*.txt", "images/**/*", "masks/**/*")),
        }
        # The last resume check, and the only one that needs staging: the trainer's actual input
        # must be what the parent trained on (§25). Still ahead of Popen — nothing has run yet.
        host_checkpoint = None
        if record.resume is not None:
            check_staged_compatibility(load_record(record.resume.parent_run_dir), record.staged)
            host_checkpoint = Path(record.resume.checkpoint)
        resume_mount: list[str] = []
        if self.config.native:
            # Native execution shares the host namespace, so the validated path is the executed
            # path; nothing is translated and nothing is mounted.
            exec_checkpoint = host_checkpoint
            cmd = backend.build_command(
                staged.path, work, profile, resume_checkpoint=exec_checkpoint
            )
            argv = cmd.argv
        else:
            if not docker_available():
                raise ContractError(
                    "docker not found; use --native for a bare environment (no digest will be recorded)"
                )
            if not self.config.image_digest():
                raise ContractError(
                    "runner image must be pinned by digest (image@sha256:...) for reproducibility (§8.2)"
                )
            # The Phase 0D entry blocker: the checkpoint was *found* on the host but the trainer
            # runs in the container, so the two namespaces must be handled separately (§17). The
            # parent's checkpoint directory — nothing more of the parent run, and read-only — is
            # mounted at /data/resume, and that is the path the trainer receives.
            exec_checkpoint = (
                container_checkpoint(host_checkpoint) if host_checkpoint is not None else None
            )
            if host_checkpoint is not None:
                resume_mount = docker_resume_mount(host_checkpoint)
            cmd = backend.build_command(
                Path("/data/run/staged"),
                Path("/data/run/backend_out"),
                profile,
                resume_checkpoint=exec_checkpoint,
                check_trainer=False,  # the trainer lives inside the image
            )
            argv = [
                "docker",
                "run",
                "--rm",
                "--gpus",
                self.config.gpus,
                "--shm-size",
                self.config.shm_size,
                "-v",
                f"{dataset_dir}:/data/dataset:ro",
                "-v",
                f"{run_dir.resolve()}:/data/run",
                *resume_mount,
                "--entrypoint",
                cmd.argv[0],
                self.config.image,
                *cmd.argv[1:],
            ]
        # §16: whatever route built the command, a requested resume must be carried by the argv
        # that is about to be executed. Without this, a backend that quietly dropped the flag
        # would start a fresh run under a run id whose run.json claims a parent.
        guard_resume_argv(argv, exec_checkpoint)
        if record.resume is not None:
            record.resume.checkpoint_exec_path = str(exec_checkpoint)
        record.command = argv
        record.T_local_from_internal = cmd.T_local_from_internal.to_list()
        record.status = RunStatus.RUNNING
        self.write_record(record, run_dir)
        log = open(run_dir / "log" / "train.log", "ab")  # noqa: SIM115 — handed to Popen
        proc = subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT, env={**_env(), **cmd.env}
        )
        handle = LocalHandle(run.run_id, run_dir, proc)
        return _FinalizingHandle(handle, backend, work, run_dir, record)


class _FinalizingHandle(RunHandle):
    """Wraps LocalHandle: on completion, normalise outputs into the run convention."""

    def __init__(self, inner: LocalHandle, backend, work: Path, run_dir: Path, record) -> None:
        super().__init__(inner.run_id, run_dir)
        self._inner, self._backend, self._work, self._record = inner, backend, work, record
        self._finalized = False

    def status(self) -> RunStatus:
        st = self._inner.status()
        if st in (RunStatus.SUCCEEDED, RunStatus.FAILED) and not self._finalized:
            self._finalized = True
            self._record.status = st
            if st == RunStatus.SUCCEEDED:
                from minegs.core.frames import Sim3

                plys = self._backend.normalize_outputs(
                    self._work, self.run_dir, Sim3.from_matrix(self._record.T_local_from_internal)
                )
                self._record.outputs = [str(p.relative_to(self.run_dir)) for p in plys]
            Runner.write_record(self._record, self.run_dir)
        return st

    def logs(self, tail: int | None = None) -> str:
        return self._inner.logs(tail)

    def fetch_artifacts(self) -> list[Path]:
        self.status()
        return self._inner.fetch_artifacts()


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)
