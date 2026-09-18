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
    RunRecord,
    RunStatus,
    cuda_available,
    docker_available,
    load_record,
    runtime_info,
    verify_postconditions,
)
from minegs.train.staging import stage_dataset


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _elapsed(start: str, end: str) -> float | None:
    from datetime import datetime

    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except ValueError:
        return None


def _record_evidence(rec: RunRecord, ev, run_dir: Path) -> None:
    """Copy what the backend found into the record, as paths relative to the run directory.

    ``normalize_outputs`` has already copied ``ckpts`` -> ``ckpt`` and ``stats`` -> ``stats``, so
    the evidence is named where a reader of ``run.json`` will actually find it rather than in the
    backend's scratch tree — which on the docker path is a container-side path anyway.
    """

    def rel(p: Path | None) -> str | None:
        if p is None:
            return None
        try:
            return str(Path("ckpt") / p.name) if p.suffix == ".pt" else str(Path(p).name)
        except Exception:  # pragma: no cover - defensive
            return str(p)

    rec.checkpoints = [str(Path("ckpt") / c.name) for c in ev.checkpoints]
    rec.renders = [str(Path("renders") / r.name) for r in ev.renders]
    rec.final_checkpoint = rel(ev.final_checkpoint)
    rec.checkpoint_step = ev.checkpoint_step
    rec.final_model = (
        str(Path("point_cloud") / ev.final_model.name) if ev.final_model is not None else None
    )
    rec.observed_final_step = ev.observed_final_step
    rec.peak_gpu_memory_gb = ev.peak_gpu_memory_gb
    rec.train_seconds = ev.train_seconds
    if ev.gaussian_count is not None:
        rec.gaussian_count = ev.gaussian_count


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
        # No resume branch here on purpose: --resume-from is refused in Runner.prepare, before
        # this method runs and before the run directory exists (docs/ROADMAP.md §Phase 0D).
        if self.config.native:
            cmd = backend.build_command(staged.path, work, profile)
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
            cmd = backend.build_command(
                Path("/data/run/staged"),
                Path("/data/run/backend_out"),
                profile,
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
                "--entrypoint",
                cmd.argv[0],
                self.config.image,
                *cmd.argv[1:],
            ]
        record.command = argv
        record.command_env = dict(cmd.env)
        record.T_local_from_internal = cmd.T_local_from_internal.to_list()
        record.image = self.config.image or None
        record.max_steps = profile.max_steps
        record.runtime = runtime_info()
        record.started_at = _now()
        record.status = RunStatus.RUNNING
        self.write_record(record, run_dir)
        log = open(run_dir / "log" / "train.log", "ab")  # noqa: SIM115 — closed in _finalize
        proc = subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT, env={**_env(), **cmd.env}
        )
        handle = LocalHandle(run.run_id, run_dir, proc)
        return _FinalizingHandle(handle, backend, work, run_dir, record, profile, log)


class _FinalizingHandle(RunHandle):
    """Wraps LocalHandle: on completion, verify the run before calling it one.

    The trainer's exit code opens the question rather than answering it. A gsplat run that
    writes no checkpoint, no PLY, or gaussians at infinity exits 0 exactly like one that
    trained, so SUCCEEDED is published only after ``verify_postconditions`` has read the
    artifacts back (§0D.2 D2-4..D2-7, §11). Anything that fails there makes the run FAILED with
    the reason recorded — never a success with a caveat.
    """

    def __init__(
        self, inner: LocalHandle, backend, work: Path, run_dir: Path, record, profile, log=None
    ) -> None:
        super().__init__(inner.run_id, run_dir)
        self._inner, self._backend, self._work, self._record = inner, backend, work, record
        self._profile, self._log = profile, log
        #: The resolved terminal status, or None while the run is still in flight. A bare
        #: "have I finalized" latch would let a finalization that *failed* be reported as the
        #: process's own exit status on the next call.
        self._terminal: RunStatus | None = None

    def status(self) -> RunStatus:
        if self._terminal is not None:
            return self._terminal
        st = self._inner.status()
        if st not in (RunStatus.SUCCEEDED, RunStatus.FAILED):
            return st
        return self._finalize(st)

    def _finalize(self, exit_status: RunStatus) -> RunStatus:
        from minegs.core.frames import Sim3

        if self._log is not None:
            self._log.close()
            self._log = None
        rec = self._record
        rec.completed_at = _now()
        if rec.started_at:
            rec.duration_s = _elapsed(rec.started_at, rec.completed_at)

        if exit_status is RunStatus.FAILED:
            rec.status = RunStatus.FAILED
            rec.failure_reason = "the trainer exited non-zero; see log/train.log"
        else:
            try:
                plys = self._backend.normalize_outputs(
                    self._work, self.run_dir, Sim3.from_matrix(rec.T_local_from_internal)
                )
                ev = self._backend.collect_evidence(self._work, self._profile)
                _record_evidence(rec, ev, self.run_dir)
                verify_postconditions(rec, ev, self.run_dir, self.run_dir / "staged", plys)
            except ContractError as e:
                rec.status = RunStatus.FAILED
                rec.failure_reason = str(e)
            else:
                rec.outputs = [str(p.relative_to(self.run_dir)) for p in plys]
                rec.status = RunStatus.SUCCEEDED

        Runner.write_record(rec, self.run_dir)
        self._terminal = rec.status
        return self._terminal

    def logs(self, tail: int | None = None) -> str:
        return self._inner.logs(tail)

    def fetch_artifacts(self) -> list[Path]:
        self.status()
        return self._inner.fetch_artifacts()


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)
