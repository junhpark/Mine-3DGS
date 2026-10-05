"""The pod side of a RunPod run (Phase 6 §6, §8): ``minegs train remote-worker <inputs.json>``.

Orchestration only. The scientific execution is LocalRunner's native path, unchanged: the same
``prepare`` (profile, capability, depth and chunk-plan verification), ``stage_dataset``,
``build_command``, trainer, postconditions and ``RunRecord``. ``RemoteWorkerRunner`` differs in
its name — so ``run.json`` says ``runner: runpod``, which is who executed it — and in adding the
remote evidence the pod knows.

Before any trainer starts, the worker re-derives everything it was told: the bundle digest, the
dataset hash of the bytes it is about to read, the chunk plan and depth artifact by identity, the
one visible GPU and its architecture, an empty run directory. Every outcome is written to
``status.json`` on the volume, atomically, with the exit codes; that file — not the pod's
lifecycle — is what the submitter reads.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from minegs.core.errors import ContractError
from minegs.core.provenance import sha256_file
from minegs.train.remote.bundle import (
    RunInputBundle,
    dataset_file_table,
    dataset_hash_of,
    load_bundle,
)
from minegs.train.remote.layout import (
    CHUNK_PLAN_FILE,
    CLAIM_FILE,
    OUTPUT_MANIFEST_FILE,
    STATUS_FILE,
    WORKER_LOG,
    RemoteLayout,
)
from minegs.train.remote.secrets import redact
from minegs.train.remote.status import (
    JobState,
    JobStatus,
    RemoteOutputRecord,
    now,
    output_entries,
    tree_digest,
    write_json_atomic,
)
from minegs.train.runner.base import RunConfig, RunnerConfig, RunStatus, load_record
from minegs.train.runner.local import LocalRunner

#: Exit codes of the worker process (and of the pod's container).
EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_REFUSED = 2
EXIT_RESTARTED = 3


class GpuUnsupportedError(ContractError):
    pass


class InputChangedError(ContractError):
    """The dataset on the volume changed after the pod verified it."""


def visible_gpus() -> dict[str, Any]:
    """What CUDA shows this process: count, names, compute capabilities."""
    try:
        import torch
    except ImportError:
        return {"count": 0, "names": [], "capabilities": [], "error": "torch is not importable"}
    if not torch.cuda.is_available():
        return {"count": 0, "names": [], "capabilities": []}
    n = torch.cuda.device_count()
    caps = []
    for i in range(n):
        major, minor = torch.cuda.get_device_capability(i)
        caps.append(f"{major}.{minor}")
    return {
        "count": n,
        "names": [torch.cuda.get_device_name(i) for i in range(n)],
        "capabilities": caps,
    }


class _Job:
    def __init__(self, job_dir: Path, run_id: str) -> None:
        self.dir, self.run_id = job_dir, run_id
        self.status = JobStatus(run_id=run_id, state=JobState.STARTING, started_at=now())

    def log(self, msg: str) -> None:
        with open(self.dir / WORKER_LOG, "a") as f:
            f.write(f"{now()} {redact(msg)}\n")

    def write(self, **update: Any) -> None:
        if "message" in update and update["message"] is not None:
            update["message"] = redact(update["message"])[-4000:]
        self.status = self.status.model_copy(update=update)
        write_json_atomic(self.dir / STATUS_FILE, self.status)

    def finish(self, ok: bool, exit_code: int, **update: Any) -> int:
        self.write(
            state=JobState.SUCCEEDED if ok else JobState.FAILED,
            completed_at=now(),
            exit_code=exit_code,
            **update,
        )
        self.log(f"final: {self.status.state.value} exit={exit_code} {update.get('message') or ''}")
        return exit_code


class RemoteWorkerRunner(LocalRunner):
    """LocalRunner's native execution, recorded as what it is: a RunPod run."""

    name = "runpod"
    native_runs_in_image = True

    def __init__(
        self,
        config: RunnerConfig,
        bundle: RunInputBundle,
        pod_dataset_hash: str,
        gpu: dict,
        table: dict[str, str] | None = None,
    ) -> None:
        super().__init__(config)
        self._bundle, self._pod_hash, self._gpu = bundle, pod_dataset_hash, gpu
        self._table = table

    def _after_staging(self, dataset_dir: Path) -> None:
        """Staging has read ``sparse/0``, ``init_points.ply``, images and masks from the volume
        after ``prepare`` hashed it. Hash the whole dataset again before the trainer starts, so
        what the trainer is given is the dataset the record names (images and masks are checked
        once more after the run: staging hard-links them)."""
        table = dataset_file_table(Path(dataset_dir))
        got = dataset_hash_of(table)
        if got != self._pod_hash:
            want = self._table or {}
            changed = sorted(k for k in set(table) | set(want) if table.get(k) != want.get(k))
            raise InputChangedError(
                f"the dataset on the volume changed while it was being staged ({self._pod_hash[:12]}"
                f" -> {got[:12]}; {changed[:5]}); the staged inputs are not the dataset the run "
                "records. The trainer was not started"
            )

    def prepare(self, run: RunConfig):
        run, manifest, profile, record = super().prepare(run)
        b = self._bundle
        # prepare hashed the dataset again; it must still be the bytes the pod verified
        if record.dataset_hash != self._pod_hash:
            raise ContractError(
                f"the dataset changed while the pod was starting ({self._pod_hash[:12]} -> "
                f"{record.dataset_hash[:12]})"
            )
        if b.chunk is not None and (record.chunk or {}).get("plan_digest") != b.chunk.plan_digest:
            raise ContractError("the verified chunk plan is not the one the input bundle names")
        if (
            b.depth_supervision is not None
            and (record.depth_supervision or {}).get("artifact_sha256")
            != b.depth_supervision.artifact_sha256
        ):
            raise ContractError("the verified depth artifact is not the one the input bundle names")
        layout = RemoteLayout(b.volume_mount, b.dataset_id)
        record.remote_execution = {
            "provider": "runpod",
            "pod_id": os.environ.get("RUNPOD_POD_ID") or None,
            "requested_gpu_types": list(b.requested_gpu_types),
            "gpu_count": b.gpu_count,
            "visible_gpus": self._gpu,
            "network_volume_id": b.network_volume_id,
            "volume_mount": b.volume_mount,
            "job_path": layout.job(b.run_id),
            "run_path": layout.run(b.run_id),
            "image": b.image,
        }
        record.remote_sync = {
            "input_bundle_digest": b.bundle_digest,
            "local_dataset_hash": b.dataset_hash,
            "pod_dataset_hash": self._pod_hash,
            "dataset_files": b.dataset_files,
            "depth_artifact_sha256": None
            if b.depth_supervision is None
            else b.depth_supervision.artifact_sha256,
            "chunk_plan_digest": None if b.chunk is None else b.chunk.plan_digest,
        }
        return run, manifest, profile, record


def verify_pod_inputs(bundle: RunInputBundle) -> tuple[str, dict[str, Any], dict[str, str]]:
    """Everything the bundle claims, re-derived from what the pod sees. Returns the dataset hash,
    the GPU description and the verified file table. Nothing here trusts the submitter's checks."""
    from minegs.chunks.plan import load_chunk_plan, verify_chunk_plan
    from minegs.train.supervision.depth import verify_depth_supervision

    layout = RemoteLayout(bundle.volume_mount, bundle.dataset_id)
    want = {
        "dataset": layout.pod(layout.dataset(bundle.dataset_hash)),
        "job": layout.pod(layout.job(bundle.run_id)),
        "run": layout.pod(layout.run(bundle.run_id)),
    }
    if bundle.paths != want:
        raise ContractError(f"inputs.json names paths {bundle.paths}, not the layout's {want}")

    gpu = visible_gpus()
    if gpu.get("count") != 1:
        raise GpuUnsupportedError(
            f"the pod shows {gpu.get('count')} GPU(s); a run trains on exactly one (gsplat goes "
            "distributed on sight of a second)"
        )
    if gpu["capabilities"][0] not in bundle.cuda_archs:
        raise GpuUnsupportedError(
            f"GPU {gpu['names'][0]} has compute capability {gpu['capabilities'][0]}, which the "
            f"image is not built for ({bundle.cuda_archs})"
        )

    ds = Path(want["dataset"])
    if not ds.is_dir():
        raise ContractError(f"{ds}: the dataset is not on the volume")
    table = dataset_file_table(ds)
    pod_hash = dataset_hash_of(table)
    if pod_hash != bundle.dataset_hash or len(table) != bundle.dataset_files:
        raise ContractError(
            f"the dataset on the volume hashes to {pod_hash[:12]} over {len(table)} files; the "
            f"bundle claims {bundle.dataset_hash[:12]} over {bundle.dataset_files}. The bytes the "
            "trainer would read are not the dataset that was submitted"
        )
    run_dir = Path(want["run"])
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ContractError(f"{run_dir} already holds files; a run directory is written once")

    if bundle.chunk is not None:
        plan_file = Path(bundle.chunk.path)
        if plan_file != Path(layout.pod(layout.chunk_plan(bundle.chunk.plan_digest))) / (
            CHUNK_PLAN_FILE
        ):
            raise ContractError(f"chunk plan path {plan_file} is not its digest-addressed path")
        plan = load_chunk_plan(plan_file)
        if plan.plan_digest != bundle.chunk.plan_digest or plan.plan_id != bundle.chunk.plan_id:
            raise ContractError(
                f"the chunk plan on the volume is {plan.plan_id} ({plan.plan_digest[:12]}), not "
                f"{bundle.chunk.plan_id} ({bundle.chunk.plan_digest[:12]})"
            )
        verify_chunk_plan(ds, plan_file)
    if bundle.depth_supervision is not None:
        art = Path(bundle.depth_supervision.path)
        if art != Path(layout.pod(layout.depth(bundle.depth_supervision.artifact_sha256))):
            raise ContractError(f"depth artifact path {art} is not its digest-addressed path")
        verified = verify_depth_supervision(ds, art)
        if verified.artifact_sha256 != bundle.depth_supervision.artifact_sha256:
            raise ContractError(
                f"the depth artifact on the volume is {verified.artifact_sha256[:12]}, not "
                f"{bundle.depth_supervision.artifact_sha256[:12]}"
            )
    return pod_hash, gpu, table


def verify_staged_inputs(run_dir: Path, table: dict[str, str]) -> None:
    """The images and masks the trainer read are the verified dataset's bytes.

    Staging links (or copies) them out of the dataset on the volume after the pod hashed it;
    checking them again after the run closes the window in which the volume could have changed
    under the trainer.
    """
    staged = Path(run_dir) / "staged"
    for sub in ("images", "masks"):
        root = staged / sub
        if not root.is_dir():
            continue
        for f in sorted(p for p in root.rglob("*") if p.is_file()):
            rel = f"{sub}/{f.relative_to(root).as_posix()}"
            want = table.get(rel)
            if want is None or sha256_file(f) != want:
                raise ContractError(
                    f"staged {rel} is not the verified dataset's file; the bytes the trainer "
                    "read are not the dataset that was submitted"
                )


def write_output_manifest(
    run_dir: Path, bundle: RunInputBundle, status: str, exit_code: int
) -> str:
    entries = output_entries(run_dir, OUTPUT_MANIFEST_FILE)
    run_json = run_dir / "run.json"
    rec = RemoteOutputRecord(
        run_id=bundle.run_id,
        status=status,
        exit_code=exit_code,
        dataset_hash=bundle.dataset_hash,
        image=bundle.image,
        run_json_sha256=sha256_file(run_json) if run_json.is_file() else None,
        entries=entries,
        output_tree_digest=tree_digest(entries),
        created_at=now(),
    )
    return sha256_file(write_json_atomic(run_dir / OUTPUT_MANIFEST_FILE, rec))


def run_worker(inputs: str | Path, poll_s: float = 5.0) -> int:
    """Run one job from its published ``inputs.json``. Returns the process exit code."""
    inputs = Path(inputs)
    job_dir = inputs.parent
    try:
        bundle = load_bundle(inputs)
    except ContractError as e:
        if job_dir.is_dir() and not (job_dir / STATUS_FILE).exists():
            j = _Job(job_dir, job_dir.name)
            return j.finish(False, EXIT_REFUSED, failure_stage="input_bundle", message=str(e))
        raise

    job = _Job(job_dir, bundle.run_id)
    try:
        fd = os.open(job_dir / CLAIM_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # The container started again (RunPod restarts an exited container). Never train twice:
        # a second attempt would be a fresh run under the first one's id, and nothing resumes.
        prior = JobStatus.load(job_dir / STATUS_FILE) if (job_dir / STATUS_FILE).exists() else None
        if prior is not None and prior.final:
            job.log("container restarted after the job finished; nothing to do")
            return int(prior.exit_code if prior.exit_code is not None else EXIT_RUN_FAILED)
        job.status = prior or job.status
        return job.finish(
            False,
            EXIT_RESTARTED,
            failure_stage="worker_restarted",
            message="the worker started again before the job finished; it is not retried (no resume)",
        )
    os.write(fd, now().encode())
    os.close(fd)

    job.write(state=JobState.STARTING, input_bundle_digest=bundle.bundle_digest)
    job.log(f"job {bundle.run_id}: inputs {bundle.bundle_digest[:12]}")
    run_dir = Path(bundle.paths["run"])
    try:
        try:
            pod_hash, gpu, table = verify_pod_inputs(bundle)
        except GpuUnsupportedError as e:
            return job.finish(False, EXIT_REFUSED, failure_stage="gpu_unsupported", message=str(e))
        except ContractError as e:
            return job.finish(
                False, EXIT_REFUSED, failure_stage="input_verification", message=str(e)
            )
        job.write(state=JobState.RUNNING, pod_dataset_hash=pod_hash)
        job.log(f"inputs verified on the pod: dataset {pod_hash[:12]}, GPU {gpu}")

        config = RunnerConfig(
            runner="runpod",
            image=bundle.image,
            native=True,
            gpus="device=0",
            cuda_archs=list(bundle.cuda_archs),
        )
        runner = RemoteWorkerRunner(config, bundle, pod_hash, gpu, table)
        try:
            handle = runner.submit(
                RunConfig(
                    run_id=bundle.run_id,
                    dataset_dir=bundle.paths["dataset"],
                    run_dir=str(run_dir),
                    profile=bundle.profile,
                    backend=bundle.backend,
                    runner="runpod",
                    chunk_id=None if bundle.chunk is None else bundle.chunk.chunk_id,
                    chunk_plan=None if bundle.chunk is None else bundle.chunk.path,
                    depth_supervision=None
                    if bundle.depth_supervision is None
                    else bundle.depth_supervision.path,
                    overrides=dict(bundle.overrides),
                )
            )
        except InputChangedError as e:
            # No output manifest: there is no run to return, only the refusal.
            return job.finish(
                False, EXIT_REFUSED, failure_stage="input_verification", message=str(e)
            )
        except ContractError as e:
            manifest_sha = (
                write_output_manifest(run_dir, bundle, "failed", EXIT_REFUSED)
                if run_dir.is_dir()
                else None
            )
            return job.finish(
                False,
                EXIT_REFUSED,
                failure_stage="training_setup",
                message=str(e),
                output_manifest_sha256=manifest_sha,
            )

        final = handle.wait(poll_s=poll_s)
        trainer_rc = handle.returncode
        rec = load_record(run_dir)
        rec.runtime = {**rec.runtime, "source": "runpod_pod"}
        runner.write_record(rec, run_dir)
        ok = final is RunStatus.SUCCEEDED and rec.status is RunStatus.SUCCEEDED and trainer_rc == 0
        stage = None
        if not ok:
            stage = "trainer_exit" if trainer_rc != 0 else "output_verification"
        else:
            try:
                verify_staged_inputs(run_dir, table)
            except ContractError as e:
                ok, stage = False, "input_verification"
                rec.status, rec.failure_reason = RunStatus.FAILED, str(e)
                runner.write_record(rec, run_dir)
        code = EXIT_OK if ok else EXIT_RUN_FAILED
        manifest_sha = write_output_manifest(run_dir, bundle, rec.status.value, code)
        return job.finish(
            ok,
            code,
            trainer_exit_code=trainer_rc,
            failure_stage=stage,
            message=rec.failure_reason,
            run_record_path="run.json",
            output_manifest_path=OUTPUT_MANIFEST_FILE,
            output_manifest_sha256=manifest_sha,
        )
    except Exception as e:
        # Recorded, then re-raised: the container exits non-zero and the status says why.
        job.finish(
            False, EXIT_RUN_FAILED, failure_stage="worker_error", message=f"{type(e).__name__}: {e}"
        )
        raise


__all__ = [
    "EXIT_OK",
    "EXIT_REFUSED",
    "EXIT_RESTARTED",
    "EXIT_RUN_FAILED",
    "GpuUnsupportedError",
    "InputChangedError",
    "RemoteWorkerRunner",
    "run_worker",
    "verify_pod_inputs",
    "verify_staged_inputs",
    "visible_gpus",
    "write_output_manifest",
]
