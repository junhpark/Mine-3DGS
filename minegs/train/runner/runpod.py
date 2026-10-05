"""RunPodRunner (§8.2, Phase 6): the same run, on a RunPod GPU (docs/PHASE6_CONTRACT.md).

Submission does every check it can before anything costs money — image pinned by digest, one
GPU, volume and storage configured, credential present, profile/depth/chunk verified by the
ordinary ``Runner.prepare`` — then uploads the dataset (exactly the bytes its hash covers) and
the bound sidecars to identity-addressed paths on the network volume, publishes ``inputs.json``
last, and only then creates the pod.

The pod runs ``minegs train remote-worker``: LocalRunner's native path, unchanged, recorded as
``runner: runpod``. Success is read from the worker's durable ``status.json`` and output
manifest on the volume and re-checked here — never from the pod's lifecycle, which has no exit
code — and outputs are published locally only after every file matches the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from minegs.core.config import VersionedModel
from minegs.core.errors import ContractError
from minegs.core.provenance import git_commit
from minegs.train.remote.bundle import (
    ChunkInput,
    DepthInput,
    RunInputBundle,
    dataset_file_table,
    dataset_hash_of,
    publish_inputs,
    require_dataset_dir,
)
from minegs.train.remote.layout import (
    CANCEL_FILE,
    OUTPUT_MANIFEST_FILE,
    STATUS_FILE,
    WORKER_LOG,
    RemoteLayout,
    safe_id,
    safe_mount,
)
from minegs.train.remote.provider import (
    ENDED_STATES,
    PodInfo,
    PodSpec,
    ProviderAllocationError,
    ProviderError,
    RunPodClient,
    SdkRunPodClient,
    require_api_key,
)
from minegs.train.remote.secrets import redact
from minegs.train.remote.status import (
    OUTPUT_EXCLUDE,
    JobStatus,
    RemoteOutputRecord,
    verify_output_tree,
)
from minegs.train.remote.store import RemoteStore, store_for
from minegs.train.runner.base import (
    RunConfig,
    RunHandle,
    Runner,
    RunRecord,
    RunStatus,
    load_record,
)

REMOTE_FILE = "remote.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RemoteExecutionRecord(VersionedModel):
    """``<run_dir>/remote.json``: what the submitter saw of a RunPod run (§10).

    Compute provenance and the verification of what came back. Scientific evidence stays in
    ``run.json`` (written by the pod); nothing here is copied into it.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    run_id: str
    dataset_id: str
    provider: str = "runpod"
    credential_source: str
    storage: str
    network_volume_id: str
    volume_mount: str
    image: str
    requested_gpu_types: list[str]
    gpu_count: int = 1
    job_path: str
    run_path: str
    local_dataset_hash: str
    input_bundle_digest: str | None = None
    state: str = "submitting"
    #: True once a pod was created: the computation was handed to the provider.
    remote_provider_execution: bool = False
    pod_id: str | None = None
    chosen_gpu_type: str | None = None
    allocation_attempts: list[dict[str, Any]] = Field(default_factory=list)
    #: The provider's latest description of the pod (lifecycle, GPU display name, hourly rate).
    provider: dict[str, Any] | None = None
    provider_checked_at: str | None = None
    job_status: dict[str, Any] | None = None
    failure_stage: str | None = None
    failure_message: str | None = None
    output_manifest_sha256: str | None = None
    pulled_output_sha256: str | None = None
    artifact_sync_verified: bool = False
    cancel_requested_at: str | None = None
    terminated_at: str | None = None
    notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


def _write_remote(rem: RemoteExecutionRecord, run_dir: Path) -> None:
    data = rem.model_copy(update={"updated_at": _now()}).model_dump(mode="json")
    tmp = run_dir / f".{REMOTE_FILE}.partial"
    tmp.write_text(redact(json.dumps(data, indent=2)))
    os.replace(tmp, run_dir / REMOTE_FILE)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RunPodRunner(Runner):
    name = "runpod"
    #: Builds the provider client. Tests substitute a fake here; nothing else is substituted.
    client_factory: ClassVar[Any] = SdkRunPodClient

    # ---------------------------------------------------------------- checks before cost
    def _check_config(self) -> None:
        cfg = self.config
        if not cfg.image_digest():
            raise ContractError(
                f"runner image {cfg.image!r} is not pinned by digest (repo@sha256:...). A RunPod "
                "run and a local run are the same computation only on the same image digest"
            )
        if not cfg.network_volume_id:
            raise ContractError(
                "runner network_volume_id is not set: the network volume holds the dataset, the "
                "job status and the outputs, and outlives the pod"
            )
        safe_id(cfg.network_volume_id, "network_volume_id")
        safe_mount(cfg.volume_mount)
        if cfg.gpu_count != 1:
            raise ContractError(
                f"gpu_count={cfg.gpu_count}: a run trains on exactly one GPU (gsplat goes "
                "distributed on sight of a second); multi-GPU is not in Phase 6"
            )
        if not cfg.gpu_types:
            raise ContractError("runner gpu_types is empty: name the RunPod GPU type(s) to request")
        if not cfg.cuda_archs:
            raise ContractError("runner cuda_archs is empty: the image's architectures are needed")
        if cfg.container_disk_gb < 1:
            raise ContractError("container_disk_gb must be at least 1")

    def _client(self) -> RunPodClient:
        return self.client_factory()

    def _store(self) -> RemoteStore:
        return store_for((self.config.sync or {}).get("remote"))

    def _profile_name(self, run: RunConfig) -> str:
        p = run.profile
        if "/" in p or p.endswith((".yaml", ".yml")) or Path(p).exists():
            raise ContractError(
                f"profile {p!r} is a file; a RunPod run takes a shipped profile by name (plus "
                "overrides), because the pod's image carries the profiles, not this machine's files"
            )
        return p

    def _bundle(self, run: RunConfig, record: RunRecord, profile, table: dict[str, str]):
        cfg = self.config
        layout = RemoteLayout(cfg.volume_mount, record.dataset_id)
        chunk = depth = None
        plan_file = depth_dir = None
        if run.chunk_plan is not None:
            plan, planned = self._chunk
            plan_file = Path(run.chunk_plan)
            plan_file = plan_file / "chunk_plan.json" if plan_file.is_dir() else plan_file
            chunk = ChunkInput(
                chunk_id=planned.chunk_id,
                plan_id=plan.plan_id,
                plan_digest=plan.plan_digest,
                path=f"{layout.pod(layout.chunk_plan(plan.plan_digest))}/chunk_plan.json",
            )
        verified = getattr(self, "_verified_supervision", None)
        if verified is not None:
            depth_dir = Path(run.depth_supervision)
            depth = DepthInput(
                supervision_id=verified.record.supervision_id,
                artifact_sha256=verified.artifact_sha256,
                path=layout.pod(layout.depth(verified.artifact_sha256)),
            )
        bundle = RunInputBundle(
            run_id=run.run_id,
            dataset_id=record.dataset_id,
            dataset_hash=record.dataset_hash,
            dataset_files=len(table),
            profile=self._profile_name(run),
            overrides=dict(run.overrides),
            backend=run.backend or profile.backend,
            chunk=chunk,
            depth_supervision=depth,
            image=cfg.image,
            requested_gpu_types=list(cfg.gpu_types),
            gpu_count=cfg.gpu_count,
            cuda_archs=list(cfg.cuda_archs),
            network_volume_id=cfg.network_volume_id,
            volume_mount=layout.volume_mount,
            paths={
                "dataset": layout.pod(layout.dataset(record.dataset_hash)),
                "job": layout.pod(layout.job(run.run_id)),
                "run": layout.pod(layout.run(run.run_id)),
            },
            submitter={"git_commit": git_commit()},
            created_at=_now(),
        )
        return layout, bundle, plan_file, depth_dir

    def _pod_spec(self, layout: RemoteLayout, run_id: str, gpu_type: str) -> PodSpec:
        cfg = self.config
        return PodSpec(
            name=f"minegs-{run_id}",
            image=cfg.image,
            gpu_type_id=gpu_type,
            gpu_count=cfg.gpu_count,
            network_volume_id=cfg.network_volume_id,
            volume_mount_path=layout.volume_mount,
            container_disk_gb=cfg.container_disk_gb,
            docker_args=layout.worker_args(run_id),
            env={},
            cloud_type=cfg.cloud_type,
            allowed_cuda_versions=cfg.allowed_cuda_versions,
        )

    def _validated(self, run: RunConfig):
        """All checks that need no external call, in order; returns what submit needs."""
        self._check_config()
        store = self._store()
        credential = require_api_key()
        client = self._client()
        ds = require_dataset_dir(run.dataset_dir)
        run.dataset_dir = str(ds)
        run.runner = self.name
        self._profile_name(run)
        if run.run_id:
            safe_id(run.run_id, "run_id")
        return store, credential, client, ds

    def plan(self, run: RunConfig) -> dict[str, Any]:
        """What ``submit`` would upload and request — with no upload, no pod, no cost."""
        self._check_config()
        store = self._store()
        credential = require_api_key()
        ds = require_dataset_dir(run.dataset_dir)
        run = run.model_copy(update={"dataset_dir": str(ds), "runner": self.name})
        self._profile_name(run)
        with tempfile.TemporaryDirectory() as td:
            run.run_dir = str(Path(td) / "run")
            run, _manifest, profile, record = self.prepare(run)
            table = dataset_file_table(ds)
            layout, bundle, _plan, _depth = self._bundle(run, record, profile, table)
        return {
            "storage": store.describe(),
            "credential_source": credential,
            "dataset_files": len(table),
            "dataset_hash": record.dataset_hash,
            "inputs": bundle.sealed().model_dump(mode="json"),
            "pod_requests": [
                self._pod_spec(layout, run.run_id, g).model_dump(mode="json")
                for g in self.config.gpu_types
            ],
            "note": "dry run: nothing was uploaded and no pod was created",
        }

    # ---------------------------------------------------------------- submit
    def submit(self, run: RunConfig) -> RunHandle:
        store, credential, client, ds = self._validated(run)
        run, _manifest, profile, record = self.prepare(run)
        run_dir = Path(run.run_dir)
        safe_id(run.run_id, "run_id")
        table = dataset_file_table(ds)
        if dataset_hash_of(table) != record.dataset_hash:
            raise ContractError(
                f"{ds} changed while the run was being prepared (dataset hash moved); submit again"
            )
        layout, bundle, plan_file, depth_dir = self._bundle(run, record, profile, table)

        record.status = RunStatus.PENDING
        record.image = self.config.image
        Runner.write_record(record, run_dir)
        rem = RemoteExecutionRecord(
            run_id=run.run_id,
            dataset_id=record.dataset_id,
            credential_source=credential,
            storage=store.describe(),
            network_volume_id=self.config.network_volume_id or "",
            volume_mount=layout.volume_mount,
            image=self.config.image,
            requested_gpu_types=list(self.config.gpu_types),
            gpu_count=self.config.gpu_count,
            job_path=layout.job(run.run_id),
            run_path=layout.run(run.run_id),
            local_dataset_hash=record.dataset_hash,
        )
        _write_remote(rem, run_dir)

        def fail(stage: str, e: Exception) -> None:
            rem.state, rem.failure_stage, rem.failure_message = "failed", stage, redact(e)
            _write_remote(rem, run_dir)
            record.status = RunStatus.FAILED
            record.failure_reason = f"RunPod {stage}: {redact(e)}"
            Runner.write_record(record, run_dir)

        try:
            sealed = publish_inputs(
                store, layout, bundle, ds, table, chunk_plan_file=plan_file, depth_dir=depth_dir
            )
        except ContractError as e:
            fail("input_sync", e)
            raise
        rem.input_bundle_digest = sealed.bundle_digest
        rem.state = "inputs_published"
        _write_remote(rem, run_dir)

        pod = None
        for gpu in self.config.gpu_types:
            try:
                pod = client.create_pod(self._pod_spec(layout, run.run_id, gpu))
            except ProviderAllocationError as e:
                rem.allocation_attempts.append({"gpu_type": gpu, "error": redact(e)})
                continue
            except ProviderError as e:
                rem.allocation_attempts.append({"gpu_type": gpu, "error": redact(e)})
                fail("pod_creation", e)
                raise
            rem.chosen_gpu_type = gpu
            break
        if pod is None:
            err = ProviderAllocationError(
                f"no pod could be allocated for any of {self.config.gpu_types}"
            )
            fail("pod_allocation", err)
            raise err
        rem.pod_id = pod.pod_id
        rem.remote_provider_execution = True
        rem.provider = pod.model_dump(mode="json")
        rem.provider_checked_at = _now()
        rem.state = "submitted"
        _write_remote(rem, run_dir)
        return RunPodHandle(run.run_id, run_dir, self, store, layout, client, rem)

    # ---------------------------------------------------------------- reattach / terminate
    def attach(self, run_dir: str | Path) -> RunPodHandle:
        """A handle for a run submitted earlier (``train fetch``/``train cancel``)."""
        run_dir = Path(run_dir)
        p = run_dir / REMOTE_FILE
        if not p.is_file():
            raise ContractError(f"{run_dir}: no {REMOTE_FILE}; not a RunPod run submitted here")
        rem = RemoteExecutionRecord.load(p)
        if rem.volume_mount != safe_mount(self.config.volume_mount):
            raise ContractError(
                f"{run_dir} ran with volume_mount {rem.volume_mount}, the config says "
                f"{self.config.volume_mount}"
            )
        store = self._store()
        if store.describe() != rem.storage:
            raise ContractError(
                f"{run_dir} was submitted through storage {rem.storage}; the config reaches "
                f"{store.describe()}"
            )
        require_api_key()
        layout = RemoteLayout(rem.volume_mount, rem.dataset_id)
        return RunPodHandle(rem.run_id, run_dir, self, store, layout, self._client(), rem)

    def terminate(self, handle: RunHandle) -> None:
        if not isinstance(handle, RunPodHandle):
            raise ContractError("not a RunPod run handle")
        handle.cancel()


class RunPodHandle(RunHandle):
    """Provider lifecycle + the worker's durable status, combined as §6.3 says."""

    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        runner: RunPodRunner,
        store: RemoteStore,
        layout: RemoteLayout,
        client: RunPodClient,
        rem: RemoteExecutionRecord,
    ) -> None:
        super().__init__(run_id, run_dir)
        self._runner, self._store, self._layout = runner, store, layout
        self._client, self._rem = client, rem
        self._terminal: RunStatus | None = None
        if rem.state in ("succeeded", "failed", "cancelled"):
            self._terminal = RunStatus(rem.state)

    # ---- reading the volume
    def _job(self, name: str) -> str:
        return f"{self._layout.job(self.run_id)}/{name}"

    def job_status(self) -> JobStatus | None:
        data = self._store.read_bytes(self._job(STATUS_FILE))
        if data is None:
            return None
        try:
            return JobStatus.model_validate_json(data)
        except ValueError as e:
            raise ContractError(f"{self._job(STATUS_FILE)} is not a job status ({e})") from e

    def _save(self) -> None:
        _write_remote(self._rem, self.run_dir)

    def _local_record(self) -> RunRecord:
        return load_record(self.run_dir)

    def _settle(self, status: RunStatus, stage: str | None = None, message: str | None = None):
        """Record a terminal state locally. The local run.json is ours until a verified one is
        published; a published (pod-written) run.json is never edited."""
        rem = self._rem
        rem.state = status.value
        if stage:
            rem.failure_stage = stage
        if message:
            rem.failure_message = redact(message)
        self._save()
        if not rem.artifact_sync_verified:
            rec = self._local_record()
            rec.status = status
            if status is not RunStatus.SUCCEEDED:
                rec.failure_reason = redact(
                    f"RunPod {stage or status.value}: {message or ''}".strip(": ")
                )
            Runner.write_record(rec, self.run_dir)
        self._terminal = status
        return status

    # ---- the lifecycle
    def status(self) -> RunStatus:
        if self._terminal is not None:
            return self._terminal
        try:
            js = self.job_status()
        except ContractError as e:
            self._rem.notes.append(redact(f"{_now()} status unreadable: {e}"))
            self._save()
            return RunStatus.UNKNOWN
        if js is not None and js.final:
            return self._finish(js)
        if self._rem.cancel_requested_at:
            return self._settle(RunStatus.CANCELLED, "cancelled", "terminated by the user")
        try:
            pod = self._client.get_pod(self._rem.pod_id or "")
        except ProviderError as e:
            # Not knowing is not failing, and certainly not succeeding.
            self._rem.notes.append(redact(f"{_now()} provider unavailable: {e}"))
            self._save()
            return RunStatus.UNKNOWN
        self._observe(pod)
        if pod is None or (pod.desired_status or "").upper() in ENDED_STATES:
            js = self.job_status()  # the worker may have finished just before the pod did
            if js is not None and js.final:
                return self._finish(js)
            return self._settle(
                RunStatus.FAILED,
                "pod_ended_without_final_status",
                f"the pod is {'gone' if pod is None else pod.desired_status} and the job wrote no "
                "final status; the run did not finish (an ended pod is not a successful run)",
            )
        self._rem.state = "running" if js is not None else "pending"
        self._save()
        return RunStatus.RUNNING if js is not None else RunStatus.PENDING

    def _observe(self, pod: PodInfo | None) -> None:
        self._rem.provider = None if pod is None else pod.model_dump(mode="json")
        self._rem.provider_checked_at = _now()

    def _finish(self, js: JobStatus) -> RunStatus:
        rem = self._rem
        rem.job_status = js.model_dump(mode="json")
        self._save()
        try:
            published = self._pull_and_publish(js) if js.output_manifest_sha256 else None
        except ContractError as e:
            self._terminate_if_done()
            return self._settle(RunStatus.FAILED, "pull_verification", str(e))
        self._terminate_if_done()
        if js.succeeded and published is not None and published.status is RunStatus.SUCCEEDED:
            return self._settle(RunStatus.SUCCEEDED)
        return self._settle(
            RunStatus.FAILED,
            js.failure_stage or "job_failed",
            js.message or "the job did not succeed",
        )

    def _terminate_if_done(self) -> None:
        if not self._runner.config.terminate_on_completion or self._rem.terminated_at:
            return
        try:
            self._client.terminate_pod(self._rem.pod_id or "")
            self._rem.terminated_at = _now()
        except ProviderError as e:
            self._rem.notes.append(redact(f"{_now()} auto-terminate failed: {e}"))
        self._save()

    def _pull_and_publish(self, js: JobStatus) -> RunRecord:
        """Download to a scratch directory, verify every byte, then publish (§9)."""
        rem = self._rem
        run_rel = self._layout.run(self.run_id)
        data = self._store.read_bytes(f"{run_rel}/{OUTPUT_MANIFEST_FILE}")
        if data is None:
            raise ContractError(f"{run_rel}/{OUTPUT_MANIFEST_FILE} is missing")
        if _sha(data) != js.output_manifest_sha256:
            raise ContractError(
                "the output manifest on the volume is not the one the job status recorded"
            )
        manifest = RemoteOutputRecord.model_validate_json(data)
        if manifest.run_id != self.run_id:
            raise ContractError(f"the output manifest is for run {manifest.run_id}")
        tmp = self.run_dir.parent / f".{self.run_dir.name}.pull"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            self._store.download_tree(run_rel, tmp, exclude=OUTPUT_EXCLUDE)
            if (tmp / OUTPUT_MANIFEST_FILE).read_bytes() != data:
                raise ContractError("the pulled output manifest differs from the one verified")
            got = verify_output_tree(tmp, manifest, OUTPUT_MANIFEST_FILE)
            rj = tmp / "run.json"
            if not rj.is_file() or _sha(rj.read_bytes()) != manifest.run_json_sha256:
                raise ContractError("the pulled run.json is not the one the manifest names")
            rec = RunRecord.load(rj)
            want_digest = self._runner.config.image_digest()
            problems = []
            if rec.run_id != self.run_id:
                problems.append(f"run_id {rec.run_id}")
            if rec.dataset_hash != rem.local_dataset_hash:
                problems.append(f"dataset_hash {rec.dataset_hash[:12]}")
            if rec.docker_digest != want_digest or rec.image != rem.image:
                problems.append(f"image {rec.image}")
            if rec.runner != "runpod":
                problems.append(f"runner {rec.runner}")
            if (rec.remote_sync or {}).get("input_bundle_digest") != rem.input_bundle_digest:
                problems.append("input bundle")
            if (rec.status is RunStatus.SUCCEEDED) != js.succeeded:
                problems.append(
                    f"run.json says {rec.status.value} but the job says {js.state.value} "
                    f"(exit {js.exit_code}, trainer exit {js.trainer_exit_code})"
                )
            if problems:
                raise ContractError(
                    f"the pulled run is not the submitted run: {'; '.join(problems)}"
                )
            mine = {"run.json", REMOTE_FILE}
            foreign = sorted(p.name for p in self.run_dir.iterdir() if p.name not in mine)
            if foreign:
                raise ContractError(
                    f"{self.run_dir} holds {foreign[:5]} that this submission did not write; "
                    "refusing to publish pulled outputs over them"
                )
            for f in sorted(p for p in tmp.rglob("*") if p.is_file()):
                rel = f.relative_to(tmp)
                if rel.as_posix() == "run.json":
                    continue
                dest = self.run_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(f, dest)
            os.replace(rj, self.run_dir / "run.json")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        rem.output_manifest_sha256 = js.output_manifest_sha256
        rem.pulled_output_sha256 = got
        rem.artifact_sync_verified = True
        self._save()
        return rec

    # ---- the rest of the handle contract
    def logs(self, tail: int | None = None) -> str:
        local = self.run_dir / "log" / "train.log"
        if local.is_file():
            text = local.read_text(errors="replace")
        else:
            raw = self._store.read_bytes(f"{self._layout.run(self.run_id)}/log/train.log")
            if raw is None:
                raw = self._store.read_bytes(self._job(WORKER_LOG)) or b""
            text = raw.decode(errors="replace")
        lines = text.splitlines()
        return "\n".join(lines[-tail:] if tail else lines)

    def fetch_artifacts(self) -> list[Path]:
        if self.status() is not RunStatus.SUCCEEDED:
            return []
        return sorted((self.run_dir / "point_cloud").glob("*.ply"))

    def cancel(self) -> None:
        rem = self._rem
        if self._terminal is not None:
            return
        rem.cancel_requested_at = _now()
        self._save()
        self._store.write_atomic(
            self._job(CANCEL_FILE),
            json.dumps({"run_id": self.run_id, "requested_at": rem.cancel_requested_at}).encode(),
        )
        try:
            self._client.terminate_pod(rem.pod_id or "")
            rem.terminated_at = _now()
        except ProviderError as e:
            rem.notes.append(redact(f"{_now()} terminate failed: {e}"))
        self._save()


def _cli_note() -> str:
    return (
        "RunPod runs need a runner config (configs/runner/runpod.yaml) with an image pinned by "
        "digest, a network volume and its storage remote, and RUNPOD_API_KEY in the environment"
    )


__all__ = [
    "REMOTE_FILE",
    "RemoteExecutionRecord",
    "RunPodHandle",
    "RunPodRunner",
    "_cli_note",
]
