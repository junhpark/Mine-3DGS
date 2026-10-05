"""Runner contract (§8.2)::

    Runner.submit(run_config) -> RunHandle
    RunHandle.status() / .logs() / .fetch_artifacts()

Both runners execute the *same* GPU image digest; ``run.json`` records it (§9).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import Field

from minegs.core.config import MigrationRegistry, VersionedModel
from minegs.core.errors import ContractError
from minegs.core.manifest import Manifest
from minegs.core.provenance import ProvenanceRecord, make_id, sha256_tree, stamp
from minegs.train.backends import get_backend
from minegs.train.profiles import Profile, load_profile

# Everything the trainer can see (§9): manifest, sparse model, init points, images, masks,
# centerline. A changed image changes the hash. raw/ is not part of the dataset.
DATASET_HASH_PATTERNS = (
    "manifest.json",
    "sparse/0/*.txt",
    "init_points.ply",
    "images/**/*",
    "masks/**/*",
    "centerline.csv",
    # Phase 3 keeps the frame set, SfM and registration records inside the dataset so they are
    # inside this hash. Evidence a manifest merely points at can be swapped afterwards with
    # every recorded digest still matching; evidence in the hashed tree cannot.
    "provenance/**/*",
)


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


RUNNER_CONFIG_MIGRATIONS = MigrationRegistry("runner_config")


@RUNNER_CONFIG_MIGRATIONS.register("1.0", "1.1")
def _runner_1_0_to_1_1(d: dict[str, Any]) -> dict[str, Any]:
    """1.1 adds the Phase 6 RunPod fields; each has a default that a 1.0 config already meant
    (one GPU, the image's CUDA architectures, terminate a pod once its evidence is durable)."""
    return d


#: The CUDA architectures ``docker/Dockerfile.gpu`` compiles for (``TORCH_CUDA_ARCH_LIST``); a
#: test keeps the two equal. A GPU outside this list cannot run the image's kernels.
IMAGE_CUDA_ARCHS = ("7.5", "8.0", "8.6", "8.9", "9.0")


class RunnerConfig(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "1.1"
    MIGRATIONS: ClassVar[MigrationRegistry | None] = RUNNER_CONFIG_MIGRATIONS
    runner: str = "local"
    image: str = ""
    data_root: str = "./data"
    #: Which GPU the baseline runs on, as a docker ``--gpus`` value. One device, by index:
    #: gsplat v1.5.3 sets ``world_size = torch.cuda.device_count()`` and spawns one process per
    #: visible device with no flag to decline (gsplat/distributed.py cli()), so "all" silently
    #: turns a single-GPU baseline into distributed training — whose PLY write is not
    #: rank-qualified, leaving every rank racing on the same point_cloud_<step>.ply.
    gpus: str = "device=0"
    shm_size: str = "16g"
    native: bool = False
    gpu_types: list[str] = Field(default_factory=list)
    network_volume_id: str | None = None
    volume_mount: str = "/data"
    container_disk_gb: int = 40
    sync: dict[str, Any] = Field(default_factory=dict)
    poll_interval_s: int = 30
    # ---- Phase 6 (schema 1.1)
    #: GPUs per pod. Only 1: a second visible device makes gsplat distributed (see ``gpus``).
    gpu_count: int = 1
    #: Compute capabilities the image supports; the pod refuses any other GPU before training.
    cuda_archs: list[str] = Field(default_factory=lambda: list(IMAGE_CUDA_ARCHS))
    #: Host CUDA versions RunPod may place the pod on (provider filter); None = no filter.
    allowed_cuda_versions: list[str] | None = None
    cloud_type: str = "ALL"
    #: Terminate the pod once its final status and output manifest are on the volume.
    terminate_on_completion: bool = True

    def image_digest(self) -> str | None:
        return self.image.split("@", 1)[1] if "@" in self.image else None


class RunConfig(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "1.0"
    run_id: str = ""
    dataset_dir: str
    run_dir: str = ""
    profile: str = "light"
    backend: str = "gsplat"
    runner: str = "local"
    # The run directory to continue from. Explicit because a boolean could not say *which* run
    # it meant — run ids are minted per submission — and an ambiguous target is one that can
    # quietly become a fresh run. No shipped backend can resume training, so setting this always
    # fails closed today; see ``Runner.prepare`` and docs/ROADMAP.md §Phase 0D.
    resume_from: str | None = None
    chunk_id: str | None = None
    #: The verified ``ChunkPlanRecord`` (``chunks/<plan_id>``) that ``chunk_id`` names (Phase 5).
    #: A chunk is only ever trained from a plan; neither is accepted without the other.
    chunk_plan: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)
    #: A verified DepthSupervisionRecord directory (Phase 4). Required exactly when the profile
    #: requests depth_loss; given to a profile that does not, it is refused, not ignored.
    depth_supervision: str | None = None


RUN_RECORD_MIGRATIONS = MigrationRegistry("run_record")


@RUN_RECORD_MIGRATIONS.register("1.0", "1.1")
def _run_1_0_to_1_1(d: dict[str, Any]) -> dict[str, Any]:
    """1.1 adds Phase 0D.2 execution evidence; 1.0 records simply have none.

    Nothing is translated because nothing moved: every 1.1 field is new, and a 1.0 run genuinely
    did not record whether its artifacts were ever checked. Leaving them unset says that, which
    is the honest reading — inventing a ``succeeded`` run's missing evidence would not be.
    """
    return d


@RUN_RECORD_MIGRATIONS.register("1.1", "1.2")
def _run_1_1_to_1_2(d: dict[str, Any]) -> dict[str, Any]:
    """1.2 adds Phase 4 trainer evidence; a 1.1 run recorded none of it, and says so by its
    absence rather than by values invented now."""
    return d


@RUN_RECORD_MIGRATIONS.register("1.3", "1.4")
def _run_1_3_to_1_4(d: dict[str, Any]) -> dict[str, Any]:
    """1.4 adds Phase 6 remote evidence. A 1.3 run did not run remotely, or recorded nothing
    about it: both fields stay null rather than being reconstructed."""
    return d


@RUN_RECORD_MIGRATIONS.register("1.2", "1.3")
def _run_1_2_to_1_3(d: dict[str, Any]) -> dict[str, Any]:
    """1.3 adds the Phase 5 chunk binding. A 1.2 run trained no plan's chunk: ``chunk`` stays
    null. (A 1.2 ``chunk_id`` came from the legacy manifest plan and is kept as written.)"""
    return d


class RunRecord(VersionedModel):
    """``runs/<run_id>/run.json`` (§9)."""

    SCHEMA_VERSION: ClassVar[str] = "1.4"
    MIGRATIONS: ClassVar[MigrationRegistry | None] = RUN_RECORD_MIGRATIONS
    run_id: str
    dataset_id: str
    dataset_hash: str
    chunk_id: str | None = None
    backend: dict[str, str]
    profile: dict[str, Any]
    runner: str
    docker_digest: str | None = None
    command: list[str] = Field(default_factory=list)
    status: RunStatus = RunStatus.PENDING
    T_local_from_internal: list[list[float]] | None = None  # Sim3 4x4 (scale-bearing)
    staged: dict[str, Any] = Field(default_factory=dict)  # images used, subset flag, init source
    T_tls_from_local: list[list[float]] | None = None
    frame_of_outputs: str = "LOCAL_METRIC"
    outputs: list[str] = Field(default_factory=list)
    provenance: ProvenanceRecord

    # ---- Phase 0D.2 execution evidence (schema 1.1). Every field defaults, so a 1.0 record
    # migrates forward by doing nothing; see _run_1_0_to_1_1.
    image: str | None = None  # the full ref; docker_digest is the @sha256 half of it
    started_at: str | None = None
    completed_at: str | None = None
    duration_s: float | None = None
    command_env: dict[str, str] = Field(default_factory=dict)
    #: What the runtime said about itself before the trainer started (§0D.2 D2-2).
    runtime: dict[str, Any] = Field(default_factory=dict)
    max_steps: int | None = None
    #: Read back from the trainer's own artifacts, not assumed from the profile (D2-4).
    observed_final_step: int | None = None
    checkpoints: list[str] = Field(default_factory=list)
    final_checkpoint: str | None = None
    checkpoint_step: int | None = None
    final_model: str | None = None
    gaussian_count: int | None = None
    peak_gpu_memory_gb: float | None = None
    train_seconds: float | None = None
    #: Renders the trainer left, for the qualitative look-at-it check (D2-10). A path, not a
    #: verdict: nothing here judges whether the result resembles a tunnel.
    renders: list[str] = Field(default_factory=list)
    #: Camera / init / output spans in metres, the evidence behind the frame check (D2-7).
    extents: dict[str, Any] = Field(default_factory=dict)
    #: Why a run is FAILED. A failed run that cannot say why is not much better than a silent one.
    failure_reason: str | None = None

    # ---- Phase 4 trainer evidence (schema 1.2). All default, so 1.1 records migrate unchanged.
    #: Which program trained: upstream directly, or the MineGS adapter around it (AD-5).
    trainer: dict[str, Any] = Field(default_factory=dict)
    #: Capability requests as the profile made them, and as the backend resolved them.
    capabilities: dict[str, Any] = Field(default_factory=dict)
    #: What the trainer's cfg.yml must say, fixed before the run, compared after it (AD-8).
    expected_trainer_config: dict[str, Any] = Field(default_factory=dict)
    #: What it did say: the compared keys, read back, and the file's digest.
    trainer_config: dict[str, Any] = Field(default_factory=dict)
    trainer_config_sha256: str | None = None
    #: The verified depth supervision this run trained with (identity + semantics), or None.
    depth_supervision: dict[str, Any] | None = None
    #: The MCMC unit rescaling: host-side expectation and what the trainer applied (AD-6).
    metric_compensation: dict[str, Any] | None = None
    #: Images upstream actually optimised, which is not every staged one (test_every).
    optimised_images: int | None = None

    # ---- Phase 5 chunk binding (schema 1.3). Null for a run that is not a plan's chunk.
    #: Plan identity, the chunk's core/support, what was selected for it, and what it used
    #: (docs/PHASE5_CONTRACT.md §6). Outputs stay in the dataset's LOCAL_METRIC frame.
    chunk: dict[str, Any] | None = None

    # ---- Phase 6 remote evidence (schema 1.4). Null for a run that did not execute remotely.
    #: Compute provenance, as the pod knew it: provider, pod id, requested GPU types, volume and
    #: remote paths (docs/PHASE6_CONTRACT.md §10). Not scientific evidence.
    remote_execution: dict[str, Any] | None = None
    #: What reached the pod, by identity: the input bundle digest, the dataset hash the submitter
    #: claimed and the one the pod re-derived, the sidecar digests.
    remote_sync: dict[str, Any] | None = None


class RunHandle(ABC):
    def __init__(self, run_id: str, run_dir: Path) -> None:
        self.run_id = run_id
        self.run_dir = run_dir

    @abstractmethod
    def status(self) -> RunStatus: ...

    @abstractmethod
    def logs(self, tail: int | None = None) -> str: ...

    @abstractmethod
    def fetch_artifacts(self) -> list[Path]:
        """Bring ``point_cloud/*.ply`` (LOCAL_METRIC), logs, run.json into ``run_dir``."""

    def wait(self, poll_s: float = 10.0, timeout_s: float | None = None) -> RunStatus:
        import time

        t0 = time.time()
        while True:
            st = self.status()
            if st in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED):
                return st
            if timeout_s is not None and time.time() - t0 > timeout_s:
                return st
            time.sleep(poll_s)


class Runner(ABC):
    name: str = "abstract"

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config

    @abstractmethod
    def submit(self, run: RunConfig) -> RunHandle: ...

    # ---------------------------------------------------------------- shared prep
    def prepare(self, run: RunConfig) -> tuple[RunConfig, Manifest, Profile, RunRecord]:
        dataset_dir = Path(run.dataset_dir)
        manifest = Manifest.load_dataset(dataset_dir)
        profile = load_profile(run.profile)
        for k, v in run.overrides.items():
            setattr(profile, k, v)
        backend = get_backend(run.backend or profile.backend)
        backend.resolve_requests(profile)  # raises ContractError naming the unmet capability
        if not run.run_id:
            run.run_id = make_id(backend.name)
        if not run.run_dir:
            run.run_dir = str(dataset_dir.parent / "runs" / run.run_id)
        run_dir = Path(run.run_dir)
        # One frame for every run, chunked or not (Phase 5 AD-4): staging never re-expresses
        # poses or points, so the outputs are in the dataset's LOCAL_METRIC and that is what is
        # recorded. (The legacy manifest plan's per-chunk origin described data that did not
        # exist; it is not used.)
        T_tls_from_local = manifest.T_tls_from_local
        chunk_binding = None
        self._chunk = None
        if run.chunk_id is not None or run.chunk_plan is not None:
            if not (run.chunk_id and run.chunk_plan):
                raise ContractError(
                    "a chunk is trained from a verified chunk plan: give --chunk-plan "
                    "<chunks/<plan_id>> together with --chunk <id> (Phase 5 AD-1); the legacy "
                    "manifest.chunks windows are not a training plan"
                )
            from minegs.chunks.plan import verify_chunk_plan

            plan = verify_chunk_plan(dataset_dir, run.chunk_plan, manifest=manifest)
            chunk = plan.chunk(run.chunk_id)
            require_whole_chunk(chunk, profile.max_images)
            self._chunk = (plan, chunk)
            chunk_binding = {
                "plan_id": plan.plan_id,
                "plan_digest": plan.plan_digest,
                "plan_path": str(Path(run.chunk_plan).resolve()),
                "chunk_id": chunk.chunk_id,
                "ordinal": chunk.ordinal,
                "n_chunks": len(plan.chunks),
                "core_range_m": list(chunk.core_range_m),
                "support_range_m": list(chunk.support_range_m),
                "capture_groups": list(chunk.capture_groups),
                "images": list(chunk.images),
                "actual_image_support_m": list(chunk.actual_image_support_m),
            }
        # `is not None`, not truthiness: resume_from="" is still a resume *request*, and one
        # that names nothing is the least honourable of all — under a truthiness test it would
        # fall through to a fresh iteration-0 run, which is the exact silent restart this phase
        # exists to forbid. No CLI value can produce it today (typer renders --resume-from ""
        # as Path(".")), but the guard should not depend on that.
        if run.resume_from is not None:
            refuse_resume(backend)
        enabled = backend.resolve_requests(profile)
        verified = None
        if enabled.get("depth_loss") and run.depth_supervision is None:
            raise ContractError(
                f"profile {profile.name} requests depth supervision (requests.depth_loss: "
                "true) and no DepthSupervisionRecord was given. Build one with `minegs dataset "
                "depth-supervision`, then pass --depth-supervision <dir>."
            )
        if run.depth_supervision is not None:
            if not enabled.get("depth_loss"):
                raise ContractError(
                    f"--depth-supervision given, but profile {profile.name} does not request "
                    "depth_loss; supervision a run would ignore is refused, not dropped"
                )
            from minegs.train.supervision.depth import verify_depth_supervision

            # Before anything is created: an artifact that fails its contract stops the run
            # while there is still nothing to clean up.
            verified = verify_depth_supervision(
                dataset_dir, run.depth_supervision, manifest=manifest
            )
            if self._chunk is not None:
                # The global artifact is reused byte for byte; the adapter will only read the
                # samples of the chunk's own images. If there are none, the run could only end
                # FAILED after training, so it is refused now (Phase 5 §5.5).
                chunk = self._chunk[1]
                if not verified.images_with_samples(list(chunk.images)):
                    raise ContractError(
                        f"chunk {chunk.chunk_id}: none of its {len(chunk.images)} training "
                        f"images carries a depth sample in {verified.record.supervision_id}; a "
                        "depth-supervised run of it would apply no depth term"
                    )
        refuse_used_run_dir(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        record = RunRecord(
            schema_version=RunRecord.SCHEMA_VERSION,
            run_id=run.run_id,
            dataset_id=manifest.dataset_id,
            dataset_hash=sha256_tree(dataset_dir, DATASET_HASH_PATTERNS),
            chunk_id=run.chunk_id,
            chunk=chunk_binding,
            backend={"name": backend.name, "version": backend.version()},
            profile=profile.model_dump(mode="json"),
            runner=self.name,
            docker_digest=self.config.image_digest(),
            T_tls_from_local=T_tls_from_local.to_list(),
            provenance=stamp(run.model_dump(mode="json"), parents=[manifest.dataset_id]),
            capabilities={"requested": dict(profile.requests), "resolved": enabled},
            depth_supervision=None
            if verified is None
            else {
                **verified.summary(),
                "path": str(Path(run.depth_supervision).resolve()),
            },
        )
        self._verified_supervision = verified
        return run, manifest, profile, record

    @staticmethod
    def write_record(record: RunRecord, run_dir: Path) -> Path:
        """Write ``run.json`` atomically.

        The status field is what tells a later reader a run died mid-flight, so the file must
        never be observed half-written. Atomicity lives here rather than in ``VersionedModel.save``,
        which is shared with manifests and ingest configs that do not have this problem.
        """
        final = run_dir / "run.json"
        tmp = run_dir / ".run.json.partial"
        record.save(tmp)
        os.replace(tmp, final)
        return final


def refuse_used_run_dir(run_dir: Path) -> None:
    """A fresh run needs a directory nothing has run in before (§0D.2 B1).

    Evidence is read back out of ``backend_out`` after the trainer exits, and nothing clears it
    between runs. So a second run into an occupied directory inherits the first one's checkpoint,
    stats and PLY — and a trainer that writes nothing then passes every postcondition on someone
    else's artifacts, which is the "exit 0 and produced nothing" case wearing a successful run's
    clothes. Refusing is the fail-closed answer: clearing would destroy the earlier run's
    evidence, and merging would be worse than either.
    """
    if not run_dir.exists():
        return
    existing = sorted(p.name for p in run_dir.iterdir())
    if not existing:
        return
    raise ContractError(
        f"{run_dir} already holds {existing[:6]}. A run directory is written once: artifacts "
        "left by an earlier run would be read back as this one's evidence. Submit without "
        "run_dir to mint a fresh run id, or point at a directory that does not exist yet."
    )


def refuse_resume(backend: Any) -> None:
    """``--resume-from`` always fails closed today, before the run directory exists.

    A restart from iteration 0 is not a slow resume, it is a different experiment recorded
    under a run id that claims to continue another one — so the only safe answer to a resume
    request no backend can honour is a refusal, never a fresh run.

    Two refusals, in order. The first is the one users hit: no shipped backend continues
    training, and gsplat v1.5.3 in particular turns ``--ckpt`` into an evaluation pass (see
    ``minegs.train.backends.gsplat``), so the adapter declares ``resume=False`` and this raises
    with that reason. The second covers a backend that *does* declare the capability: resuming
    needs a checkpoint contract that restores the whole training state — optimizer, schedulers,
    strategy state, step, RNG — and Mine-3DGS does not own one yet. Deliberately not a partial
    implementation: a resume that silently drops optimizer state is a different experiment too.
    """
    from minegs.core.errors import NotYetImplementedError

    if not backend.capabilities().has("resume"):
        note = backend.capability_notes.get("resume", "")
        raise ContractError(
            f"backend {backend.name} does not support resuming training, so --resume-from "
            f"cannot be honoured. {note}".strip()
        )
    raise NotYetImplementedError(
        f"resuming a run (--resume-from) with backend {backend.name}: checkpoint discovery, "
        "host/container path translation and parent-run compatibility are not implemented",
        "0D.3",
    )


#: How far the output may differ in span from the points it was initialised with before the run
#: is refused (§0D.2 D2-7). Densification legitimately spreads gaussians past the input cloud, so
#: this is deliberately loose; what it exists to catch is a *scale* change. gsplat's
#: ``normalize_world_space`` rescales a scene to roughly unit size, which for a 60-100 m drift is
#: a 30-100x shrink — an order of magnitude clear of anything densification does. A blow-up in the
#: other direction is equally suspect, so the test is two-sided. This bounds the failure it is
#: named for; it is not evidence that the output geometry is correct.
MAX_EXTENT_RATIO = 20.0


#: Only the explicit ``device=<n>`` form, optionally quoted as docker's own docs write it. A
#: bare integer is deliberately refused: to docker ``--gpus 2`` means *two* GPUs, not GPU 2, and
#: a spec whose meaning flips between "index" and "count" is not one to guess at.
_ONE_DEVICE = re.compile(r"^\"?device=(\d+)\"?$")


def single_device_index(gpus: str) -> str:
    """The one GPU index this baseline may use, or a refusal (§0D.2 B2).

    Multi-GPU is not a faster version of this baseline. gsplat decides to go distributed purely
    from how many devices it can see, and in that mode every rank writes the same
    ``ply/point_cloud_<step>.ply`` — so what lands on disk is whichever rank finished last, or a
    torn file. Distributed training is out of Phase 0D.2's scope, so the run is pinned to one
    device rather than left to discover this at the end of a long job.
    """
    m = _ONE_DEVICE.match(str(gpus).strip())
    if m is None:
        raise ContractError(
            f"runner.gpus={gpus!r} exposes more than one GPU. gsplat v1.5.3 reads "
            "torch.cuda.device_count() and spawns one training process per visible device, and "
            "its PLY export is not rank-qualified, so the ranks would overwrite each other's "
            "point_cloud_<step>.ply. Phase 0D.2 is a single-GPU baseline: write "
            "gpus: device=<n> (a bare number is docker's *count*, not an index). Multi-GPU is "
            "not in scope — docs/ROADMAP.md §Phase 0D."
        )
    return m.group(1)


def runtime_info() -> dict[str, Any]:
    """What the machine says about itself, best effort, before a trainer starts (§0D.2 D2-2).

    Deliberately not fatal: the CUDA *gate* is ``cuda_available()``, which already refuses a run
    with no GPU. This only describes what was there. ``nvidia-smi`` is queried rather than torch
    because on the docker path torch lives inside the image and the host may not have it at all;
    where torch is importable (the ``--native`` path) its view is recorded alongside.
    """
    info: dict[str, Any] = {}
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if out.returncode == 0 and out.stdout.strip():
                gpus = [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]
                info["gpus"] = gpus
                first = gpus[0].split(",")
                if len(first) >= 2:
                    info["gpu_model"] = first[0].strip()
                    info["driver_version"] = first[1].strip()
        except (OSError, subprocess.SubprocessError) as e:
            info["nvidia_smi_error"] = str(e)
    info["source"] = "host"
    try:  # present on the native path; absent on a host that only runs the container
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        info["torch"] = None
    try:  # the library the native trainer imports; None where it is not installed
        import gsplat

        info["gsplat"] = getattr(gsplat, "__version__", None)
    except ImportError:
        info["gsplat"] = None
    return info


def pinned_upstream(record: Any) -> bool:
    """Whether the run's own runtime evidence names the pinned gsplat and its pinned trainer.

    Both are recorded by the runner from the machine that trained (the host on ``--native``, the
    image on the docker route), never from the request.
    """
    from minegs.train.backends.gsplat import PINNED_GSPLAT, UPSTREAM_TRAINER_SHA256

    rt = dict(getattr(record, "runtime", None) or {})
    return (
        str(rt.get("gsplat")) == PINNED_GSPLAT
        and rt.get("trainer_sha256") == UPSTREAM_TRAINER_SHA256
    )


def real_gpu_evidence(record: Any) -> bool:
    """A GPU was there, and the pinned upstream trained on it, as the run itself recorded.

    The one rule for "real GPU training" (e2e report and run comparison alike). A trainer that
    is not v1.5.3's ``simple_trainer.py``, or an unrecorded gsplat, is not real gsplat training,
    whatever the host has.
    """
    rt = dict(getattr(record, "runtime", None) or {})
    gpu = bool(rt.get("gpu_model")) or rt.get("torch_cuda_available") is True
    return gpu and pinned_upstream(record)


#: Asked of the image itself, so the recorded versions are the ones that trained.
_PROBE = (
    "import json,sys,torch;"
    "d={'python':sys.version.split()[0],'torch':torch.__version__,"
    "'torch_cuda':torch.version.cuda,'torch_cuda_available':torch.cuda.is_available(),"
    "'device_count':torch.cuda.device_count()};"
    "d['gpu_model']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None;"
    "\ntry:\n import gsplat; d['gsplat']=gsplat.__version__\nexcept Exception as e:"
    " d['gsplat']=None\n"
    "import hashlib\ntry:\n d['trainer_sha256']=hashlib.sha256(open(sys.argv[1],'rb').read())"
    ".hexdigest()\nexcept (IndexError, OSError):\n d['trainer_sha256']=None\n"
    "print('MINEGS_PROBE'+json.dumps(d))"
)


def container_runtime_info(image: str, device: str, trainer_path: str) -> dict[str, Any]:
    """What the *container* reports about itself, before the trainer starts (§0D.2 D2-2, S1).

    The host's torch and the image's torch are different installations, and it is the image's
    that trains — on a machine with no torch at all the host view is empty while the run is
    perfectly fine. So the image is asked directly, with the same single device the run will
    use, and the answer is a gate as well as a description: a container whose torch cannot see
    a GPU cannot produce a GPU baseline, however healthy ``nvidia-smi`` looks outside it.
    """
    argv = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        f"device={device}",
        "-e",
        "CUDA_VISIBLE_DEVICES=0",
        "--entrypoint",
        "python",
        image,
        "-c",
        _PROBE,
        trainer_path,
    ]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=300, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        raise ContractError(f"could not probe the GPU image {image}: {e}") from e
    marker = next((ln for ln in out.stdout.splitlines() if ln.startswith("MINEGS_PROBE")), None)
    if out.returncode != 0 or marker is None:
        raise ContractError(
            f"the GPU image {image} could not report its runtime (exit {out.returncode}). "
            f"stderr: {(out.stderr or '').strip()[:400]}"
        )
    info: dict[str, Any] = json.loads(marker[len("MINEGS_PROBE") :])
    info["source"] = "container"
    info["trainer_path"] = trainer_path
    if not info.get("torch_cuda_available"):
        raise ContractError(
            f"torch inside {image} reports no CUDA device. A baseline cannot be trained on the "
            "CPU, and this is the runtime that would have trained it (§0D.2 D2-2)."
        )
    if int(info.get("device_count") or 0) != 1:
        raise ContractError(
            f"the container sees {info.get('device_count')} GPUs; gsplat would go distributed "
            "and its ranks would overwrite one another's PLY (§0D.2 B2)."
        )
    return info


def _span(xyz: Any) -> float:
    """Largest side of the axis-aligned box, in metres. 0.0 for a degenerate set."""
    import numpy as np

    a = np.asarray(xyz, dtype=float)
    if a.ndim != 2 or len(a) == 0:
        return 0.0
    return float(np.max(a.max(axis=0) - a.min(axis=0)))


def verify_postconditions(
    record: RunRecord,
    evidence: Any,
    run_dir: Path,
    staged_dir: Path,
    outputs: list[Path],
) -> None:
    """Everything that must be true before a run may be called SUCCEEDED (§0D.2 D2-4..D2-7).

    Raises ``ContractError`` naming the first thing that is not. Exit code 0 is the weakest of
    the signals here: a trainer that writes nothing exits 0 exactly like one that trains.
    """
    import numpy as np

    from minegs.core.pointcloud import read_ply

    if evidence.final_checkpoint is None:
        raise ContractError(
            "the trainer exited 0 but wrote no checkpoint. A run with no checkpoint is not a "
            "baseline; it is a process that ended (§0D.2 D2-5)"
        )
    if evidence.final_checkpoint.stat().st_size == 0:
        raise ContractError(f"{evidence.final_checkpoint} is empty (§0D.2 D2-5)")

    if evidence.final_model is None:
        raise ContractError(
            "the trainer exited 0 but produced no model PLY (enable save_ply) (§0D.2 D2-6)"
        )
    if not outputs:
        raise ContractError("no PLY reached the run's point_cloud/ (§0D.2 D2-6)")

    # Step progression. Upstream writes its last checkpoint at ``max_steps - 1``, so a run that
    # reached the end reports one less than the profile asked for; anything short of that ran
    # fewer iterations than requested, whatever its exit code said.
    want = evidence.configured_max_steps
    got = evidence.observed_final_step
    if want is not None:
        if got is None:
            raise ContractError(
                f"the trainer left no step evidence, so reaching {want} steps cannot be shown "
                "(§0D.2 D2-4)"
            )
        if got < want - 1:
            raise ContractError(
                f"training stopped at step {got} of a configured {want} ({want - 1} expected as "
                "the final step). A short run is a different experiment (§0D.2 D2-4)"
            )

    final = read_ply(_in_run(outputs, evidence))
    if len(final.xyz) == 0:
        raise ContractError(f"{evidence.final_model}: the model holds no gaussians (§0D.2 D2-6)")
    if not np.all(np.isfinite(final.xyz)):
        n = int((~np.isfinite(final.xyz)).any(axis=1).sum())
        raise ContractError(
            f"{n} of {len(final.xyz)} gaussian centres are not finite; training diverged "
            "(§0D.2 D2-6)"
        )

    # Frame invariant (§0D.2 D2-7). The evidence is gathered in full and recorded *before*
    # either gate fires, so a refused run still explains itself in run.json.
    ext: dict[str, Any] = {"output_span_m": _span(final.xyz), "unit": "m"}
    sparse = staged_dir / "sparse" / "0"
    if (sparse / "points3D.txt").exists():
        from minegs.ingest.common import colmap_io

        pts = colmap_io.read_model(sparse)
        init_xyz = np.array([p.xyz for p in pts.points3D.values()])
        cams = np.array([im.center for im in pts.images.values()])
        ext["init_span_m"] = _span(init_xyz)
        ext["camera_span_m"] = _span(cams)
    init_span = ext.get("init_span_m") or 0.0
    out_span = ext["output_span_m"]
    ratio = out_span / init_span if init_span > 0 and out_span > 0 else None
    if ratio is not None:
        ext["output_over_init"] = ratio
    declared = (getattr(evidence, "trainer_config", None) or {}).get("normalize_world_space")
    if declared is not None:
        ext["trainer_normalize_world_space"] = declared
    # Assigned once, fully built: the model validates on assignment, so it stores a *copy* and
    # anything written into ``ext`` afterwards would never reach the record — including the
    # figures that explain a refusal.
    record.extents = ext
    record.gaussian_count = len(final.xyz)

    # The trainer's own record of how it was configured leads. Upstream gsplat defaults
    # ``normalize_world_space`` to True, so this is the difference between a baseline in metres
    # and one in arbitrary units, and cfg.yml says which happened instead of leaving it inferred.
    if declared is not None and declared.strip().lower() not in ("false", "0", "no"):
        raise ContractError(
            f"the trainer ran with normalize_world_space={declared!r}. BACKEND_INTERNAL must be "
            "LOCAL_METRIC for this baseline, so the output is in arbitrary units and the run is "
            "not a metric one (§0D.2 D2-7). Why it stays refused: docs/PHASE4_CONTRACT.md §8"
        )

    # The span ratio corroborates; it does not lead. It is blind to rotation and translation by
    # construction, and its margin against a real normalisation is thin — so it is the backstop
    # for a backend that records no configuration, not the primary test.
    if ratio is not None and not 1.0 / MAX_EXTENT_RATIO <= ratio <= MAX_EXTENT_RATIO:
        raise ContractError(
            f"output span {out_span:.3f} m against an initialisation span of "
            f"{init_span:.3f} m is a factor of {ratio:.3g}, beyond the {MAX_EXTENT_RATIO}x "
            "this baseline allows. BACKEND_INTERNAL is supposed to be LOCAL_METRIC, so a "
            "scale change of this size means something normalised the scene (§0D.2 D2-7)"
        )

    check_trainer_evidence(record, evidence, staged_dir, n_final=len(final.xyz))


def _cfg_value(cfg: dict, key: str) -> Any:
    """A compared key's value in the trainer's cfg.yml; ``strategy`` is the class name."""
    from minegs.train.backends.gsplat import strategy_name

    if key == "strategy":
        return strategy_name(cfg)
    if key.startswith("strategy."):
        st = cfg.get("strategy")
        return st.get(key.split(".", 1)[1], _MISSING) if isinstance(st, dict) else _MISSING
    return cfg.get(key, _MISSING)


_MISSING = object()


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a is b or (a == b and type(a) is type(b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def _close(a: Any, b: float, rel: float = 1e-6) -> bool:
    return isinstance(a, (int, float)) and not isinstance(a, bool) and abs(a - b) <= rel * abs(b)


def staged_metric_scale(staged_dir: Path) -> float:
    """``s`` that upstream normalisation would apply to the staged cameras (host-side port)."""
    from minegs.ingest.common import colmap_io
    from minegs.train.backends.gsplat import similarity_from_cameras
    from minegs.train.trainers.advanced_gs import metric_scale_from_cameras

    model = colmap_io.read_model(Path(staged_dir) / "sparse" / "0")
    names = sorted(model.image_by_name())
    by_name = model.image_by_name()
    c2w = np.stack([by_name[n].world_from_cam.matrix() for n in names])
    return metric_scale_from_cameras(c2w, similarity_from_cameras)


def require_whole_chunk(chunk, max_images: int | None) -> None:
    """A chunk trains on every image of its capture groups, or not at all (Phase 5 §5.3).

    The plan selects groups atomically; a profile's ``max_images`` thins image by image, which
    would leave part of a 360 ring or of a video segment. Refused rather than thinned.
    """
    if max_images is not None and max_images < len(chunk.images):
        raise ContractError(
            f"chunk {chunk.chunk_id} plans {len(chunk.images)} training images in "
            f"{len(chunk.capture_groups)} capture groups, but the profile caps max_images at "
            f"{max_images}; thinning image by image would split capture groups. Train the chunk "
            "with max_images: null (or at least the planned count), or plan smaller chunks"
        )


def staged_scene_scale(staged_dir: Path, global_scale: float) -> float:
    """``scene_scale`` upstream derives from the staged cameras (host-side port).

    gsplat 1.5.3: ``Parser.scene_scale`` is the largest distance of a camera centre from their
    mean over every parsed image, and the runner multiplies it by ``1.1 * global_scale``.
    ``normalize_world_space`` is refused, so the centres are the staged ``LOCAL_METRIC`` ones.
    """
    from minegs.ingest.common import colmap_io

    model = colmap_io.read_model(Path(staged_dir) / "sparse" / "0")
    c = np.stack([im.world_from_cam.matrix()[:3, 3] for im in model.images.values()])
    return float(np.max(np.linalg.norm(c - c.mean(axis=0), axis=1)) * 1.1 * global_scale)


def check_trainer_evidence(
    record: RunRecord, evidence: Any, staged_dir: Path, n_final: int
) -> None:
    """What the trainer says it ran, against what the run asked for (Phase 4 AD-8).

    Every mismatch is a refusal. A requested capability that the trainer's own config shows
    switched off is a different experiment, and so is one switched on that was never asked for.
    A run whose evidence cannot be read cannot show which experiment it was. ``real`` does not
    enter here: whether hardware ran is decided by the runtime evidence alone.
    """
    from minegs.train.backends.gsplat import (
        PINNED_GSPLAT,
        UPSTREAM_MCMC_NOISE_LR,
        UPSTREAM_MCMC_SCALE_REG,
    )

    expected = dict(record.expected_trainer_config or {})
    if evidence.trainer_config_sha256 is None:
        raise ContractError(
            "the trainer left no cfg.yml, so the configuration it actually ran under cannot be "
            "compared with the one requested (Phase 4 AD-8)"
        )
    cfg = dict(evidence.trainer_config_full or {})
    if not cfg:
        raise ContractError("the trainer's cfg.yml could not be read as a configuration")
    seen: dict[str, Any] = {}
    differ = []
    for key, want in sorted(expected.items()):
        got = _cfg_value(cfg, key)
        seen[key] = None if got is _MISSING else got
        if got is _MISSING:
            differ.append(f"{key}: missing (expected {want!r})")
        elif not _same(got, want):
            differ.append(f"{key}: {got!r} (expected {want!r})")
    record.trainer_config = seen
    record.trainer_config_sha256 = evidence.trainer_config_sha256
    if differ:
        raise ContractError(
            "the trainer did not run the configuration this run requested — "
            + "; ".join(differ)
            + ". A run is recorded as what it was, so it is not called succeeded."
        )

    # ---- the MineGS adapter, when it was the entrypoint
    trainer = dict(record.trainer or {})
    adapter = evidence.adapter_evidence
    if trainer.get("entrypoint") == "minegs_adapter":
        if adapter is None:
            raise ContractError(
                "the run was built to train through the MineGS adapter, but no adapter evidence "
                "was written; nothing shows its extensions were attached"
            )
        if (adapter.get("adapter") or {}).get("sha256") != trainer.get("adapter_sha256"):
            raise ContractError(
                "the adapter that ran is not the adapter this MineGS ships "
                f"({(adapter.get('adapter') or {}).get('sha256', '?')[:12]} vs "
                f"{str(trainer.get('adapter_sha256'))[:12]}): the image and the host disagree"
            )
        record.optimised_images = len(adapter.get("optimised_images") or [])
        ran = (adapter.get("upstream_trainer") or {}).get("sha256")
        hashed = (record.runtime or {}).get("trainer_sha256")
        if ran and hashed and ran != hashed:
            raise ContractError(
                f"the adapter ran upstream trainer {ran[:12]}, but the runner recorded "
                f"{hashed[:12]} as the trainer of this run"
            )
        if record.depth_supervision is None and adapter.get("depth_supervision") is not None:
            raise ContractError(
                "the adapter reports depth supervision in a run that recorded none; the run "
                "would be stated as something other than what trained"
            )
    elif adapter is not None:
        raise ContractError("adapter evidence appeared in a run that did not use the adapter")
    else:
        n = int((record.staged or {}).get("n_images") or 0)
        te = int(expected.get("test_every") or 8)
        record.optimised_images = n - (-(-n // te)) if n else None
    if record.chunk is not None:
        # Upstream derives scene_scale (learning rates, densification thresholds, the depth
        # term, MCMC's noise) from the staged cameras, so it differs chunk to chunk. Recorded,
        # not compensated (Phase 5 AD-9): the adapter's own value when it ran, otherwise
        # upstream's formula applied to the cameras this run staged.
        if adapter is not None:
            scale, source = adapter.get("scene_scale"), "adapter"
        else:
            global_scale = _cfg_value(cfg, "global_scale")
            if global_scale is _MISSING or isinstance(global_scale, bool):
                raise ContractError(
                    "the trainer's cfg.yml has no global_scale, so this chunk's scene_scale "
                    "cannot be stated"
                )
            scale = staged_scene_scale(staged_dir, float(global_scale))
            source = "host_from_staged_cameras"
        record.chunk = {**record.chunk, "scene_scale": scale, "scene_scale_source": source}

    # ---- depth supervision actually acted, on the artifact this run recorded
    sup = record.depth_supervision
    if sup is not None:
        got = (adapter or {}).get("depth_supervision")
        if not got:
            raise ContractError("depth supervision was requested and the trainer reports none")
        if got.get("artifact_sha256") != sup.get("artifact_sha256"):
            raise ContractError(
                f"the trainer consumed depth supervision {str(got.get('artifact_sha256'))[:12]}, "
                f"but this run recorded {str(sup.get('artifact_sha256'))[:12]}"
            )
        if not got.get("steps_with_depth_term"):
            raise ContractError(
                "depth supervision was loaded but no training step applied the depth term"
            )
        if not got.get("images_with_samples"):
            raise ContractError("no optimised image carried a depth sample")
        optimised = set((adapter or {}).get("optimised_images") or [])
        if not set(got.get("images_with_samples") or []) <= optimised:
            raise ContractError("the trainer reports depth samples on images it did not optimise")
        if not int(got.get("n_samples_in_domain") or 0) > 0:
            raise ContractError("the trainer reports no depth sample inside its images")
        calls = got.get("training_calls")
        if not isinstance(calls, int) or not 0 < int(got["steps_with_depth_term"]) <= calls:
            raise ContractError(
                f"the trainer reports the depth term on {got.get('steps_with_depth_term')} steps "
                f"of {calls} training renders; that cannot have happened"
            )
        if not _close(got.get("depth_lambda"), float(expected.get("depth_lambda", 0.01))):
            raise ContractError("the depth weight the trainer used is not the one requested")
        sup = dict(sup)
        sup["trainer"] = {
            k: got.get(k)
            for k in (
                "n_samples_in_domain",
                "n_samples_out_of_domain",
                "steps_with_depth_term",
                "depth_term_mean",
                "depth_term_last",
                "depth_lambda",
                "formula",
            )
        }
        sup["trainer"]["images_with_samples"] = len(got.get("images_with_samples") or [])
        sup["trainer"]["images_without_samples"] = len(got.get("images_without_samples") or [])
        record.depth_supervision = sup

    # ---- MCMC in metres: the host's expectation, the trainer's numbers, the trainer's config
    if expected.get("strategy") == "MCMCStrategy":
        s_host = staged_metric_scale(staged_dir)
        want_noise = UPSTREAM_MCMC_NOISE_LR * s_host * s_host
        want_reg = UPSTREAM_MCMC_SCALE_REG * s_host
        comp = (adapter or {}).get("mcmc_metric_compensation") or {}
        checks = (
            ("trainer scale", comp.get("s"), s_host),
            ("cfg.yml strategy.noise_lr", _cfg_value(cfg, "strategy.noise_lr"), want_noise),
            ("cfg.yml scale_reg", cfg.get("scale_reg"), want_reg),
            ("upstream noise_lr", comp.get("noise_lr_base"), UPSTREAM_MCMC_NOISE_LR),
            ("upstream scale_reg", comp.get("scale_reg_base"), UPSTREAM_MCMC_SCALE_REG),
        )
        bad = [
            f"{what}: {got!r} vs {want!r}" for what, got, want in checks if not _close(got, want)
        ]
        record.metric_compensation = {
            "s_host": s_host,
            "noise_lr_expected": want_noise,
            "scale_reg_expected": want_reg,
            "trainer": comp or None,
        }
        if bad:
            raise ContractError(f"MCMC metric compensation does not hold ({'; '.join(bad)})")

    # ---- export dropped nothing; the trained-with gsplat is the pinned one
    if evidence.stats_gaussian_count is None:
        raise ContractError(
            "the trainer's final gaussian count cannot be read from its last stats file; "
            "upstream always writes an integer num_GS there, so whether the export dropped "
            "diverged gaussians cannot be checked"
        )
    if evidence.stats_gaussian_count != n_final:
        raise ContractError(
            f"the trainer ended with {evidence.stats_gaussian_count} gaussians but its PLY holds "
            f"{n_final}: export drops non-finite splats silently (gsplat/exporter.py:515-538), "
            "so the difference is gaussians that diverged"
        )
    trained_with = (record.runtime or {}).get("gsplat")
    if trained_with and str(trained_with) != PINNED_GSPLAT:
        raise ContractError(f"trained with gsplat {trained_with}, pinned {PINNED_GSPLAT}")


def _in_run(outputs: list[Path], evidence: Any) -> Path:
    """The normalised copy of the backend's final model, by name."""
    by_name = {p.name: p for p in outputs}
    return by_name.get(evidence.final_model.name, outputs[-1])


def load_record(run_dir: str | Path) -> RunRecord:
    p = Path(run_dir) / "run.json"
    if not p.exists():
        raise ContractError(f"{run_dir}: no run.json")
    return RunRecord.load(p)


def get_runner(name: str, config: RunnerConfig | None = None) -> Runner:
    config = config or RunnerConfig(runner=name)
    if name == "local":
        from minegs.train.runner.local import LocalRunner

        return LocalRunner(config)
    if name == "runpod":
        from minegs.train.runner.runpod import RunPodRunner

        return RunPodRunner(config)
    raise ContractError(f"unknown runner {name!r}")


def docker_available() -> bool:
    return shutil.which("docker") is not None


def cuda_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return (
            subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10, check=False
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
