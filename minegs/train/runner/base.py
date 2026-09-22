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


class RunnerConfig(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "1.0"
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
    overrides: dict[str, Any] = Field(default_factory=dict)


RUN_RECORD_MIGRATIONS = MigrationRegistry("run_record")


@RUN_RECORD_MIGRATIONS.register("1.0", "1.1")
def _run_1_0_to_1_1(d: dict[str, Any]) -> dict[str, Any]:
    """1.1 adds Phase 0D.2 execution evidence; 1.0 records simply have none.

    Nothing is translated because nothing moved: every 1.1 field is new, and a 1.0 run genuinely
    did not record whether its artifacts were ever checked. Leaving them unset says that, which
    is the honest reading — inventing a ``succeeded`` run's missing evidence would not be.
    """
    return d


class RunRecord(VersionedModel):
    """``runs/<run_id>/run.json`` (§9)."""

    SCHEMA_VERSION: ClassVar[str] = "1.1"
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
        T_tls_from_local = manifest.T_tls_from_local
        if run.chunk_id:
            if manifest.chunks is None:
                raise ContractError("run.chunk_id set but manifest has no chunks")
            chunk = next((c for c in manifest.chunks.items if c.id == run.chunk_id), None)
            if chunk is None:
                raise ContractError(f"unknown chunk {run.chunk_id}")
            if chunk.T_tls_from_local is not None:
                from minegs.core.frames import SE3

                T_tls_from_local = SE3.from_matrix(chunk.T_tls_from_local)
        # `is not None`, not truthiness: resume_from="" is still a resume *request*, and one
        # that names nothing is the least honourable of all — under a truthiness test it would
        # fall through to a fresh iteration-0 run, which is the exact silent restart this phase
        # exists to forbid. No CLI value can produce it today (typer renders --resume-from ""
        # as Path(".")), but the guard should not depend on that.
        if run.resume_from is not None:
            refuse_resume(backend)
        refuse_used_run_dir(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        record = RunRecord(
            run_id=run.run_id,
            dataset_id=manifest.dataset_id,
            dataset_hash=sha256_tree(dataset_dir, DATASET_HASH_PATTERNS),
            chunk_id=run.chunk_id,
            backend={"name": backend.name, "version": backend.version()},
            profile=profile.model_dump(mode="json"),
            runner=self.name,
            docker_digest=self.config.image_digest(),
            T_tls_from_local=T_tls_from_local.to_list(),
            provenance=stamp(run.model_dump(mode="json"), parents=[manifest.dataset_id]),
        )
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
    return info


#: Asked of the image itself, so the recorded versions are the ones that trained.
_PROBE = (
    "import json,sys,torch;"
    "d={'python':sys.version.split()[0],'torch':torch.__version__,"
    "'torch_cuda':torch.version.cuda,'torch_cuda_available':torch.cuda.is_available(),"
    "'device_count':torch.cuda.device_count()};"
    "d['gpu_model']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None;"
    "\ntry:\n import gsplat; d['gsplat']=gsplat.__version__\nexcept Exception as e:"
    " d['gsplat']=None\n"
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
            "not a metric one (§0D.2 D2-7)"
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
