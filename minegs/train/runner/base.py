"""Runner contract (§8.2)::

    Runner.submit(run_config) -> RunHandle
    RunHandle.status() / .logs() / .fetch_artifacts()

Both runners execute the *same* GPU image digest; ``run.json`` records it (§9).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from minegs.core.config import VersionedModel
from minegs.core.errors import ContractError
from minegs.core.manifest import Manifest
from minegs.core.provenance import ProvenanceRecord, make_id, sha256_tree, stamp
from minegs.train.backends import get_backend
from minegs.train.profiles import Profile, load_profile


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
    gpus: str = "all"
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
    resume: bool = False
    chunk_id: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)


class RunRecord(VersionedModel):
    """``runs/<run_id>/run.json`` (§9)."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"
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
    T_tls_from_local: list[list[float]] | None = None
    frame_of_outputs: str = "LOCAL_METRIC"
    outputs: list[str] = Field(default_factory=list)
    provenance: ProvenanceRecord


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
        missing = backend.check_profile(profile)
        if missing:
            raise ContractError(
                f"profile {profile.name} requires {missing}, backend {backend.name} lacks them"
            )
        if not run.run_id:
            run.run_id = make_id(backend.name)
        if not run.run_dir:
            run.run_dir = str(dataset_dir.parent / "runs" / run.run_id)
        run_dir = Path(run.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
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
        record = RunRecord(
            run_id=run.run_id,
            dataset_id=manifest.dataset_id,
            dataset_hash=sha256_tree(
                dataset_dir, ("manifest.json", "sparse/0/*.txt", "init_points.ply")
            ),
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
        return record.save(run_dir / "run.json")


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
