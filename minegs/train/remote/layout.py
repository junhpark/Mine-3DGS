"""Where everything lives on the network volume (Phase 6 §4).

Every path is derived from identities, never from a string the user appends: the dataset by its
hash, sidecars by their digests, a job and its run by the run id. Two datasets cannot collide, a
new push cannot overwrite bytes another run is reading, and a run cannot inherit another's files.
The same relative path is reached locally through the storage remote and in the pod under the
volume mount.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from minegs.core.errors import ContractError

#: Identifiers that become path segments and worker arguments. The RunPod SDK interpolates
#: ``docker_args`` into GraphQL without escaping, so nothing outside this set may reach it.
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SAFE_MOUNT = re.compile(r"^/[A-Za-z0-9_./-]+$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

ROOT = "minegs"
INPUTS_FILE = "inputs.json"
STATUS_FILE = "status.json"
CLAIM_FILE = "claim"
WORKER_LOG = "worker.log"
CANCEL_FILE = "cancel_requested.json"
DATASET_CLAIM_FILE = "dataset.json"
CHUNK_PLAN_FILE = "chunk_plan.json"
OUTPUT_MANIFEST_FILE = "output_manifest.json"


def safe_id(value: str, what: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.match(value) or ".." in value:
        raise ContractError(
            f"{what} {value!r} cannot name a remote path or a worker argument; it must match "
            f"{SAFE_ID.pattern} (the provider passes worker arguments unescaped)"
        )
    return value


def safe_mount(value: str) -> str:
    p = PurePosixPath(str(value))
    if (
        not SAFE_MOUNT.match(str(value))
        or str(p) == "/"
        or ".." in p.parts
        or str(p) != str(value).rstrip("/")
    ):
        raise ContractError(
            f"volume_mount {value!r} is not a usable mount point: an absolute path other than /, "
            "without '..', made of [A-Za-z0-9_./-]"
        )
    return str(p)


def _hex64(value: str, what: str) -> str:
    if not isinstance(value, str) or not _HEX64.match(value):
        raise ContractError(f"{what} {value!r} is not a sha256 hex digest")
    return value


@dataclass(frozen=True)
class RemoteLayout:
    """Relative paths under the volume root, and their pod paths under ``volume_mount``."""

    volume_mount: str
    dataset_id: str

    def __post_init__(self) -> None:
        safe_mount(self.volume_mount)
        safe_id(self.dataset_id, "dataset_id")

    @property
    def base(self) -> str:
        return f"{ROOT}/{self.dataset_id}"

    def dataset(self, dataset_hash: str) -> str:
        return f"{self.base}/datasets/{_hex64(dataset_hash, 'dataset_hash')}"

    def chunk_plan(self, plan_digest: str) -> str:
        return f"{self.base}/sidecars/chunk_plans/{_hex64(plan_digest, 'plan_digest')}"

    def depth(self, artifact_sha256: str) -> str:
        return f"{self.base}/sidecars/depth/{_hex64(artifact_sha256, 'depth artifact sha256')}"

    def job(self, run_id: str) -> str:
        return f"{self.base}/jobs/{safe_id(run_id, 'run_id')}"

    def run(self, run_id: str) -> str:
        return f"{self.base}/runs/{safe_id(run_id, 'run_id')}"

    def pod(self, rel: str) -> str:
        """The pod-side absolute path of a volume-relative path."""
        return f"{self.volume_mount}/{rel}"

    def worker_args(self, run_id: str) -> str:
        """What the pod runs after the image ENTRYPOINT (``minegs``)."""
        return f"train remote-worker {self.pod(self.job(run_id))}/{INPUTS_FILE}"


__all__ = [
    "CANCEL_FILE",
    "CHUNK_PLAN_FILE",
    "CLAIM_FILE",
    "DATASET_CLAIM_FILE",
    "INPUTS_FILE",
    "OUTPUT_MANIFEST_FILE",
    "STATUS_FILE",
    "WORKER_LOG",
    "RemoteLayout",
    "safe_id",
    "safe_mount",
]
