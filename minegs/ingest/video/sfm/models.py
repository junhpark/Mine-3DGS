"""The SfM artifact (§Phase 3 C1) — a reconstruction that survives its own process.

``SfMResult`` said how many images registered and then went away with the interpreter. What
stayed on disk was a COLMAP model in a directory, indistinguishable in shape from the
``sparse/0`` of a metric dataset and carrying nothing that said which images produced it, which
binary, with which options, or that its coordinates mean nothing until they are registered.

``SfmRecord`` is that missing statement. It names the frame set by digest, the backend by
version, the commands that actually ran, the component that was chosen out of however many
COLMAP produced, and the model's own digest. Two of its fields are load-bearing rather than
descriptive:

* ``frame: SFM_INTERNAL`` — this reconstruction is in its own coordinates. Every metric
  consumer in the repository asks for ``TLS_GLOBAL`` or ``LOCAL_METRIC`` by name, so it is
  refused by construction until a measured Sim(3) moves it.
* ``metric_state: arbitrary_scale`` — and the only thing that changes it is a registration
  artifact, never an edit to this file.

``real_sfm_execution`` is false when the reconstruction came from the substituted backend the
structural gate uses. It is recorded next to every number the record carries, because a
reconstruction nothing reconstructed is not evidence about a mine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from minegs.core.config import VersionedModel
from minegs.core.errors import ContractError
from minegs.core.frames import Frame
from minegs.core.provenance import ProvenanceRecord, sha256_tree

SFM_FILE = "sfm.json"

#: The files a COLMAP text model is made of. The digest covers exactly these, so a model is
#: the same model when its directory picks up a log file and different when a pose changes.
SPARSE_FILES = ("cameras.txt", "images.txt", "points3D.txt")

MetricState = Literal["arbitrary_scale", "registered_metric"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentSummary(_Strict):
    """One reconstruction COLMAP produced. There is usually one, and sometimes not."""

    name: str
    registered_images: int = Field(ge=0)
    points: int = Field(ge=0)
    sha256: str = Field(min_length=1)


class CameraSummary(_Strict):
    camera_id: int
    model: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    params: list[float]


class SfmRecord(VersionedModel):
    """``<sfm_dir>/sfm.json`` — what this reconstruction is, and of what."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    sfm_id: str = Field(min_length=1)
    frameset_id: str = Field(min_length=1)
    images_sha256: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    backend_version: str | None = None
    #: The commands that ran, not the ones that were composed. A substituted backend records
    #: an empty list rather than the commands it did not execute.
    commands: list[list[str]] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)
    #: Where the intrinsics came from, when they were not estimated. Fixing a reconstruction
    #: onto intrinsics measured somewhere else is allowed and is never silent.
    intrinsics_source: str = "estimated_by_sfm"
    #: Relative to the directory holding this file.
    model_dir: str = "sparse/0"
    model_sha256: str = Field(min_length=1)
    components: list[ComponentSummary] = Field(default_factory=list)
    selected_component: str = Field(min_length=1)
    registered_images: int = Field(ge=0)
    points: int = Field(ge=0)
    cameras: list[CameraSummary] = Field(default_factory=list)
    frame: Literal["SFM_INTERNAL"] = Frame.SFM_INTERNAL.value
    metric_state: MetricState = "arbitrary_scale"
    #: False when the reconstruction came from a substituted backend (the structural gate).
    real_sfm_execution: bool
    provenance: ProvenanceRecord

    @field_validator("model_dir")
    @classmethod
    def _relative(cls, v: str) -> str:
        p = Path(v)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"model_dir must be relative to the sfm directory, got {v!r}")
        return v

    def model_path(self, sfm_dir: str | Path) -> Path:
        return Path(sfm_dir) / self.model_dir


def model_digest(model_dir: str | Path) -> str:
    """Digest of a COLMAP text model — its three files, nothing else."""
    return sha256_tree(model_dir, SPARSE_FILES)


def find_sfm(path: str | Path) -> Path | None:
    p = Path(path)
    if p.is_dir():
        j = p / SFM_FILE
        return j if j.is_file() else None
    if p.is_file() and p.name == SFM_FILE:
        return p
    return None


def load_sfm(path: str | Path) -> tuple[SfmRecord, Path]:
    found = find_sfm(path)
    if found is None:
        raise ContractError(f"{path}: not an SfM artifact (no {SFM_FILE})")
    return SfmRecord.load(found), found.parent


def check_sfm(rec: SfmRecord, sfm_dir: str | Path, *, images_sha256: str | None = None) -> Path:
    """Re-read the model off disk and refuse it if it is not the one the record describes.

    The attack this closes is the cheap one: leave ``sfm.json`` in place and swap the model
    beside it — for a TLS-derived model, say, whose ``sparse/0`` has exactly the same shape.
    The digest is what makes the record a statement about these poses and these points.
    """
    model_dir = rec.model_path(sfm_dir)
    missing = [n for n in SPARSE_FILES if not (model_dir / n).is_file()]
    if missing:
        raise ContractError(
            f"SfM {rec.sfm_id}: {model_dir} is missing {missing}; it is not a COLMAP text model"
        )
    digest = model_digest(model_dir)
    if digest != rec.model_sha256:
        raise ContractError(
            f"SfM {rec.sfm_id}: the model at {model_dir} hashes to {digest[:12]}, but the "
            f"record says {rec.model_sha256[:12]}. These are not the cameras and points this "
            "reconstruction produced."
        )
    if images_sha256 is not None and images_sha256 != rec.images_sha256:
        raise ContractError(
            f"SfM {rec.sfm_id} was reconstructed from image set {rec.images_sha256[:12]}, but "
            f"the frame set now digests to {images_sha256[:12]}; the images changed after the "
            "reconstruction, so this model is of a different set of pictures"
        )
    if rec.metric_state != "arbitrary_scale":
        raise ContractError(
            f"SfM {rec.sfm_id} declares metric_state={rec.metric_state!r}. An independent "
            "reconstruction is arbitrary-scale until a measured registration says otherwise, "
            "and that promotion lives in a RegistrationRecord, not in this file."
        )
    return model_dir


__all__ = [
    "SFM_FILE",
    "SPARSE_FILES",
    "CameraSummary",
    "ComponentSummary",
    "SfmRecord",
    "check_sfm",
    "find_sfm",
    "load_sfm",
    "model_digest",
]
