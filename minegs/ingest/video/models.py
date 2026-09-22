"""The frame-set artifact (§Phase 3 C1) — what an SfM reconstruction was actually fed.

Before this existed, `minegs ingest video select` wrote a list of decisions that nothing read
and `minegs ingest video sfm` feature-extracted every image in a directory. The two were not
connected, so a frame the selector rejected went into the reconstruction anyway, and a
reconstruction could not say afterwards which bytes it had seen.

``FrameSetRecord`` closes that. It is the identity of one SfM input set: which video (by
digest, not by name), with which extraction settings, which frames survived selection and why,
for 360 which crop came from which panorama at which yaw and pitch, which masks cover which
images, and the digest of every image in the set. ``check_frameset`` re-reads all of it from
disk, so an edited frame or a swapped crop is a different frame set rather than the same one
with different contents.

Two rules are worth stating outright, because both were holes the Phase 3 audit found:

* **A crop's orientation is the record, not its filename.** ``p0y03/`` is a convenience. The
  yaw, pitch and intrinsics live here, bound to the crop's digest, and the rig configuration is
  derived from *this* rather than from a directory listing. A crop whose bytes change is caught
  even when its path does not.
* **A 360 video's panorama convention has to be declared.** The library default names E57 as
  its source, which is true of a scanner's panorama and false of a camera's. An unmeasured
  convention presented as measured is fabricated evidence, so the video path requires one
  explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from minegs.core.config import VersionedModel
from minegs.core.errors import ContractError
from minegs.core.provenance import ProvenanceRecord, sha256_file
from minegs.ingest.common.geometry import PanoConvention

FRAMESET_FILE = "frameset.json"

#: Suffixes a frame set reads, matched case-insensitively. Defined here rather than beside the
#: builder because the check that the SfM input set is *exactly* the record has to enumerate the
#: directory the same way the builder filled it; two lists would mean a file one of them counts
#: and the other does not.
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})

#: What kind of input a frame set was made from. ``video360`` is the only kind that carries
#: crops and a panorama convention; ``image_set`` skips extraction entirely.
FrameSetKind = Literal["video", "video360", "image_set"]

#: How the frames got onto disk. ``copy`` is the substitution used where ffmpeg is absent — it
#: is recorded rather than hidden, because a frame set assembled by copying files is not
#: evidence that this project can decode that operator's video.
ExtractionMethod = Literal["ffmpeg", "copy", "none"]

MaskMethod = Literal["nadir_crop", "nadir_equirect", "box"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImageEntry(_Strict):
    """One image in the set handed to SfM, by name and by content."""

    name: str
    sha256: str = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _relative(cls, v: str) -> str:
        p = Path(v)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"image name must be relative to the frame set, got {v!r}")
        return v


class FrameDecision(_Strict):
    """Why one extracted frame is in the set or is not.

    Rejected frames stay in the record. Which frames were dropped, and on which threshold, is
    part of what a reconstruction rests on — a set that kept 40 of 400 is a different kind of
    evidence from one that kept 390, and the numbers are gone if only the survivors are listed.
    """

    name: str
    blur: float
    phash: str = Field(min_length=1)
    keep: bool
    reason: str = ""


class SelectionRecord(_Strict):
    blur_threshold: float
    hamming_threshold: int
    max_frames: int | None = None
    considered: int = Field(ge=0)
    kept: int = Field(ge=0)
    decisions: list[FrameDecision] = Field(default_factory=list)


class ExtractionRecord(_Strict):
    method: ExtractionMethod
    argv: list[str] = Field(default_factory=list)
    fps: float | None = None
    scale_width: int | None = None
    start_s: float | None = None
    duration_s: float | None = None
    pattern: str | None = None
    tool_version: str | None = None
    #: False when the frames were produced by something other than the real decoder. The Phase
    #: 3 report reads this; a substituted extraction cannot support a claim about a real survey.
    real_execution: bool = True


class CropRecord(_Strict):
    """One perspective crop of one equirectangular frame, with the geometry that made it."""

    name: str
    parent_frame: str
    view: str
    yaw_deg: float
    pitch_deg: float
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    sha256: str = Field(min_length=1)

    def camera_params(self) -> list[float]:
        """COLMAP ``PINHOLE`` parameters for this crop."""
        return [self.fx, self.fy, self.cx, self.cy]


class MaskRecord(_Strict):
    image: str
    mask_file: str
    method: MaskMethod
    parameters: dict[str, Any] = Field(default_factory=dict)
    sha256: str = Field(min_length=1)


class FrameSetRecord(VersionedModel):
    """``<frameset_dir>/frameset.json`` — the identity of one SfM input set."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    frameset_id: str = Field(min_length=1)
    kind: FrameSetKind
    #: Name only, never a path: where the operator keeps their raw survey is not this record's
    #: business, and the digest below is what identifies it.
    source_name: str | None = None
    source_sha256: str | None = None
    extraction: ExtractionRecord
    selection: SelectionRecord | None = None
    #: Declared, never defaulted, for ``video360``. See the module docstring.
    pano_convention: PanoConvention | None = None
    crop_spec: dict[str, Any] | None = None
    crops: list[CropRecord] = Field(default_factory=list)
    masks: list[MaskRecord] = Field(default_factory=list)
    #: Relative to the frame set directory. Where the SfM backend is pointed.
    images_dir: str = "images"
    masks_dir: str | None = None
    #: Exactly the images SfM is given — the survivors of selection, or their crops.
    images: list[ImageEntry] = Field(min_length=1)
    #: Order-independent digest of the set above. What downstream artifacts name.
    images_sha256: str = Field(min_length=1)
    provenance: ProvenanceRecord

    @field_validator("images_dir", "masks_dir")
    @classmethod
    def _relative_dir(cls, v: str | None) -> str | None:
        if v is None:
            return None
        p = Path(v)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"directory must be relative to the frame set, got {v!r}")
        return v

    def image_root(self, frameset_dir: str | Path) -> Path:
        return Path(frameset_dir) / self.images_dir

    def mask_root(self, frameset_dir: str | Path) -> Path | None:
        return None if self.masks_dir is None else Path(frameset_dir) / self.masks_dir


def images_digest(images: list[ImageEntry]) -> str:
    """Order-independent digest of a set of (name, content) pairs."""
    import hashlib

    h = hashlib.sha256()
    for entry in sorted(images, key=lambda e: e.name):
        h.update(entry.name.encode())
        h.update(entry.sha256.encode())
    return h.hexdigest()


def find_frameset(path: str | Path) -> Path | None:
    """``frameset.json`` for *path*, or None when *path* is not a frame set artifact."""
    p = Path(path)
    if p.is_dir():
        j = p / FRAMESET_FILE
        return j if j.is_file() else None
    if p.is_file() and p.name == FRAMESET_FILE:
        return p
    return None


def load_frameset(path: str | Path) -> tuple[FrameSetRecord, Path]:
    """Load the artifact at *path* (directory or ``frameset.json``) and its directory."""
    found = find_frameset(path)
    if found is None:
        raise ContractError(f"{path}: not a frame set artifact (no {FRAMESET_FILE})")
    return FrameSetRecord.load(found), found.parent


def check_frameset(rec: FrameSetRecord, frameset_dir: str | Path) -> None:
    """Re-read the frame set off disk and refuse it if it is not what the record describes.

    Every consumer of a frame set — the SfM backend, the dataset builder through the SfM
    record — hangs off ``images_sha256``. That digest is only worth something if somebody
    recomputes it from the files, so this does, together with the crop and mask bindings that
    make a 360 set say where its geometry came from.
    """
    root = rec.image_root(frameset_dir)
    if not root.is_dir():
        raise ContractError(f"frame set {rec.frameset_id}: no images directory at {root}")

    for entry in rec.images:
        path = root / entry.name
        if not path.is_file():
            raise ContractError(
                f"frame set {rec.frameset_id}: {entry.name} is in the record and not on disk"
            )
        digest = sha256_file(path)
        if digest != entry.sha256:
            raise ContractError(
                f"frame set {rec.frameset_id}: {entry.name} hashes to {digest[:12]}, but the "
                f"record says {entry.sha256[:12]}. These are not the images that were selected."
            )

    recomputed = images_digest(rec.images)
    if recomputed != rec.images_sha256:
        raise ContractError(
            f"frame set {rec.frameset_id}: the image list digests to {recomputed[:12]}, but the "
            f"record says {rec.images_sha256[:12]}; the list was edited after it was written"
        )

    _check_no_stowaways(rec, root)
    _check_selection_consistency(rec)
    _check_crops(rec, root)
    _check_masks(rec, frameset_dir)


def _check_no_stowaways(rec: FrameSetRecord, root: Path) -> None:
    """``images/`` must hold the record's images and nothing else.

    The record's digests prove that every image it names is unchanged. They say nothing about a
    file it does not name, and the SfM backend is pointed at this directory rather than at the
    list: it globs. So a frame dropped by selection, or one from another survey entirely,
    reaches the reconstruction by being copied in afterwards, with every recorded digest still
    matching. Enumerating the directory is what closes that, and it is why the check reads the
    world instead of the record.
    """
    on_disk = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }
    surplus = sorted(on_disk - {e.name for e in rec.images})
    if surplus:
        raise ContractError(
            f"frame set {rec.frameset_id}: {len(surplus)} image(s) under {root} are not in the "
            f"record (e.g. {surplus[:5]}). SfM is given this directory, not the list, so a file "
            "put here afterwards would be reconstructed from without appearing anywhere."
        )


def _check_selection_consistency(rec: FrameSetRecord) -> None:
    """The set must be the frames selection kept — no more, no fewer.

    This is the hole the artifact exists to close: a rejected frame reaching the reconstruction
    anyway. For a 360 set the images are crops of the kept panoramas, so the comparison runs
    through ``crops`` rather than over the names directly.
    """
    if rec.selection is None:
        return
    kept = {d.name for d in rec.selection.decisions if d.keep}
    dropped = {d.name for d in rec.selection.decisions if not d.keep}
    if not rec.selection.decisions:
        return
    if rec.crops:
        used_parents = {c.parent_frame for c in rec.crops}
        stowaways = sorted(used_parents & dropped)
        if stowaways:
            raise ContractError(
                f"frame set {rec.frameset_id}: crops were taken from {stowaways[:5]}, which "
                "selection rejected. A frame the selector dropped cannot reach the "
                "reconstruction through its crops."
            )
        missing = sorted(kept - used_parents)
        if missing:
            raise ContractError(
                f"frame set {rec.frameset_id}: selection kept {missing[:5]} but no crop names "
                "them as a parent; the set is not the selection's result"
            )
        return
    names = {e.name for e in rec.images}
    stowaways = sorted(names & dropped)
    if stowaways:
        raise ContractError(
            f"frame set {rec.frameset_id}: {stowaways[:5]} are in the SfM input set and "
            "selection rejected them"
        )
    missing = sorted(kept - names)
    if missing:
        raise ContractError(
            f"frame set {rec.frameset_id}: selection kept {missing[:5]}, which are not in the "
            "SfM input set; the set is not the selection's result"
        )


def _check_crops(rec: FrameSetRecord, image_root: Path) -> None:
    if not rec.crops:
        if rec.kind == "video360":
            raise ContractError(
                f"frame set {rec.frameset_id} is a 360 set with no crop records; the rig and "
                "every camera in it are derived from those records, so there is nothing to "
                "derive them from"
            )
        return
    if rec.pano_convention is None:
        raise ContractError(
            f"frame set {rec.frameset_id}: crops were cut without a declared panorama "
            "convention. An unmeasured azimuth sign or elevation flip becomes a fixed "
            "geometric constraint on the reconstruction, so it has to be stated, not defaulted."
        )
    by_name = {e.name: e for e in rec.images}
    for crop in rec.crops:
        entry = by_name.get(crop.name)
        if entry is None:
            raise ContractError(
                f"frame set {rec.frameset_id}: crop {crop.name} is recorded but is not in the "
                "SfM input set"
            )
        if entry.sha256 != crop.sha256:
            raise ContractError(
                f"frame set {rec.frameset_id}: crop {crop.name} is recorded at yaw "
                f"{crop.yaw_deg:g}°/pitch {crop.pitch_deg:g}° with digest "
                f"{crop.sha256[:12]}, but the image in the set hashes to {entry.sha256[:12]}. "
                "The orientation of a crop is this record, not its path, so a crop whose bytes "
                "changed is a different view however it is named."
            )
        path = image_root / crop.name
        if path.is_file() and sha256_file(path) != crop.sha256:  # pragma: no cover - see above
            raise ContractError(f"frame set {rec.frameset_id}: crop {crop.name} moved on disk")


def _check_masks(rec: FrameSetRecord, frameset_dir: str | Path) -> None:
    """Masks must name images that exist, because COLMAP ignores the ones that do not.

    A mask whose filename does not line up with its image is not an error there: the feature
    extractor simply does not apply it, and the tripod it was meant to hide is matched like any
    other texture. The reconstruction succeeds and is quietly worse, which is exactly the class
    of failure this project refuses to leave silent.
    """
    if not rec.masks:
        return
    root = rec.mask_root(frameset_dir)
    if root is None:
        raise ContractError(
            f"frame set {rec.frameset_id}: masks are recorded but masks_dir is unset"
        )
    names = {e.name for e in rec.images}
    for mask in rec.masks:
        if mask.image not in names:
            raise ContractError(
                f"frame set {rec.frameset_id}: mask {mask.mask_file} is for {mask.image}, which "
                "is not in the SfM input set. COLMAP would ignore it without a word."
            )
        path = root / mask.mask_file
        if not path.is_file():
            raise ContractError(
                f"frame set {rec.frameset_id}: mask {mask.mask_file} is recorded and not on disk"
            )
        digest = sha256_file(path)
        if digest != mask.sha256:
            raise ContractError(
                f"frame set {rec.frameset_id}: mask {mask.mask_file} hashes to {digest[:12]}, "
                f"the record says {mask.sha256[:12]}"
            )


def expected_mask_name(image_name: str) -> str:
    """COLMAP looks for ``<mask_path>/<image_name>.png``, keeping the image's own extension."""
    return f"{image_name}.png"


__all__ = [
    "FRAMESET_FILE",
    "IMAGE_SUFFIXES",
    "CropRecord",
    "ExtractionRecord",
    "FrameDecision",
    "FrameSetKind",
    "FrameSetRecord",
    "ImageEntry",
    "MaskRecord",
    "SelectionRecord",
    "check_frameset",
    "expected_mask_name",
    "find_frameset",
    "images_digest",
    "load_frameset",
]
