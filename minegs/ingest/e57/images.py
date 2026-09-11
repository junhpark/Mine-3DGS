"""Image asset discovery (Phase 0B.2) — what image data exists, and nothing more.

This module answers "which images are here and what are they?". It deliberately does not
answer "which station does this image belong to?" — that is :mod:`minegs.ingest.e57.mapping`,
and it requires evidence rather than inference.

Two sources:

* **embedded** — entries under an E57's ``/images2D``.
* **external** — image files in a directory next to the survey.

Both produce the same :class:`ImageAsset` contract. Discovery never opens an image's pixel
data: an E57 blob's byte count comes from the node, and an external file's dimensions come
from the header PIL reads, so a directory of 4096x4096 panoramas costs no more than its
directory listing plus a few hundred header bytes each.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from minegs.ingest.e57 import _nodes

#: How an E57 says an image is projected. ``unknown`` covers both "the file declares a
#: representation we do not model" and "an external file, whose projection nothing states".
Representation = Literal["spherical", "cylindrical", "pinhole", "visual_reference", "unknown"]

ImageSource = Literal["e57_embedded", "external_file"]

#: E57 representation node name -> our label.
_REPRESENTATION_NODES: dict[str, Representation] = {
    "sphericalRepresentation": "spherical",
    "cylindricalRepresentation": "cylindrical",
    "pinholeRepresentation": "pinhole",
    "visualReferenceRepresentation": "visual_reference",
}

#: Representations whose geometry is a full or partial sphere, i.e. panorama-shaped.
#: A pinhole image is NOT one, even when six of them tile a sphere: that is a cube map, and
#: recognising it takes evidence about the set, not about one image (Phase 0C).
PANORAMA_REPRESENTATIONS: frozenset[str] = frozenset({"spherical", "cylindrical"})

_BLOB_FIELDS = ("jpegImage", "pngImage")
_EXTERNAL_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


class ImageAsset(BaseModel):
    """One discovered image. What the source declares, with nothing inferred."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    source: ImageSource
    source_index: int
    #: Vendor identifier, kept for provenance. Never used as this image's identity.
    guid: str | None = None
    name: str | None = None
    #: E57 ``associatedData3DGuid``. Mapping evidence, resolved in :mod:`.mapping`.
    associated_scan_guid: str | None = None

    representation: Representation = "unknown"
    #: The raw node/extension name behind ``representation``, for diagnosis.
    representation_source: str | None = None
    width: int | None = None
    height: int | None = None

    #: Embedded only: which blob field holds the pixels, and how many bytes it is.
    blob_field: str | None = None
    blob_bytes: int | None = None
    #: External only: the file on disk.
    path: str | None = None

    vendor_metadata: dict[str, Any] | None = None
    #: Problems: this image cannot be interpreted or extracted as declared.
    issues: list[str] = Field(default_factory=list)
    #: Observations worth telling the user, but not problems.
    notes: list[str] = Field(default_factory=list)

    @property
    def panorama_candidate(self) -> bool:
        """Whether the declared projection is panorama-shaped.

        A candidate, never a mapping: this says what the image *is*, not which station it
        belongs to. ``mapping.py`` decides that, and only from evidence.
        """
        return self.representation in PANORAMA_REPRESENTATIONS

    @property
    def is_extractable(self) -> bool:
        """Whether Phase 0B.3 can write this image out as a file."""
        if self.source == "external_file":
            return self.path is not None
        return self.blob_field is not None


def image_id_for(index: int) -> str:
    """``0 -> "image_000"``, from the image's index inside its source.

    Same philosophy as ``scan_id_for``: independent of vendor name and GUID so a file that
    omits those is still referable, dependent on order within the source, and unambiguous
    once paired with the source file's SHA-256 in provenance.
    """
    if index < 0:
        raise ValueError("image index must be >= 0")
    return f"image_{index:03d}"


def classify_representation(node: Any) -> tuple[Representation, str | None]:
    """Which representation an ``/images2D`` entry declares.

    Returns ``("unknown", None)`` when the entry declares none we model, and never falls back
    to a different projection: reading a cylindrical image as spherical would silently distort
    every pixel coordinate derived from it.
    """
    for key, label in _REPRESENTATION_NODES.items():
        if _nodes.is_defined(node, key):
            return label, key
    return "unknown", None


def _representation_details(node: Any, rep_node_name: str | None) -> dict[str, Any]:
    if rep_node_name is None:
        return {}
    rep = node[rep_node_name]
    out: dict[str, Any] = {
        "width": _nodes.value(rep, "imageWidth"),
        "height": _nodes.value(rep, "imageHeight"),
        "blob_field": None,
        "blob_bytes": None,
    }
    for blob_field in _BLOB_FIELDS:
        if _nodes.is_defined(rep, blob_field):
            out["blob_field"] = blob_field
            try:
                out["blob_bytes"] = int(rep[blob_field].byteCount())
            except Exception:
                out["blob_bytes"] = None
            break
    return out


def _vendor_metadata(node: Any, rep_node_name: str | None) -> dict[str, Any] | None:
    """Intrinsics and pose numbers kept verbatim for later phases, interpreted by none."""
    out: dict[str, Any] = {}
    if rep_node_name is not None:
        rep = node[rep_node_name]
        for key in (
            "focalLength",
            "pixelWidth",
            "pixelHeight",
            "principalPointX",
            "principalPointY",
            "radius",
        ):
            v = _nodes.value(rep, key)
            if v is not None:
                out[key] = v
    if _nodes.is_defined(node, "pose"):
        pose = node["pose"]
        rot = [_nodes.value(pose, "rotation", k) for k in ("w", "x", "y", "z")]
        tr = [_nodes.value(pose, "translation", k) for k in ("x", "y", "z")]
        if all(v is not None for v in rot):
            out["pose_rotation_wxyz"] = [float(v) for v in rot]
        if all(v is not None for v in tr):
            out["pose_translation"] = [float(v) for v in tr]
    return out or None


def discover_embedded_images(path: str | Path) -> list[ImageAsset]:
    """Enumerate an E57's ``/images2D``. Metadata only; no blob is read."""
    with _nodes.open_e57(path) as handle:
        root = handle.root
        if not _nodes.is_defined(root, "images2D"):
            return []
        try:
            images = root["images2D"]
            count = int(images.childCount())
        except Exception as e:
            raise _unenumerable(path, e) from e
        assets: list[ImageAsset] = []
        for i in range(count):
            node = images.get(i)
            rep, rep_node_name = classify_representation(node)
            details = _representation_details(node, rep_node_name)
            issues: list[str] = []
            if rep == "unknown":
                issues.append(
                    "entry declares no representation this reader models; it is recorded but "
                    "cannot be interpreted or extracted"
                )
            elif details.get("blob_field") is None:
                issues.append(
                    f"{rep} representation declares no {' or '.join(_BLOB_FIELDS)} blob; "
                    "pixels are not embedded in this file"
                )
            assets.append(
                ImageAsset(
                    image_id=image_id_for(i),
                    source="e57_embedded",
                    source_index=i,
                    guid=_nodes.str_value(node, "guid"),
                    name=_nodes.str_value(node, "name"),
                    associated_scan_guid=_nodes.str_value(node, "associatedData3DGuid"),
                    representation=rep,
                    representation_source=rep_node_name,
                    width=_as_int(details.get("width")),
                    height=_as_int(details.get("height")),
                    blob_field=details.get("blob_field"),
                    blob_bytes=details.get("blob_bytes"),
                    vendor_metadata=_vendor_metadata(node, rep_node_name),
                    issues=issues,
                )
            )
    return assets


def _unenumerable(path: str | Path, error: Exception) -> Exception:
    from minegs.ingest.e57.exceptions import E57UnsupportedStructureError

    return E57UnsupportedStructureError(path, f"/images2D could not be enumerated ({error})")


def _as_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def discover_external_images(
    directory: str | Path, suffixes: tuple[str, ...] = _EXTERNAL_SUFFIXES
) -> list[ImageAsset]:
    """Enumerate image files in a directory, sorted by name for a deterministic index.

    Every asset comes back with ``representation="unknown"``. A JPEG on disk states nothing
    about its projection: an equirectangular panorama and a pinhole photo are the same file
    format, and the file name is not evidence (§10). Declaring it is the user's job, and
    mapping it needs an explicit mapping file.
    """
    d = Path(directory)
    if not d.exists():
        from minegs.ingest.e57.exceptions import E57FileNotFoundError

        raise E57FileNotFoundError(d)
    if not d.is_dir():
        from minegs.ingest.e57.exceptions import E57NotAFileError

        raise E57NotAFileError(d)

    files = sorted(
        (p for p in d.iterdir() if p.is_file() and p.suffix.lower() in suffixes),
        key=lambda p: p.name,
    )
    assets: list[ImageAsset] = []
    for i, p in enumerate(files):
        width = height = None
        issues: list[str] = []
        try:
            from PIL import Image

            with Image.open(p) as im:  # reads the header, not the pixels
                width, height = im.size
        except Exception as e:
            issues.append(f"could not read image header: {e}")
        notes = [
            "external file: projection is not declared anywhere, so it is neither a panorama "
            "candidate nor mappable without an explicit mapping"
        ]
        assets.append(
            ImageAsset(
                image_id=image_id_for(i),
                source="external_file",
                source_index=i,
                name=p.name,
                representation="unknown",
                width=width,
                height=height,
                path=str(p),
                issues=issues,
                notes=notes,
            )
        )
    return assets
