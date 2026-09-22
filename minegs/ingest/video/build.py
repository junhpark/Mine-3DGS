"""Building a frame set: extraction, selection, 360 crops, masks (§Phase 3 C1).

One function assembles the whole thing — ``build_frameset`` — and the layout it writes is the
point of the module:

    <frameset_dir>/
      frameset.json     the record (minegs/ingest/video/models.py)
      frames/           what came out of the video, before selection
      images/           *exactly* what SfM is given
      masks/            optional, one per image, named the way COLMAP looks for them

``images/`` is not a filtered view of ``frames/`` that a caller is trusted to respect; it is a
separate directory holding only the frames selection kept, or their crops. The SfM backend is
pointed at it. That is what makes "a rejected frame cannot reach the reconstruction" a
property of the layout rather than of everyone remembering to pass the right flag.

The extractor is a seam. On a machine without ffmpeg the frames can be produced by copying an
existing directory, and the record says so (``real_execution: false``). There is no CLI flag
for it: substituting the decoder is something a test does to exercise the rest of the chain,
not something an operator should be able to do to a survey.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from minegs.core.errors import ContractError
from minegs.core.provenance import make_id, sha256_file, source_asset, stamp
from minegs.ingest.common.equirect import CropView, RingCropSpec, crop_equirect
from minegs.ingest.common.geometry import PanoConvention
from minegs.ingest.video.models import (
    FRAMESET_FILE,
    # Matched case-insensitively. The old selector globbed `*.png` and `*.jpg` literally and
    # non-recursively, so a `.jpeg` from a phone and every crop under a view directory vanished
    # without a word. Shared with the record's own check, which enumerates this same set.
    IMAGE_SUFFIXES,
    CropRecord,
    ExtractionRecord,
    FrameDecision,
    FrameSetKind,
    FrameSetRecord,
    ImageEntry,
    MaskRecord,
    SelectionRecord,
    check_frameset,
    expected_mask_name,
    images_digest,
)

FRAMES_DIRNAME = "frames"
IMAGES_DIRNAME = "images"
MASKS_DIRNAME = "masks"

_OWNED = frozenset({FRAMESET_FILE, FRAMES_DIRNAME, IMAGES_DIRNAME, MASKS_DIRNAME})

#: Signature of the frame-extraction seam: (video, out_dir, settings) -> the frames written.
Extractor = Callable[[Path, Path, dict[str, Any]], list[Path]]


def find_images(root: str | Path) -> list[Path]:
    """Every image under *root*, recursively, sorted by path relative to it.

    Recursive because a 360 crop set lives in per-view subdirectories, and sorted so that a
    sequential matcher sees frames in a stable order.
    """
    root = Path(root)
    if not root.is_dir():
        raise ContractError(f"{root}: not a directory")
    found = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(found, key=lambda p: p.relative_to(root).as_posix())


def prepare_frameset_dir(out_dir: str | Path, overwrite: bool) -> None:
    """Refuse to write into a directory that is not ours, or is ours and already occupied.

    Same rule as the E57 extractor (§0B): a foreign file is refused whatever the flags say,
    because this never deletes anything it did not write; an occupied frame set of our own is
    refused unless the caller asked for a replacement, because two extractions mixed in one
    directory cannot be told apart afterwards.
    """
    out = Path(out_dir)
    if out.is_symlink():
        raise ContractError(
            f"{out} is a symlink. A frame set is published by writing a directory tree, so "
            "point this at the real directory."
        )
    if out.exists() and not out.is_dir():
        raise ContractError(f"{out} exists and is not a directory")
    if not out.exists() or not any(out.iterdir()):
        return
    foreign = sorted(e.name for e in out.iterdir() if e.name not in _OWNED or e.is_symlink())
    if foreign:
        raise ContractError(
            f"{out} holds files this command did not write ({foreign[:5]}). It replaces a "
            "whole frame set and will not delete anything else. Use a directory of its own."
        )
    if not overwrite:
        raise ContractError(
            f"{out} already holds a frame set. Writing into it would mix two extractions whose "
            "frames cannot be told apart afterwards. Use a new directory, or pass --overwrite."
        )
    for name in sorted(_OWNED):
        path = out / name
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()


def _ffmpeg_extract(video: Path, out_dir: Path, settings: dict[str, Any]) -> list[Path]:
    from minegs.ingest.video.frames import extract_frames

    return extract_frames(video, out_dir, **settings)


def copy_extractor(source_dir: str | Path) -> Extractor:
    """A substitution: take the frames from a directory that already holds them.

    For the structural gate, which has no ffmpeg and no video. Recorded as
    ``real_execution: false`` by :func:`extract_frames_recorded`, so nothing downstream can
    mistake the result for evidence that a real decode happened.
    """
    src = Path(source_dir)

    def _copy(_video: Path, out_dir: Path, _settings: dict[str, Any]) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        for image in find_images(src):
            target = out_dir / image.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target)
        return find_images(out_dir)

    return _copy


def extract_frames_recorded(
    video: str | Path | None,
    frames_dir: Path,
    *,
    fps: float | None = None,
    scale_width: int | None = None,
    start_s: float | None = None,
    duration_s: float | None = None,
    pattern: str = "v_%06d.png",
    extractor: Extractor | None = None,
) -> tuple[list[Path], ExtractionRecord]:
    """Produce the frames and say exactly how, including whether the real decoder ran."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    if video is None:
        frames = find_images(frames_dir)
        return frames, ExtractionRecord(method="none", real_execution=True)

    settings: dict[str, Any] = {"pattern": pattern}
    if fps is not None:
        settings["fps"] = fps
    if scale_width is not None:
        settings["scale_width"] = scale_width
    if start_s is not None:
        settings["start_s"] = start_s
    if duration_s is not None:
        settings["duration_s"] = duration_s

    from minegs.ingest.video.frames import extract_command

    argv = extract_command(video, frames_dir, **settings)
    if extractor is None:
        frames = _ffmpeg_extract(Path(video), frames_dir, settings)
        method, real = "ffmpeg", True
        from minegs.core.provenance import _cli_version

        tool = _cli_version("ffmpeg")
    else:
        frames = extractor(Path(video), frames_dir, settings)
        method, real, tool = "copy", False, None
    if not frames:
        raise ContractError(
            f"extraction produced no frames in {frames_dir}. A frame set with nothing in it is "
            "not a smaller reconstruction, it is no reconstruction; check the input and the "
            "extraction settings."
        )
    return frames, ExtractionRecord(
        method=method,  # type: ignore[arg-type]
        argv=[str(a) for a in argv],
        fps=fps,
        scale_width=scale_width,
        start_s=start_s,
        duration_s=duration_s,
        pattern=pattern,
        tool_version=tool,
        real_execution=real,
    )


def select_frames_recorded(
    frames: list[Path],
    frames_root: Path,
    *,
    blur_threshold: float,
    hamming_threshold: int,
    max_frames: int | None,
) -> SelectionRecord:
    """Score the frames and record every decision, kept or not.

    Refuses an empty input rather than reporting that it kept none of none. "kept 0 / 0" with a
    success exit is how a mistyped directory used to look exactly like a clean run.
    """
    if not frames:
        raise ContractError(
            f"no images found under {frames_root} (looked for {sorted(IMAGE_SUFFIXES)}, "
            "recursively). Nothing to select from."
        )
    from minegs.ingest.video.dedup_blur import select_frames

    decisions = select_frames(frames, blur_threshold, hamming_threshold, max_frames)
    recorded = [
        FrameDecision(
            name=Path(d.path).relative_to(frames_root).as_posix(),
            blur=d.blur,
            phash=f"{d.hash:016x}",
            keep=d.keep,
            reason=d.reason,
        )
        for d in decisions
    ]
    kept = sum(1 for d in recorded if d.keep)
    if kept == 0:
        raise ContractError(
            f"selection kept none of {len(recorded)} frames (blur < {blur_threshold}, or all "
            "near-duplicates). Loosen the thresholds rather than reconstructing from nothing."
        )
    return SelectionRecord(
        blur_threshold=blur_threshold,
        hamming_threshold=hamming_threshold,
        max_frames=max_frames,
        considered=len(recorded),
        kept=kept,
        decisions=recorded,
    )


def _read_image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _write_image(arr: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def cut_ring_crops(
    kept: list[Path],
    frames_root: Path,
    images_dir: Path,
    spec: RingCropSpec,
    convention: PanoConvention,
) -> list[CropRecord]:
    """Cut every kept panorama into the ring's views, recording each crop's own geometry.

    The crop's yaw, pitch and intrinsics are written next to its digest. A path like
    ``p0y03/v_000007_p0y03.png`` is convenient and is not evidence: the rig is derived from
    these records, and a crop whose bytes change stops matching the orientation it was cut at.
    """
    views = spec.crops()
    K = spec.K()
    out: list[CropRecord] = []
    for pano_path in kept:
        pano = _read_image(pano_path)
        parent = pano_path.relative_to(frames_root).as_posix()
        stem = Path(parent).stem
        for view in views:
            arr = crop_equirect(pano, view, convention)
            # The view is in the file name and not only in the directory. Downstream, a depth
            # map is named after an image's *stem* (§Phase 1B), so `p0y00/v_07.png` and
            # `p0y01/v_07.png` would both want `v_07.npy`: one render would overwrite the
            # other and both views would then back-project the same map. Repeating the view
            # here is redundant to a reader and is the thing that keeps the stems distinct.
            name = f"{view.name}/{stem}_{view.name}.png"
            target = images_dir / name
            _write_image(arr, target)
            out.append(
                CropRecord(
                    name=name,
                    parent_frame=parent,
                    view=view.name,
                    yaw_deg=view.yaw_deg,
                    pitch_deg=view.pitch_deg,
                    width=spec.width,
                    height=spec.height,
                    fx=float(K[0, 0]),
                    fy=float(K[1, 1]),
                    cx=float(K[0, 2]),
                    cy=float(K[1, 2]),
                    sha256=sha256_file(target),
                )
            )
    return out


def _copy_kept(kept: list[Path], frames_root: Path, images_dir: Path) -> list[str]:
    names: list[str] = []
    for path in kept:
        rel = path.relative_to(frames_root).as_posix()
        target = images_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        names.append(rel)
    return names


def write_masks(
    frameset_dir: Path,
    images: list[str],
    crops: list[CropRecord],
    spec: RingCropSpec | None,
    *,
    nadir_el_deg: float | None = None,
    boxes: list[tuple[int, int, int, int]] | None = None,
) -> list[MaskRecord]:
    """Write one mask per image, named the way COLMAP looks for it, and record the binding.

    COLMAP ignores a mask whose name does not match an image, without complaining, so the
    names are produced here from the image list rather than assembled by hand somewhere else.
    """
    if nadir_el_deg is None and not boxes:
        return []
    from minegs.ingest.video.masks import box_mask, nadir_mask_for_crop, write_mask

    masks_dir = frameset_dir / MASKS_DIRNAME
    by_name = {c.name: c for c in crops}
    out: list[MaskRecord] = []
    for name in images:
        crop = by_name.get(name)
        if nadir_el_deg is not None and crop is not None and spec is not None:
            view = CropView(crop.view, crop.yaw_deg, crop.pitch_deg, spec)
            arr = nadir_mask_for_crop(view, nadir_el_deg)
            method: Any = "nadir_crop"
            params: dict[str, Any] = {"nadir_el_deg": nadir_el_deg}
        elif boxes:
            sample = _read_image(frameset_dir / IMAGES_DIRNAME / name)
            arr = box_mask(sample.shape[1], sample.shape[0], boxes)
            method, params = "box", {"boxes": [list(b) for b in boxes]}
        else:
            continue
        mask_name = expected_mask_name(name)
        target = masks_dir / mask_name
        write_mask(arr, name, masks_dir)
        out.append(
            MaskRecord(
                image=name,
                mask_file=mask_name,
                method=method,
                parameters=params,
                sha256=sha256_file(target),
            )
        )
    return out


def build_frameset(
    out_dir: str | Path,
    *,
    kind: FrameSetKind,
    video: str | Path | None = None,
    image_dir: str | Path | None = None,
    fps: float | None = None,
    scale_width: int | None = None,
    start_s: float | None = None,
    duration_s: float | None = None,
    pattern: str = "v_%06d.png",
    blur_threshold: float = 60.0,
    hamming_threshold: int = 6,
    max_frames: int | None = None,
    ring: RingCropSpec | None = None,
    pano_convention: PanoConvention | None = None,
    nadir_el_deg: float | None = None,
    mask_boxes: list[tuple[int, int, int, int]] | None = None,
    overwrite: bool = False,
    extractor: Extractor | None = None,
) -> tuple[FrameSetRecord, Path]:
    """Extract, select, crop, mask, and write the artifact that says what SfM will be fed."""
    out = Path(out_dir)
    prepare_frameset_dir(out, overwrite)
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / FRAMES_DIRNAME
    images_dir = out / IMAGES_DIRNAME

    if kind == "video360" and ring is None:
        raise ContractError("a 360 frame set needs a ring crop spec")
    if kind == "video360" and pano_convention is None:
        raise ContractError(
            "a 360 video frame set needs an explicit panorama convention. The library default "
            "names E57 as its source, which is a statement about a scanner's panorama and not "
            "about this camera's; leaving it implicit would make an unmeasured azimuth sign "
            "and elevation flip into a fixed geometric constraint on the reconstruction."
        )
    if pano_convention is not None and pano_convention.source == "E57Embedded":
        raise ContractError(
            "the panorama convention for this frame set claims source 'E57Embedded'. That is "
            "the scanner path's convention, carried by the file a scanner wrote; a camera's "
            "has to be calibrated or configured and named as what it is."
        )

    if image_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for image in find_images(image_dir):
            target = frames_dir / image.relative_to(Path(image_dir))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target)
        frames, extraction = (
            find_images(frames_dir),
            ExtractionRecord(method="none", real_execution=True),
        )
        if not frames:
            raise ContractError(f"no images found under {image_dir}")
    else:
        frames, extraction = extract_frames_recorded(
            video,
            frames_dir,
            fps=fps,
            scale_width=scale_width,
            start_s=start_s,
            duration_s=duration_s,
            pattern=pattern,
            extractor=extractor,
        )

    selection = select_frames_recorded(
        frames,
        frames_dir,
        blur_threshold=blur_threshold,
        hamming_threshold=hamming_threshold,
        max_frames=max_frames,
    )
    kept = [frames_dir / d.name for d in selection.decisions if d.keep]

    images_dir.mkdir(parents=True, exist_ok=True)
    if kind == "video360":
        assert ring is not None and pano_convention is not None
        crops = cut_ring_crops(kept, frames_dir, images_dir, ring, pano_convention)
        names = [c.name for c in crops]
    else:
        crops = []
        names = _copy_kept(kept, frames_dir, images_dir)

    masks = write_masks(out, names, crops, ring, nadir_el_deg=nadir_el_deg, boxes=mask_boxes)
    entries = [ImageEntry(name=n, sha256=sha256_file(images_dir / n)) for n in names]
    record = FrameSetRecord(
        frameset_id=make_id("frameset"),
        kind=kind,
        source_name=Path(video).name if video is not None else None,
        source_sha256=sha256_file(video) if video is not None else None,
        extraction=extraction,
        selection=selection,
        pano_convention=pano_convention,
        crop_spec=ring.model_dump(mode="json") if ring is not None else None,
        crops=crops,
        masks=masks,
        images_dir=IMAGES_DIRNAME,
        masks_dir=MASKS_DIRNAME if masks else None,
        images=entries,
        images_sha256=images_digest(entries),
        provenance=stamp(
            {"kind": kind, "blur": blur_threshold, "hamming": hamming_threshold},
            assets=[source_asset(video)] if video is not None else None,
        ),
    )
    check_frameset(record, out)
    (out / FRAMESET_FILE).write_text(record.model_dump_json(indent=2))
    return record, out


__all__ = [
    "FRAMES_DIRNAME",
    "IMAGES_DIRNAME",
    "IMAGE_SUFFIXES",
    "MASKS_DIRNAME",
    "build_frameset",
    "copy_extractor",
    "cut_ring_crops",
    "extract_frames_recorded",
    "find_images",
    "prepare_frameset_dir",
    "select_frames_recorded",
    "write_masks",
]
