"""Pinhole axis-convention calibration (§11–§12): measure it, do not assume it.

For each of the 24 axis-aligned proper rotations, the station's own scan is projected into
the station's own images with that rotation as ``R_e57cam_from_cam``; the rotation under
which the projected points land on pixels of their own colour wins. The scan must have RGB:
the comparison is direct and needs no feature matching, and the Phase 0C baseline has no
other scoring that can clear a threshold on its own — a scan without colour is refused
rather than scored by a weaker signal that could never pass.

One station is not evidence. The winner must be the same at ≥ 3 spatially separated
stations (early / middle / late along the survey's principal axis — chosen from the station
positions, never from hard-coded indices), and it must win by a margin. Fewer than three
eligible stations, disagreement between them, or a thin margin are recorded as
``insufficient`` / ``inconsistent`` / ``ambiguous`` and refused by the dataset builder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage

import minegs
from minegs.core.errors import ContractError
from minegs.core.frames import SE3
from minegs.core.pointcloud import read_ply
from minegs.core.provenance import git_commit
from minegs.dataset.cameras import (
    CameraCalibration,
    CameraConvention,
    CandidateScore,
    StationCalibration,
    axis_aligned_rotations,
    convention_label,
    image_pose_source_from_e57cam,
    pinhole_intrinsics,
)
from minegs.dataset.reprojection import project_points, rgb_residual
from minegs.dataset.staging_input import StagingTree

DEFAULT_MIN_MARGIN = 5.0  # RGB residual units (0-255) between best and runner-up
#: A calibrated convention needs this many spatially separated stations to agree (§12).
MIN_STATIONS = 3


@dataclass
class _Station:
    station_id: str
    scan_id: str
    image_ids: list[str]
    position_source: np.ndarray


def pinhole_stations(tree: StagingTree) -> dict[str, _Station]:
    """Stations with at least one resolved pinhole image, keyed by station id."""
    out: dict[str, _Station] = {}
    for im in tree.extraction.image_outputs:
        if im.representation != "pinhole":
            continue
        rec = tree.record(im.image_id)
        if not rec.is_resolved:
            continue
        scan_id = rec.scan_id
        assert scan_id is not None
        sid = rec.station_id or tree.station_of(scan_id)
        st = out.get(sid)
        if st is None:
            st = out[sid] = _Station(sid, scan_id, [], tree.scan_pose(scan_id).t)
        st.image_ids.append(im.image_id)
    for st in out.values():
        st.image_ids.sort()
    return out


def select_spread(positions: dict[str, np.ndarray], n: int) -> list[str]:
    """``n`` ids spread along the principal axis of the positions: first, ..., last.

    "Early / middle / late" by *position*, so a survey whose scan order wanders still gets
    stations from different places; ties resolve by id so the choice is deterministic.
    """
    ids = sorted(positions)
    if len(ids) <= n:
        return ids
    P = np.array([positions[i] for i in ids])
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    axis = vt[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    t = (P - c) @ axis
    order = np.lexsort((ids, t))  # by projection, then id
    picks = np.linspace(0, len(ids) - 1, n).round().astype(int)
    return [ids[order[k]] for k in picks]


def calibrate_camera_convention(
    tree: StagingTree,
    n_stations: int = MIN_STATIONS,
    max_points: int = 150_000,
    min_margin: float = DEFAULT_MIN_MARGIN,
    seed: int = 0,
) -> CameraCalibration:
    if n_stations < MIN_STATIONS:
        raise ContractError(
            f"calibration samples {n_stations} station(s); a calibrated convention needs at "
            f"least {MIN_STATIONS} spatially separated stations to agree (§12). A convention "
            "that holds at one station and breaks at the next is not a convention."
        )
    stations = pinhole_stations(tree)
    if not stations:
        raise ContractError(
            f"{tree.root}: no resolved pinhole images. Calibration needs pinhole images mapped "
            "to their scans (see pano_mapping.json); a spherical tree does not use this step."
        )
    chosen = select_spread({k: v.position_source for k, v in stations.items()}, n_stations)
    notes: list[str] = []

    candidates = axis_aligned_rotations()
    labels = [convention_label(R) for R in candidates]
    rng = np.random.default_rng(seed)
    per_station: list[StationCalibration] = []
    agg = np.zeros(len(candidates))
    agg_n = np.zeros(len(candidates))
    input_hashes: dict[str, str] = {}
    sampled_images: list[str] = []

    for sid in chosen:
        st = stations[sid]
        # the digest is verified against the extractor's record before the bytes are used
        input_hashes[f"scans/{st.scan_id}.ply"] = tree.verify_scan(st.scan_id)
        cloud = read_ply(tree.scan_ply(st.scan_id))
        if cloud.frame != "SOURCE":
            raise ContractError(
                f"{st.scan_id}: scan cloud is in frame {cloud.frame!r}, expected SOURCE"
            )
        if cloud.rgb is None:
            raise ContractError(
                f"{st.scan_id}: scan has no RGB. Automatic axis-convention calibration compares "
                "point colour with pixel colour and has no other scoring that can pass on its "
                "own; supply the convention explicitly (camera.R_e57cam_from_cam) and take "
                "responsibility for it, or use a scan with colour."
            )
        if len(cloud) > max_points:
            cloud = cloud.select(np.sort(rng.choice(len(cloud), max_points, replace=False)))
        st_sum = np.zeros(len(candidates))
        st_n = np.zeros(len(candidates))
        for image_id in st.image_ids:
            asset = tree.asset(image_id)
            intr = pinhole_intrinsics(asset)
            T_source_from_e57cam = image_pose_source_from_e57cam(asset)
            input_hashes[f"images/{tree.image_path(image_id).name}"] = tree.verify_image(image_id)
            img = np.asarray(PILImage.open(tree.image_path(image_id)).convert("RGB"))
            if img.shape[0] != intr.height or img.shape[1] != intr.width:
                raise ContractError(
                    f"{image_id}: file is {img.shape[1]}x{img.shape[0]} but the E57 declares "
                    f"{intr.width}x{intr.height}"
                )
            sampled_images.append(image_id)
            K = intr.K()
            for ci, R in enumerate(candidates):
                T_source_from_cam = T_source_from_e57cam @ SE3(R, np.zeros(3))
                proj = project_points(
                    K, T_source_from_cam.inverse(), cloud.xyz, intr.width, intr.height
                )
                res, n = rgb_residual(img, proj, cloud.rgb)
                if n:
                    st_sum[ci] += res * n
                    st_n[ci] += n
        agg += st_sum
        agg_n += st_n
        scores = _scores(labels, st_sum, st_n)
        best, runner, b_s, r_s = _rank(scores)
        per_station.append(
            StationCalibration(
                station_id=sid,
                scan_id=st.scan_id,
                image_ids=list(st.image_ids),
                best_label=best,
                runner_up_label=runner,
                best_score=b_s,
                runner_up_score=r_s,
                margin=None if r_s is None else abs(r_s - b_s),
                candidates=scores,
            )
        )

    overall = _scores(labels, agg, agg_n)
    best, runner, b_s, r_s = _rank(overall)
    margin = None if r_s is None else abs(r_s - b_s)
    consistent = all(s.best_label == best for s in per_station)
    status: str
    if len(chosen) < MIN_STATIONS:
        status = "insufficient"
        notes.append(
            f"only {len(chosen)} eligible station(s) ({chosen}); a calibrated convention needs "
            f">= {MIN_STATIONS} spatially separated stations that agree (§12)"
        )
    elif not consistent:
        status = "inconsistent"
        notes.append(
            "stations disagree on the best convention: "
            + ", ".join(f"{s.station_id}->{s.best_label}" for s in per_station)
        )
    elif margin is None or margin < min_margin or not np.isfinite(b_s):
        status = "ambiguous"
        notes.append(
            f"best {best} beats runner-up {runner} by {margin} (< {min_margin}); the evidence "
            "does not single out one convention"
        )
    else:
        status = "selected"
    R_best = candidates[labels.index(best)]
    convention = (
        CameraConvention(
            R_e57cam_from_cam=R_best.tolist(),
            label=best,
            source="calibrated",
            origin="minegs dataset calibrate-camera",
        )
        if status == "selected"
        else None
    )
    return CameraCalibration(
        status=status,
        convention=convention,
        scoring="rgb_residual",
        candidates=overall,
        best_score=b_s,
        runner_up_score=r_s,
        margin=margin,
        min_margin=min_margin,
        stations=per_station,
        sampled_station_ids=list(chosen),
        sampled_image_ids=sampled_images,
        notes=notes,
        source_sha256=tree.source_sha256,
        input_hashes={**{a.path: a.sha256 for a in tree.artifact_assets}, **input_hashes},
        tool_version=minegs.__version__,
        provenance={"git_commit": git_commit(), "seed": seed, "max_points": max_points},
    )


def _scores(labels, total, n) -> list[CandidateScore]:
    return [
        CandidateScore(
            label=label,
            rgb_residual=float(total[i] / n[i]) if n[i] else None,
            n_points=int(n[i]),
        )
        for i, label in enumerate(labels)
    ]


def _rank(scores: list[CandidateScore]):
    """(best label, runner-up label, best score, runner-up score); lower residual wins."""
    vals = sorted(
        (s.rgb_residual if s.rgb_residual is not None else float("inf"), s.label) for s in scores
    )
    best_s, best = vals[0]
    runner_s, runner = vals[1] if len(vals) > 1 else (None, None)
    return best, runner, float(best_s), (None if runner_s is None else float(runner_s))
