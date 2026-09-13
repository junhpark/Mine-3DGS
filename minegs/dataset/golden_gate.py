"""Golden Gate (§24–§29): show, on real geometry, that TLS points, camera projections and
``init_points.ply`` occupy one physical space.

The automated part projects each sampled station's own scan into each of its images and
writes overlays (depth-coloured and, where the scan has colour, TLS-RGB-coloured) so that an
axis flip, a 90° or 180° turn, a mirrored image, a displaced centre or a wrong focal length
is visible at a glance. For the pinhole path it also re-scores all 24 axis conventions on the
sampled stations and checks that the one the dataset was built with is the one the evidence
picks, by a margin, at every station.

``structural_result`` is what the numbers say. ``real_data_validation_status`` is always
``pending_human_inspection``: the Phase 0C gate (G2) includes a person looking at the
overlays and at the Viser scene, and no score here stands in for that (§24, §38).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

import minegs
from minegs.core.centerline import Centerline
from minegs.core.errors import ContractError
from minegs.core.frames import SE3
from minegs.core.manifest import Manifest
from minegs.core.pointcloud import PointCloud, read_ply, write_ply
from minegs.core.provenance import git_commit, sha256_file
from minegs.dataset.calibrate import DEFAULT_MIN_MARGIN, select_spread
from minegs.dataset.cameras import (
    CameraConvention,
    axis_aligned_rotations,
    convention_label,
    image_pose_source_from_e57cam,
)
from minegs.dataset.frames import ROUNDTRIP_TOL_M, validate_rigid
from minegs.dataset.materialize import (
    BUILD_CONFIG_FILE,
    CAMERA_EXTENT_MARGIN_M,
    CONVENTION_FILE,
    holdout_mask,
)
from minegs.dataset.reprojection import (
    depth_colours,
    project_points,
    render_overlay,
    rgb_residual,
    save_image,
)
from minegs.dataset.staging_input import load_staging
from minegs.ingest.common import colmap_io

REPORT_FILE = "report.json"
TLS_SAMPLE_FILE = "tls_local_metric.ply"
HUMAN_CHECKLIST = [
    "wall edges and tunnel boundary follow the projected points",
    "pipes, cables, signs and objects coincide with their points",
    "ceiling is up and floor is down in every face",
    "scanner/camera position sits on the actual station",
    "left/right are consistent across faces and stations",
    "TLS, init points and frustums coincide in Viser; frustums look along the tunnel",
]


def run_golden_gate(
    dataset_dir: str | Path,
    staging_dir: str | Path,
    out_dir: str | Path,
    n_stations: int = 3,
    max_points: int = 150_000,
    min_margin: float = DEFAULT_MIN_MARGIN,
    seed: int = 0,
) -> dict[str, Any]:
    ds = Path(dataset_dir)
    out = Path(out_dir)
    manifest = Manifest.load_dataset(ds)
    if not (ds / BUILD_CONFIG_FILE).is_file():
        raise ContractError(
            f"{ds}: no {BUILD_CONFIG_FILE}; the Golden Gate needs a dataset built by `minegs dataset from-e57`"
        )
    resolved = json.loads((ds / BUILD_CONFIG_FILE).read_text())
    frames = resolved["frames"]
    T_tls_from_source = validate_rigid(frames["T_tls_from_source"], "T_tls_from_source")
    T_tls_from_local = validate_rigid(frames["T_tls_from_local"], "T_tls_from_local")
    if not T_tls_from_local.allclose(manifest.T_tls_from_local):
        raise ContractError("build_config.json and manifest.json disagree about T_tls_from_local")
    T_local_from_source = T_tls_from_local.inverse() @ T_tls_from_source
    tree = load_staging(staging_dir)
    if tree.source_sha256 != _source_sha_of(manifest):
        raise ContractError(
            "the staging tree's source digest is not among the dataset's source assets; this "
            "is not the tree the dataset was built from"
        )
    model = colmap_io.read_model(ds / "sparse" / "0")
    by_name = model.image_by_name()
    convention: CameraConvention | None = None
    if resolved.get("camera_convention"):
        convention = CameraConvention.model_validate(resolved["camera_convention"])
    pinhole = resolved["config"]["camera"]["mode"] == "e57_pinhole"

    # ---- stations: early / middle / late by chainage, else by position
    groups = {g: cg for g, cg in manifest.capture_groups.items() if cg.type == "tls_station"}
    if not groups:
        raise ContractError("dataset has no tls_station groups")
    positions = {
        g: np.mean([by_name[m].center for m in cg.members], axis=0) for g, cg in groups.items()
    }
    if all(cg.chainage_m is not None for cg in groups.values()):
        order = sorted(groups, key=lambda g: (groups[g].chainage_m, g))
        picks = np.linspace(0, len(order) - 1, min(n_stations, len(order))).round().astype(int)
        chosen = [order[k] for k in picks]
    else:
        chosen = select_spread(positions, n_stations)

    rng = np.random.default_rng(seed)
    candidates = axis_aligned_rotations()
    labels = [convention_label(R) for R in candidates]
    problems: list[str] = []
    station_reports = []
    overlays: list[str] = []
    tls_parts: list[PointCloud] = []
    test_set = set(manifest.test_images())
    for g in chosen:
        scan_id = resolved["stations"][g]["scan_id"]
        cloud = read_ply(tree.scan_ply(scan_id))
        if cloud.frame != "SOURCE":
            raise ContractError(f"{scan_id}: expected a SOURCE-frame scan, got {cloud.frame}")
        if len(cloud) > max_points:
            cloud = cloud.select(np.sort(rng.choice(len(cloud), max_points, replace=False)))
        local = cloud.transformed(T_local_from_source, "LOCAL_METRIC")
        tls_parts.append(local)
        image_reports = []
        cand_rgb = np.zeros(len(candidates))
        cand_n = np.zeros(len(candidates))
        for name in groups[g].members:
            im = by_name[name]
            cam = model.cameras[im.camera_id]
            K = cam.K()
            img = np.asarray(PILImage.open(ds / "images" / name).convert("RGB"))
            proj = project_points(K, im.cam_from_world, local.xyz, cam.width, cam.height)
            rep: dict[str, Any] = {
                "image": name,
                "role": "test" if name in test_set else "train",
                "n_points": len(local),
                "n_in_front": int((proj.depth > 1e-3).sum()),
                "n_inside": proj.n_inside,
                "camera_centre_local": im.center.round(4).tolist(),
            }
            if proj.n_inside == 0:
                problems.append(f"{name}: no TLS point projects into the image")
            depth_png = out / "overlays" / f"{g}_{Path(name).stem}_depth.png"
            save_image(depth_png, render_overlay(img, proj, depth_colours(proj)))
            overlays.append(str(depth_png.relative_to(out)))
            if local.rgb is not None:
                res, _n = rgb_residual(img, proj, local.rgb)
                rep["rgb_residual"] = res
                rgb_png = out / "overlays" / f"{g}_{Path(name).stem}_tlsrgb.png"
                save_image(rgb_png, render_overlay(img, proj, local.rgb))
                overlays.append(str(rgb_png.relative_to(out)))
            if pinhole and local.rgb is not None:
                # re-score every convention from the E57 pose, independent of the dataset pose
                image_id = name.split("_", 1)[1].rsplit(".", 1)[0]
                asset = tree.asset(image_id)
                T_source_from_e57cam = image_pose_source_from_e57cam(asset)
                for ci, R in enumerate(candidates):
                    T_local_from_cam = (
                        T_local_from_source @ T_source_from_e57cam @ SE3(R, np.zeros(3))
                    )
                    p2 = project_points(
                        K, T_local_from_cam.inverse(), local.xyz, cam.width, cam.height
                    )
                    r2, n2 = rgb_residual(img, p2, local.rgb)
                    if n2:
                        cand_rgb[ci] += r2 * n2
                        cand_n[ci] += n2
            image_reports.append(rep)
        st_rep: dict[str, Any] = {
            "station_id": g,
            "scan_id": scan_id,
            "chainage_m": groups[g].chainage_m,
            "position_local": positions[g].round(4).tolist(),
            "images": image_reports,
        }
        if pinhole and cand_n.any():
            scores = {
                lab: float(cand_rgb[i] / cand_n[i]) for i, lab in enumerate(labels) if cand_n[i]
            }
            ranked = sorted(scores.items(), key=lambda kv: kv[1])
            best, runner = ranked[0], (ranked[1] if len(ranked) > 1 else (None, None))
            st_rep["convention_scores"] = scores
            st_rep["best_convention"] = best[0]
            st_rep["margin"] = None if runner[1] is None else runner[1] - best[1]
            if convention is not None and best[0] != convention.label:
                problems.append(f"{g}: evidence picks {best[0]}, dataset uses {convention.label}")
            elif st_rep["margin"] is not None and st_rep["margin"] < min_margin:
                problems.append(f"{g}: convention margin {st_rep['margin']:.2f} < {min_margin}")
        station_reports.append(st_rep)

    # ---- numerical checks (§27)
    init = read_ply(ds / manifest.initialization.file)
    if init.frame != "LOCAL_METRIC":
        problems.append(f"init_points frame is {init.frame}")
    tls_lo, tls_hi = (np.array(b) for b in resolved["tls_bounds_local_metric"])
    imin, imax = init.bounds()
    if np.any(imin < tls_lo - 1e-6) or np.any(imax > tls_hi + 1e-6):
        problems.append("init point bounds exceed the TLS bounds")
    centres = np.array([im.center for im in model.images.values()])
    if not np.all(np.isfinite(centres)):
        problems.append("non-finite camera centre")
    if np.any(centres < tls_lo - CAMERA_EXTENT_MARGIN_M) or np.any(
        centres > tls_hi + CAMERA_EXTENT_MARGIN_M
    ):
        problems.append("a camera centre lies outside the survey extent")
    sample = init.xyz[: min(len(init), 10_000)]
    back = T_tls_from_local.inverse().apply(T_tls_from_local.apply(sample))
    rt = float(np.max(np.abs(back - sample))) if len(sample) else 0.0
    if rt > ROUNDTRIP_TOL_M:
        problems.append(f"LOCAL -> TLS -> LOCAL round trip error {rt:.2e} m")
    if not np.allclose(T_tls_from_local.R, np.eye(3)):
        problems.append("T_tls_from_local carries a rotation; the baseline is translation-only")
    holdout_leak = None
    if manifest.split.geometry_holdout and manifest.centerline:
        cl = Centerline.from_csv(
            ds / manifest.centerline.file, manifest.centerline.frame, manifest.centerline.source
        )
        if manifest.centerline.frame == "TLS_GLOBAL":
            cl = cl.transformed(T_tls_from_local.inverse(), "LOCAL_METRIC")
        s, _ = cl.project(init.xyz)
        holdout_leak = int(holdout_mask(s, manifest.split.geometry_holdout.chainage_ranges_m).sum())
        if holdout_leak:
            problems.append(f"{holdout_leak} init points inside holdout ranges")

    # ---- Viser-ready TLS sample (LOCAL_METRIC)
    tls = tls_parts[0]
    for p in tls_parts[1:]:
        tls = tls.concat(p)
    tls = tls.subsample(max(1, max_points * 2), seed)
    write_ply(tls, out / TLS_SAMPLE_FILE)

    report = {
        "dataset_id": manifest.dataset_id,
        "dataset_dir": str(ds.resolve()),
        "staging_dir": str(Path(staging_dir).resolve()),
        "frames": frames,
        "T_tls_from_source": T_tls_from_source.to_list(),
        "T_tls_from_local": T_tls_from_local.to_list(),
        "camera_convention": convention.model_dump(mode="json") if convention else None,
        "sampled_stations": chosen,
        "sampled_images": [r["image"] for s in station_reports for r in s["images"]],
        "stations": station_reports,
        "coordinate_bounds_local_metric": {
            "tls": [tls_lo.tolist(), tls_hi.tolist()],
            "cameras": [centres.min(0).tolist(), centres.max(0).tolist()],
        },
        "init_point_bounds_local_metric": [imin.tolist(), imax.tolist()],
        "init_points": len(init),
        "init_points_in_holdout": holdout_leak,
        "roundtrip_error_m": rt,
        "overlays": overlays,
        "tls_sample": TLS_SAMPLE_FILE,
        "viser": f"minegs viz view {ds} --tls {out / TLS_SAMPLE_FILE}",
        "input_hashes": {
            "manifest.json": sha256_file(ds / "manifest.json"),
            BUILD_CONFIG_FILE: sha256_file(ds / BUILD_CONFIG_FILE),
            **(
                {CONVENTION_FILE: sha256_file(ds / CONVENTION_FILE)}
                if (ds / CONVENTION_FILE).is_file()
                else {}
            ),
            "init_points.ply": sha256_file(ds / manifest.initialization.file),
            "source_e57": tree.source_sha256,
        },
        "config_hash": manifest.provenance.config_hash,
        "structural_result": "pass" if not problems else "fail",
        "structural_problems": problems,
        "real_data_validation_status": "pending_human_inspection",
        "human_inspection_checklist": HUMAN_CHECKLIST,
        "tool_version": minegs.__version__,
        "git_commit": git_commit(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _source_sha_of(manifest: Manifest) -> str | None:
    for a in manifest.provenance.source_assets:
        if a.path.endswith(".e57"):
            return a.sha256
    return None
