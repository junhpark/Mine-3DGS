"""The image-only Golden Gate (§Phase 3 C3) — one physical space, without a scanner in it.

The TLS gate asks whether each station's own scan projects into that station's images. An
image-only dataset has no station scans, so that gate does not apply to it and
``run_golden_gate`` refuses it outright. What must not happen is for the refusal to be worked
around, or for the absence of a gate to be reported as a pass.

This asks the same question with the evidence the image-only path does have:

* **Its own structure.** Every camera is asked how much of the reconstruction it can see. A
  model whose poses and points do not occupy one space has cameras that see almost nothing,
  and it shows here before any evaluation runs.
* **Its registration.** The measured transform, its residuals, the support it was fitted
  against, and whether that support permits a claim at all.
* **The reference, projected in.** The TLS cloud is put through the registered cameras and
  drawn over the images. That is the picture a person looks at, and using the TLS this way is
  evaluation evidence: it is never the initialisation, and it enters after the reconstruction
  is finished.

``real_data_validation_status`` is ``pending_human_inspection`` and nothing here writes
anything else into it. The gate kind is recorded as ``image_sfm_registered`` so that no reader
can mistake this for the TLS gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import minegs
from minegs.core.errors import ContractError
from minegs.core.frames import SE3, quat_to_rotmat
from minegs.core.manifest import Manifest
from minegs.core.pointcloud import read_ply
from minegs.core.provenance import git_commit, sha256_file
from minegs.dataset.from_sfm import PROVENANCE_DIR, check_image_only_dataset
from minegs.ingest.common import colmap_io

GATE_KIND = "image_sfm_registered"
GATE_FILE = "golden_gate_sfm.json"

HUMAN_CHECKLIST = (
    "the TLS overlay follows the walls in every sampled view, not just near the camera",
    "the reconstruction's own points sit on surfaces, not in the air",
    "the registered scale looks right: a 2 m roof is 2 m in the overlay",
    "no view is aligned only near the registration support and drifting away from it",
    "Viser: cameras run along the drift, reconstruction and TLS coincide",
)


def _cam_from_world(image: colmap_io.Image) -> SE3:
    return SE3(quat_to_rotmat(image.qvec), image.tvec)


def _visibility(model: colmap_io.ColmapModel) -> dict[str, dict[str, float]]:
    """How much of the reconstruction each camera can actually see."""
    xyz = model.points_xyz()
    out: dict[str, dict[str, float]] = {}
    for im in model.images.values():
        cam = model.cameras[im.camera_id]
        uv, depth = colmap_io.project(cam.K(), _cam_from_world(im), xyz)
        vis = colmap_io.visible_mask(uv, depth, cam.width, cam.height)
        n = int(vis.sum())
        out[im.name] = {
            "visible_points": n,
            "visible_fraction": float(n / max(len(xyz), 1)),
            "median_depth_m": float(np.median(depth[vis])) if n else float("nan"),
        }
    return out


def _overlay(
    image_path: Path, cam: colmap_io.Camera, pose: SE3, ref_xyz: np.ndarray, out_path: Path
) -> dict[str, Any]:
    """Draw the registered reference cloud over one image, and say how much of it landed."""
    from PIL import Image

    from minegs.viz.overlay import colormap_turbo_like, save_overlay

    with Image.open(image_path) as im:
        arr = np.asarray(im.convert("RGB")).copy()
    uv, depth = colmap_io.project(cam.K(), pose, ref_xyz)
    vis = colmap_io.visible_mask(uv, depth, cam.width, cam.height)
    n = int(vis.sum())
    if n:
        d = depth[vis]
        span = float(d.max() - d.min())
        colours = (colormap_turbo_like((d - d.min()) / max(span, 1e-6)) * 255).astype(np.uint8)
        cols = uv[vis, 0].astype(int).clip(0, arr.shape[1] - 1)
        rows = uv[vis, 1].astype(int).clip(0, arr.shape[0] - 1)
        arr[rows, cols] = colours
    save_overlay(out_path, arr)
    return {
        "image": image_path.name,
        "overlay": out_path.name,
        "reference_points_projected": n,
        "reference_fraction": float(n / max(len(ref_xyz), 1)),
    }


def run_image_only_gate(
    dataset_dir: str | Path,
    out_dir: str | Path,
    *,
    reference_ply: str | Path | None = None,
    n_views: int = 3,
    max_reference_points: int = 200_000,
    min_visible_fraction: float = 0.01,
    raise_on_fail: bool = True,
) -> dict[str, Any]:
    """Assemble the image-only evidence and write the report."""
    ds, out = Path(dataset_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load_dataset(ds)
    if manifest.source not in ("video", "video360"):
        raise ContractError(
            f"{ds} has source {manifest.source!r}. This is the image-only gate; a TLS dataset "
            "goes through `run_golden_gate`, which has the station scans this one does not."
        )
    check_image_only_dataset(ds)

    model = colmap_io.read_model(ds / "sparse" / "0")
    if not model.images:
        raise ContractError(f"{ds}: sparse/0 registers no images")
    visibility = _visibility(model)
    problems: list[str] = []
    blind = sorted(
        name for name, v in visibility.items() if v["visible_fraction"] < min_visible_fraction
    )
    if blind:
        problems.append(
            f"{len(blind)} of {len(visibility)} cameras see less than "
            f"{min_visible_fraction:.1%} of the reconstruction (e.g. {blind[:3]}): the poses "
            "and the points may not be in one space"
        )

    prov = json.loads((ds / PROVENANCE_DIR / "init_provenance.json").read_text())
    reg = manifest.registration
    assert reg is not None  # check_image_only_dataset guarantees it
    overlays: list[dict[str, Any]] = []
    reference_info: dict[str, Any] | None = None
    if reference_ply is not None:
        ref = read_ply(reference_ply)
        if ref.frame != "TLS_GLOBAL":
            raise ContractError(
                f"{reference_ply} declares frame {ref.frame}; the overlay puts the reference "
                "through cameras that live in this dataset's metric frame, so it has to be the "
                "survey's own"
            )
        ref_local = ref.xyz - np.asarray(manifest.T_tls_from_local.t)
        if len(ref_local) > max_reference_points:
            idx = np.random.default_rng(0).choice(
                len(ref_local), max_reference_points, replace=False
            )
            ref_local = ref_local[np.sort(idx)]
        names = sorted(visibility)
        picks = np.linspace(0, len(names) - 1, min(n_views, len(names))).round().astype(int)
        by_name = model.image_by_name()
        (out / "overlays").mkdir(exist_ok=True)
        for k in dict.fromkeys(int(i) for i in picks):
            name = names[k]
            im = by_name[name]
            info = _overlay(
                ds / "images" / name,
                model.cameras[im.camera_id],
                _cam_from_world(im),
                ref_local,
                out / "overlays" / f"{Path(name).stem}_tls.png",
            )
            overlays.append(info)
        landed = [o["reference_points_projected"] for o in overlays]
        if overlays and max(landed) == 0:
            problems.append(
                "no reference point projects into any sampled view: the registered "
                "reconstruction and the reference are not in the same place"
            )
        reference_info = {
            "file": Path(reference_ply).name,
            "sha256": sha256_file(reference_ply),
            "points": len(ref.xyz),
        }

    report: dict[str, Any] = {
        "gate_kind": GATE_KIND,
        "dataset_id": manifest.dataset_id,
        "source": manifest.source,
        "minegs_version": minegs.__version__,
        "git_commit": git_commit(),
        "reconstruction": {
            "sfm_id": prov.get("sfm_id"),
            "sfm_model_sha256": prov.get("sfm_model_sha256"),
            "registered_images": len(model.images),
            "points": len(model.points3D),
            "visibility": visibility,
        },
        "registration": {
            "registration_id": reg.registration_id,
            "basis": reg.basis,
            "scale": reg.scale,
            "rmse_m": reg.rmse_m,
            "inlier_ratio": reg.inlier_ratio,
            "support_ranges_m": reg.support_ranges_m,
            "claim_allowed": reg.claim_allowed,
            "claim_refusals": list(reg.claim_refusals),
        },
        "reference": reference_info,
        "overlays": overlays,
        "execution": {
            "real_sfm_execution": prov.get("real_sfm_execution"),
            "frame_extraction_real": prov.get("frame_extraction_real"),
        },
        "problems": problems,
        "structural_result": "pass" if not problems else "fail",
        # Never written by this code. A person looks at the overlays and the Viser scene, and
        # an automatic score is not that (§24, §38).
        "real_data_validation_status": "pending_human_inspection",
        "human_inspection_checklist": list(HUMAN_CHECKLIST),
    }
    (out / GATE_FILE).write_text(json.dumps(report, indent=2) + "\n")
    if problems and raise_on_fail:
        raise ContractError(
            f"image-only golden gate structural_result=fail ({len(problems)} problem(s); see "
            f"{out / GATE_FILE}): {problems[0]}"
        )
    return report


__all__ = ["GATE_FILE", "GATE_KIND", "HUMAN_CHECKLIST", "run_image_only_gate"]
