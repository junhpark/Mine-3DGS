"""Phase 0C §32 — camera contract: E57 pinhole → COLMAP, convention, mapping refusals."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from minegs.core.errors import ContractError
from minegs.core.frames import SE3, rot_x, rot_z, rotmat_to_quat
from minegs.dataset.build_config import DatasetBuildConfig
from minegs.dataset.cameras import (
    CameraConvention,
    axis_aligned_rotations,
    colmap_pose_local_from_cam,
    convention_label,
    image_pose_source_from_e57cam,
    pinhole_intrinsics,
)
from minegs.dataset.materialize import build_dataset
from minegs.dataset.reprojection import project_points, rgb_residual
from minegs.dataset.staging_input import load_staging
from minegs.ingest.common import colmap_io
from minegs.ingest.e57.images import ImageAsset
from PIL import Image as PILImage


def _asset(**meta_over) -> ImageAsset:
    meta = {
        "focalLength": 0.004096,
        "pixelWidth": 2e-6,
        "pixelHeight": 2e-6,
        "principalPointX": 2048.0,
        "principalPointY": 2048.0,
        "pose_rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "pose_translation": [1.0, 2.0, 3.0],
    }
    meta.update(meta_over)
    meta = {k: v for k, v in meta.items() if v is not None}
    return ImageAsset(
        image_id="image_000",
        source="e57_embedded",
        source_index=0,
        name="Skybox 0",
        representation="pinhole",
        representation_source="pinholeRepresentation",
        width=4096,
        height=4096,
        blob_field="jpegImage",
        vendor_metadata=meta,
    )


# 11. known-value intrinsics conversion
def test_pinhole_intrinsics_known_values():
    intr = pinhole_intrinsics(_asset())
    assert intr.fx == pytest.approx(2048.0) and intr.fy == pytest.approx(2048.0)
    assert (intr.cx, intr.cy, intr.width, intr.height) == (2048.0, 2048.0, 4096, 4096)
    assert np.allclose(intr.K(), [[2048, 0, 2048], [0, 2048, 2048], [0, 0, 1]])


# 12. known camera pose -> COLMAP pose (dataset images match the synthetic truth)
def test_known_pose_becomes_colmap_pose(dataset_small, staging_small):
    model = colmap_io.read_model(dataset_small.dataset_dir / "sparse" / "0")
    frames = json.loads((dataset_small.dataset_dir / "build_config.json").read_text())["frames"]
    T_tls_from_source = SE3.from_matrix(frames["T_tls_from_source"])
    T_local_from_source = SE3.from_matrix(frames["T_tls_from_local"]).inverse() @ T_tls_from_source
    for im in model.images.values():
        image_id = im.name.split("_", 1)[1].rsplit(".", 1)[0]
        truth = T_local_from_source @ staging_small.image_poses_cam[image_id]
        assert im.world_from_cam.allclose(truth, atol=1e-6), im.name


# 13. world_from_cam / cam_from_world direction regression
def test_pose_direction_regression():
    conv = CameraConvention(R_e57cam_from_cam=np.eye(3).tolist(), source="explicit")
    T_local_from_source = SE3.from_translation([-100.0, 0.0, 0.0])
    # rot_x(-90) turns the camera's optical axis (+z) onto source +y
    T_source_from_e57cam = SE3(rot_x(-90.0), [100.0, 5.0, 1.0])
    T_local_from_cam = colmap_pose_local_from_cam(T_local_from_source, T_source_from_e57cam, conv)
    im = colmap_io.Image.from_world_from_cam(1, T_local_from_cam, 1, "a.png")
    assert np.allclose(im.center, [0.0, 5.0, 1.0])
    assert np.allclose(T_local_from_cam.R @ [0, 0, 1], [0, 1, 0], atol=1e-12)
    # a point 2 m in front of the camera lands at centre + 2·(source +y), with camera z > 0
    p_local = T_local_from_cam.apply([0.0, 0.0, 2.0])
    assert np.allclose(im.cam_from_world.apply(p_local), [0.0, 0.0, 2.0], atol=1e-9)
    assert np.allclose(p_local, [0.0, 7.0, 1.0])


# 14. axis-convention regression: the true convention projects points onto their own colour
def test_axis_convention_regression(staging_small):
    tree = load_staging(staging_small.staging_dir)
    from minegs.core.pointcloud import read_ply

    st = tree.inventory.station_candidates[1]
    scan_id = st.scan_ids[0]
    cloud = read_ply(tree.scan_ply(scan_id)).subsample(20_000)
    truth = staging_small.R_e57cam_from_cam

    def residual(R):
        tot, n_tot = 0.0, 0
        for m in tree.mapping.mappings:
            if m.scan_id != scan_id:
                continue
            asset = tree.asset(m.image_id)
            intr = pinhole_intrinsics(asset)
            T = image_pose_source_from_e57cam(asset)
            img = np.asarray(PILImage.open(tree.image_path(m.image_id)).convert("RGB"))
            proj = project_points(
                intr.K(), (T @ SE3(R, np.zeros(3))).inverse(), cloud.xyz, intr.width, intr.height
            )
            res, n = rgb_residual(img, proj, cloud.rgb)
            tot, n_tot = tot + res * n, n_tot + n
        return tot / n_tot, n_tot

    res_true, n_true = residual(truth)
    res_wrong, _ = residual(np.eye(3))
    # the contract is the margin: the wrong convention is far worse, not merely worse
    assert n_true > 5000 and res_true < 35.0
    assert res_wrong > res_true + 30.0
    assert convention_label(truth) == "cam(+X,-Y,-Z)"


# 15. missing required intrinsic -> refusal
@pytest.mark.parametrize(
    "field", ["focalLength", "pixelWidth", "pixelHeight", "principalPointX", "principalPointY"]
)
def test_missing_intrinsic_refused(field):
    with pytest.raises(ContractError, match=field):
        pinhole_intrinsics(_asset(**{field: None}))
    with pytest.raises(ContractError, match="positive"):
        pinhole_intrinsics(_asset(pixelWidth=0.0))


# 16. invalid image pose -> refusal
def test_invalid_image_pose_refused():
    with pytest.raises(ContractError, match="no image pose"):
        image_pose_source_from_e57cam(_asset(pose_rotation_wxyz=None))
    with pytest.raises(ContractError, match="norm"):
        image_pose_source_from_e57cam(_asset(pose_rotation_wxyz=[0.5, 0.0, 0.0, 0.0]))
    with pytest.raises(ContractError, match="non-finite"):
        image_pose_source_from_e57cam(_asset(pose_translation=[float("nan"), 0, 0]))
    ok = image_pose_source_from_e57cam(
        _asset(pose_rotation_wxyz=rotmat_to_quat(rot_z(45)).tolist())
    )
    assert ok.allclose(SE3(rot_z(45), [1, 2, 3]))


# 17. image file name / index is not orientation evidence
def test_image_names_are_not_orientation_evidence(staging_small, build_config_small, tmp_path):
    src = staging_small.staging_dir
    shutil.copytree(src, tmp_path / "staging")
    rep = json.loads((tmp_path / "staging" / "pano_mapping.json").read_text())
    for a in rep["images"]:
        a["name"] = "Skybox 5" if a["name"] != "Skybox 5" else "Skybox 0"  # lie about every face
    (tmp_path / "staging" / "pano_mapping.json").write_text(json.dumps(rep))
    cfg = build_config_small.model_copy(
        update={
            "split": build_config_small.split.model_copy(),
            "geometry_holdout": None,
            "centerline": None,
        }
    )
    a = build_dataset(src, tmp_path / "a", cfg)
    b = build_dataset(tmp_path / "staging", tmp_path / "b", cfg)
    assert (a.dataset_dir / "sparse" / "0" / "images.txt").read_text() == (
        b.dataset_dir / "sparse" / "0" / "images.txt"
    ).read_text()


def _with_status(staging_root, tmp_path, image_id: str, status: str):
    shutil.copytree(staging_root, tmp_path / "staging")
    p = tmp_path / "staging" / "pano_mapping.json"
    rep = json.loads(p.read_text())
    for m in rep["mappings"]:
        if m["image_id"] == image_id:
            m["status"], m["scan_id"], m["station_id"] = status, None, None
            m["evidence_type"] = "none"
    p.write_text(json.dumps(rep))
    # extraction_manifest carries the same statuses; keep them consistent
    p2 = tmp_path / "staging" / "extraction_manifest.json"
    em = json.loads(p2.read_text())
    for io in em["image_outputs"]:
        if io["image_id"] == image_id:
            io["mapping_status"], io["mapped_scan_id"], io["mapped_station_id"] = status, None, None
    for m in em["mapping_report"]["mappings"]:
        if m["image_id"] == image_id:
            m["status"], m["scan_id"], m["station_id"] = status, None, None
            m["evidence_type"] = "none"
    p2.write_text(json.dumps(em))
    return tmp_path / "staging"


# 18. unresolved mapping -> refusal (not a silent drop)
def test_unresolved_mapping_refused(staging_small, build_config_small, tmp_path):
    stg = _with_status(staging_small.staging_dir, tmp_path, "image_003", "unmapped")
    cfg = build_config_small.model_copy(update={"geometry_holdout": None, "centerline": None})
    with pytest.raises(ContractError, match="image_003=unmapped"):
        build_dataset(stg, tmp_path / "ds", cfg)
    assert not (tmp_path / "ds").exists()


# 19. mapping conflict -> refusal
@pytest.mark.parametrize("status", ["conflict", "ambiguous", "orphan"])
def test_conflicting_mapping_refused(staging_small, build_config_small, tmp_path, status):
    stg = _with_status(staging_small.staging_dir, tmp_path, "image_007", status)
    cfg = build_config_small.model_copy(update={"geometry_holdout": None, "centerline": None})
    with pytest.raises(ContractError, match=f"image_007={status}"):
        build_dataset(stg, tmp_path / "ds", cfg)


# 20. one station with several pinhole images -> one capture group
def test_station_faces_form_one_group(dataset_small):
    m = dataset_small.manifest
    assert all(g.type == "tls_station" for g in m.capture_groups.values())
    assert all(len(g.members) == 6 for g in m.capture_groups.values())
    assert len(m.capture_groups) == 5


# 21. deterministic naming and order
def test_deterministic_image_naming(dataset_small, staging_small, build_config_small, tmp_path):
    names = dataset_small.manifest.all_images()
    assert names == sorted(names)
    assert names[:2] == ["S000_image_000.png", "S000_image_001.png"]
    again = build_dataset(staging_small.staging_dir, tmp_path / "again", build_config_small)
    assert again.manifest.all_images() == names
    a = (dataset_small.dataset_dir / "sparse" / "0" / "images.txt").read_text()
    b = (again.dataset_dir / "sparse" / "0" / "images.txt").read_text()
    assert a == b
    assert again.manifest.provenance.config_hash == dataset_small.manifest.provenance.config_hash


# 22. spherical path: equirect -> ring crops -> COLMAP, and it reprojects
def test_spherical_path(tmp_path):
    from minegs.core.pointcloud import read_ply
    from minegs.core.synthetic_staging import StagingSpec, generate_staging

    r = generate_staging(
        tmp_path / "s",
        StagingSpec(
            length_m=45,
            station_spacing_m=15,
            points_per_m=2000,
            image_size=48,
            image_mode="spherical",
        ),
    )
    cfg = DatasetBuildConfig.model_validate(
        {
            "dataset_id": "sph",
            "source_frame": {"mode": "explicit_identity"},
            "camera": {
                "mode": "e57_spherical",
                "ring_crop": {"n_yaw": 4, "fov_deg": 90, "width": 48, "height": 48},
            },
            "initialization": {"voxel_m": 0.1, "max_points": 20000, "sparse_max_points": 1000},
        }
    )
    res = build_dataset(r.staging_dir, tmp_path / "ds", cfg)
    m = res.manifest
    assert len(m.all_images()) == 3 * 4 and m.pano_convention is not None
    model = colmap_io.read_model(res.dataset_dir / "sparse" / "0")
    init = read_ply(res.dataset_dir / "init_points.ply")
    im = next(iter(model.images.values()))
    cam = model.cameras[im.camera_id]
    proj = project_points(cam.K(), im.cam_from_world, init.xyz, cam.width, cam.height)
    assert proj.n_inside > 100


# 23. cylindrical is not silently treated as spherical
def test_cylindrical_is_refused(tmp_path):
    from minegs.core.synthetic_staging import StagingSpec, generate_staging

    r = generate_staging(
        tmp_path / "s",
        StagingSpec(
            length_m=45,
            station_spacing_m=15,
            points_per_m=1000,
            image_size=32,
            image_mode="spherical",
        ),
    )
    for name in ("pano_mapping.json", "extraction_manifest.json"):
        p = r.staging_dir / name
        p.write_text(
            p.read_text()
            .replace('"spherical"', '"cylindrical"')
            .replace("sphericalRepresentation", "cylindricalRepresentation")
        )
    cfg = DatasetBuildConfig.model_validate(
        {
            "dataset_id": "cyl",
            "source_frame": {"mode": "explicit_identity"},
            "camera": {
                "mode": "e57_spherical",
                "ring_crop": {"n_yaw": 4, "width": 32, "height": 32},
            },
        }
    )
    with pytest.raises(ContractError, match="cylindrical"):
        build_dataset(r.staging_dir, tmp_path / "ds", cfg)


def test_convention_candidates_are_proper_rotations():
    rots = axis_aligned_rotations()
    assert len(rots) == 24 and len({convention_label(R) for R in rots}) == 24
    for R in rots:
        assert np.allclose(R.T @ R, np.eye(3)) and abs(np.linalg.det(R) - 1) < 1e-12
    with pytest.raises(Exception, match="reflection"):
        CameraConvention(R_e57cam_from_cam=np.diag([1.0, 1.0, -1.0]).tolist(), source="explicit")
