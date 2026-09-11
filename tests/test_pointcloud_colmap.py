import numpy as np
import pytest
from minegs.core.errors import FrameError
from minegs.core.frames import SE3, rot_z
from minegs.core.pointcloud import PointCloud, read_ply, voxel_downsample, write_ply
from minegs.ingest.common import colmap_io
from minegs.ingest.common.equirect import RingCropSpec, crop_equirect
from minegs.ingest.common.geometry import (
    PanoConvention,
    dirs_to_equirect_uv,
    equirect_uv_to_dirs,
    spherical_to_cart,
)


def test_ply_roundtrip_binary_and_ascii(tmp_path, rng):
    pc = PointCloud(
        rng.normal(size=(100, 3)),
        rng.integers(0, 255, (100, 3)),
        extra={"opacity": rng.normal(size=100).astype(np.float32)},
        frame="LOCAL_METRIC",
    )
    for binary in (True, False):
        p = write_ply(pc, tmp_path / f"a_{binary}.ply", binary=binary)
        back = read_ply(p)
        assert np.allclose(back.xyz, pc.xyz, atol=1e-5)
        assert np.array_equal(back.rgb, pc.rgb)
        assert back.frame == "LOCAL_METRIC"
        assert np.allclose(back.extra["opacity"], pc.extra["opacity"], atol=1e-6)


def test_ply_float32_guard(tmp_path):
    pc = PointCloud([[318300.0, 4012300.0, 120.0]], frame="TLS_GLOBAL")
    with pytest.raises(FrameError):
        write_ply(pc, tmp_path / "bad.ply")
    write_ply(pc, tmp_path / "ok.ply", xyz_dtype="f8")
    assert np.allclose(read_ply(tmp_path / "ok.ply").xyz, pc.xyz)


def test_gaussian_cloud_transform_rotates_quaternions(rng):
    n = 5
    extra = {
        "opacity": np.zeros(n),
        "scale_0": np.zeros(n),
        "scale_1": np.zeros(n),
        "scale_2": np.zeros(n),
    }
    extra.update({f"rot_{i}": (np.ones(n) if i == 0 else np.zeros(n)) for i in range(4)})
    pc = PointCloud(rng.normal(size=(n, 3)), extra=extra)
    assert pc.is_gaussian_cloud()
    out = pc.transformed(SE3(rot_z(90), np.zeros(3)))
    q = np.stack([out.extra[f"rot_{i}"] for i in range(4)], 1)
    assert np.allclose(np.abs(q[0]), [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], atol=1e-9)


def test_voxel_downsample():
    xyz = np.array([[0, 0, 0], [0.001, 0, 0], [1, 1, 1]])
    assert list(voxel_downsample(xyz, 0.1)) == [0, 2]


def test_colmap_model_roundtrip_with_rig(tmp_path):
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    cams = {1: colmap_io.Camera.pinhole(1, K, 640, 480)}
    T = SE3(rot_z(30), [1, 2, 3])
    im = colmap_io.Image.from_world_from_cam(1, T, 1, "a.png")
    im2 = colmap_io.Image.from_world_from_cam(2, T @ SE3(rot_z(90), np.zeros(3)), 1, "b.png")
    pts = {
        7: colmap_io.Point3D(
            7, np.array([1.0, 2.0, 3.5]), np.array([1, 2, 3]), 0.1, np.array([1]), np.array([0])
        )
    }
    im.xys = np.array([[10.0, 20.0]])
    im.point3D_ids = np.array([7])
    model = colmap_io.ColmapModel(cams, {1: im, 2: im2}, pts)
    model.rigs[1] = colmap_io.Rig(1, [colmap_io.RigSensor(1, None)])
    model.frames[1] = colmap_io.Frame(1, 1, T.inverse(), [(1, 1), (1, 2)])
    d = colmap_io.write_model(model, tmp_path / "sparse")
    back = colmap_io.read_model(d)
    assert back.images[1].world_from_cam.allclose(T, 1e-6)
    assert back.cameras[1].K()[0, 0] == 500
    assert np.allclose(back.points3D[7].xyz, [1, 2, 3.5])
    assert back.images[1].point3D_ids.tolist() == [7]
    assert back.frames[1].image_ids == [(1, 1), (1, 2)]
    assert back.frames[1].rig_from_world.allclose(T.inverse(), 1e-6)
    # re-express in another frame keeps relative geometry
    moved = back.transformed(SE3.from_translation([10, 0, 0]))
    assert np.allclose(moved.images[1].center, np.asarray(T.t) + np.array([10, 0, 0]))
    _uv, z = colmap_io.project(K, moved.images[1].cam_from_world, moved.points3D[7].xyz[None])
    assert z[0] > 0


def test_pano_convention_roundtrip(rng):
    d = rng.normal(size=(200, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    for conv in (PanoConvention(), PanoConvention(-1, True, 180.0)):
        uv = dirs_to_equirect_uv(d, 4000, 2000, conv)
        back = equirect_uv_to_dirs(uv, 4000, 2000, conv)
        assert np.allclose(back, d, atol=1e-9)
    assert np.allclose(spherical_to_cart(np.ones(1), np.zeros(1), np.zeros(1)), [[1, 0, 0]])


def test_ring_crop_intrinsics_and_directions():
    spec = RingCropSpec(n_yaw=4, fov_deg=90, width=64, height=64)
    K = spec.K()
    assert K[0, 0] == pytest.approx(32.0)  # 90° fov -> f = w/2
    views = spec.crops()
    assert len(views) == 4
    # crop 0 looks along +x, crop 1 (yaw 90) along +y; camera z is forward
    for v, fwd in zip(views, ([1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]), strict=True):
        assert np.allclose(v.R_scanner_from_cam @ [0, 0, 1], fwd, atol=1e-12)
        assert np.allclose(
            v.R_scanner_from_cam @ [0, 1, 0], [0, 0, -1], atol=1e-12
        )  # image down = -z
    # a synthetic equirect with a bright band at elevation 0 lands in the middle row of each crop
    pano = np.zeros((200, 400, 3), np.uint8)
    pano[98:102] = 255
    img = crop_equirect(pano, views[0])
    assert img[32, 32, 0] > 200 and img[5, 32, 0] < 30
