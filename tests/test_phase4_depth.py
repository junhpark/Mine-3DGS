"""Phase 4 C1 — the depth supervision artifact and its leakage contract.

Every negative test here contaminates a valid artifact on purpose and expects a refusal. When
the point is the leakage check, ``_reseal`` makes every *other* field consistent again: the
sample hash, the counts and the support ranges. The only thing left wrong is then the thing
under test, and the verifier has to find it by re-deriving the sample, not by noticing a
stale digest.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from minegs.core.errors import ContractError
from minegs.core.manifest import Manifest
from minegs.core.pointcloud import PointCloud, read_ply, write_ply
from minegs.core.provenance import sha256_file, sha256_tree
from minegs.ingest.common import colmap_io
from minegs.train.runner.base import DATASET_HASH_PATTERNS
from minegs.train.supervision.build import build_sfm_tracks, build_tls_projection
from minegs.train.supervision.depth import (
    RECORD_FILE,
    SAMPLE_DTYPE,
    SAMPLES_FILE,
    DepthSupervisionRecord,
    recorded_support,
    verify_depth_supervision,
)
from minegs.train.supervision.support import Support

from phase4_scene import tls_scene, video_scene


@pytest.fixture(scope="module")
def tls(tmp_path_factory):
    sc = tls_scene(tmp_path_factory.mktemp("p4_tls"))
    art = build_tls_projection(sc.dataset_dir, sc.cloud, tmp_path_factory.mktemp("p4a") / "tls")
    sc.artifact = art
    return sc


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    sc = video_scene(tmp_path_factory.mktemp("p4_vid"))
    art = build_sfm_tracks(sc.dataset_dir, tmp_path_factory.mktemp("p4b") / "sfm")
    sc.artifact = art
    return sc


def _copy(art, tmp_path: Path, name: str = "a") -> Path:
    dst = tmp_path / name
    shutil.copytree(art.path, dst)
    return dst


def _reseal(dataset_dir: Path, art: Path, samples: np.ndarray, **record_over) -> None:
    """Rewrite samples and make every bookkeeping field agree with them again."""
    np.save(art / SAMPLES_FILE, samples, allow_pickle=False)
    rec = DepthSupervisionRecord.load(art / RECORD_FILE)
    data = rec.model_dump(mode="json")
    data.update(record_over)
    names = data["images"]
    manifest = Manifest.load_dataset(dataset_dir)
    model = colmap_io.read_model(dataset_dir / "sparse" / "0")
    data["samples_sha256"] = sha256_file(art / SAMPLES_FILE)
    data["n_samples"] = len(samples)
    data["n_samples_in_loss"] = int((samples["confidence"] > 0).sum())
    data["per_image_counts"] = np.bincount(
        samples["image"].astype(np.int64), minlength=len(names)
    ).tolist()
    support = Support.of_dataset(dataset_dir, manifest)
    sr = recorded_support(samples, names, model, support)
    data["support_ranges_m"] = None if sr is None else [list(r) for r in sr]
    (art / RECORD_FILE).write_text(json.dumps(data, indent=2))


def _sample_at(model, name: str, xyz: np.ndarray, image_index: int) -> np.ndarray:
    """A sample that observes LOCAL_METRIC point `xyz` from image `name`."""
    im = model.image_by_name()[name]
    cam = model.cameras[im.camera_id]
    uv, z = colmap_io.project(cam.K(), im.cam_from_world, xyz.reshape(1, 3))
    r = np.zeros(1, dtype=SAMPLE_DTYPE)
    r["image"], r["u"], r["v"], r["depth_m"], r["confidence"] = (
        image_index,
        uv[0, 0],
        uv[0, 1],
        z[0],
        1.0,
    )
    return r


def _visible_from(model, names, xyz, want):
    """First (image_index, point) where a point satisfying `want` is in frame."""
    by_name = model.image_by_name()
    for k, n in enumerate(names):
        im = by_name[n]
        cam = model.cameras[im.camera_id]
        uv, z = colmap_io.project(cam.K(), im.cam_from_world, xyz)
        vis = colmap_io.visible_mask(uv, z, cam.width - 1, cam.height - 1, near=0.5) & want
        if vis.any():
            return k, n, xyz[np.flatnonzero(vis)[0]]
    raise AssertionError("no image sees such a point")


# ================================================================ the artifact itself


@pytest.mark.parametrize("which", ["tls", "video"])
def test_builders_emit_verified_metric_camera_z_artifacts(which, request):
    sc = request.getfixturevalue(which)
    rec = sc.artifact.record
    assert rec.frame == "LOCAL_METRIC" and rec.depth_unit == "m"
    assert rec.depth_semantics == "camera_z" and rec.pixel_convention == "colmap_continuous"
    assert rec.confidence_semantics == "binary_mask"
    assert rec.n_samples_in_loss > 0
    assert rec.excluded_holdout_ranges_m  # both scenes declare a holdout
    # verification is repeatable on the bytes alone
    again = verify_depth_supervision(sc.dataset_dir, sc.artifact.path)
    assert again.artifact_sha256 == sc.artifact.artifact_sha256


def test_tls_depth_is_the_camera_z_of_a_tls_point(tls):
    """Back-projected samples land on the scanned surface: depth means camera +z, in metres."""
    from minegs.train.supervision.depth import backproject_samples
    from scipy.spatial import cKDTree

    manifest = Manifest.load_dataset(tls.dataset_dir)
    model = colmap_io.read_model(tls.dataset_dir / "sparse" / "0")
    cloud = manifest.T_local_from_tls.apply(read_ply(tls.cloud).xyz)
    pts, _ = backproject_samples(tls.artifact.samples, tls.artifact.record.images, model)
    d, _ = cKDTree(cloud).query(pts)
    assert float(np.max(d)) < 1e-3  # float32 storage, not a different surface


def test_missing_record_is_not_an_artifact(tls, tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ContractError, match="not a depth supervision artifact"):
        verify_depth_supervision(tls.dataset_dir, tmp_path / "empty")


def test_tampered_sample_bytes_are_refused(tls, tmp_path):
    art = _copy(tls.artifact, tmp_path)
    s = np.load(art / SAMPLES_FILE)
    s["depth_m"][0] *= 1.5
    np.save(art / SAMPLES_FILE, s, allow_pickle=False)
    with pytest.raises(ContractError, match="changed after the record was written"):
        verify_depth_supervision(tls.dataset_dir, art)


def test_artifact_of_another_dataset_is_refused(tls, video):
    with pytest.raises(ContractError, match="made for dataset"):
        verify_depth_supervision(tls.dataset_dir, video.artifact.path)


def test_moved_camera_breaks_the_dataset_binding(tls, tmp_path):
    """Same dataset id, one pose changed: the samples were measured through other cameras."""
    ds = tmp_path / "ds"
    shutil.copytree(tls.dataset_dir, ds)
    model = colmap_io.read_model(ds / "sparse" / "0")
    im = next(iter(model.images.values()))
    im.tvec = np.asarray(im.tvec) + np.array([0.05, 0.0, 0.0])
    colmap_io.write_model(model, ds / "sparse" / "0")
    with pytest.raises(ContractError, match="poses_sha256"):
        verify_depth_supervision(ds, tls.artifact.path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("frame", "TLS_GLOBAL"),
        ("depth_unit", "mm"),
        ("depth_semantics", "ray_distance"),
        ("depth_semantics", "unknown"),
        ("pixel_convention", "pixel_corner"),
    ],
)
def test_non_metric_or_unknown_semantics_are_refused(tls, tmp_path, field, value):
    art = _copy(tls.artifact, tmp_path)
    data = json.loads((art / RECORD_FILE).read_text())
    data[field] = value
    (art / RECORD_FILE).write_text(json.dumps(data))
    with pytest.raises(ContractError, match=field):
        verify_depth_supervision(tls.dataset_dir, art)


@pytest.mark.parametrize("bad", [0.5, -1.0, 2.0, float("nan")])
def test_invalid_confidence_is_refused(tls, tmp_path, bad):
    art = _copy(tls.artifact, tmp_path)
    s = np.load(art / SAMPLES_FILE)
    s["confidence"][3] = bad
    _reseal(tls.dataset_dir, art, s)
    with pytest.raises(ContractError, match=r"confidence|non-finite"):
        verify_depth_supervision(tls.dataset_dir, art)


def test_unit_interval_weights_accept_fractions_but_not_out_of_range(tls, tmp_path):
    art = _copy(tls.artifact, tmp_path)
    s = np.load(art / SAMPLES_FILE)
    s["confidence"][3] = 0.5
    _reseal(tls.dataset_dir, art, s, confidence_semantics="unit_interval_weight")
    verify_depth_supervision(tls.dataset_dir, art)
    s["confidence"][3] = 1.5
    _reseal(tls.dataset_dir, art, s, confidence_semantics="unit_interval_weight")
    with pytest.raises(ContractError, match=r"outside \[0, 1\]"):
        verify_depth_supervision(tls.dataset_dir, art)


def test_unknown_confidence_semantics_is_refused(tls, tmp_path):
    art = _copy(tls.artifact, tmp_path)
    _reseal(tls.dataset_dir, art, np.load(art / SAMPLES_FILE), confidence_semantics="sigma_m")
    with pytest.raises(ContractError, match="confidence_semantics"):
        verify_depth_supervision(tls.dataset_dir, art)


def test_nonexistent_image_is_refused(tls, tmp_path):
    art = _copy(tls.artifact, tmp_path)
    data = json.loads((art / RECORD_FILE).read_text())
    data["images"][0] = "S99_nowhere.png"
    (art / RECORD_FILE).write_text(json.dumps(data))
    with pytest.raises(ContractError, match="not in this dataset"):
        verify_depth_supervision(tls.dataset_dir, art)


def test_sample_outside_its_image_is_refused(tls, tmp_path):
    art = _copy(tls.artifact, tmp_path)
    s = np.load(art / SAMPLES_FILE)
    s["u"][0] = 1e4
    _reseal(tls.dataset_dir, art, s)
    with pytest.raises(ContractError, match="outside its"):
        verify_depth_supervision(tls.dataset_dir, art)


def test_sensor_depth_is_not_pretended_to_be_supported(tls, tmp_path):
    art = _copy(tls.artifact, tmp_path)
    data = json.loads((art / RECORD_FILE).read_text())
    data["source_kind"] = "sensor_depth"
    (art / RECORD_FILE).write_text(json.dumps(data))
    with pytest.raises(ContractError, match="no Phase 4 builder"):
        verify_depth_supervision(tls.dataset_dir, art)


def test_a_record_cannot_grant_itself_a_laxer_support_rule(tls, tmp_path):
    art = _copy(tls.artifact, tmp_path)
    data = json.loads((art / RECORD_FILE).read_text())
    data["creation_params"]["max_radial_m"] = 500.0
    (art / RECORD_FILE).write_text(json.dumps(data))
    with pytest.raises(ContractError, match="contract allows"):
        verify_depth_supervision(tls.dataset_dir, art)


# ================================================================ leakage (AD-3)


def test_tls_sample_inside_the_holdout_is_refused(tls, tmp_path):
    manifest = Manifest.load_dataset(tls.dataset_dir)
    model = colmap_io.read_model(tls.dataset_dir / "sparse" / "0")
    support = Support.of_dataset(tls.dataset_dir, manifest)
    cloud = manifest.T_local_from_tls.apply(read_ply(tls.cloud).xyz)
    s, ok = support.locate(cloud)
    names = tls.artifact.record.images
    k, _, X = _visible_from(model, names, cloud, ok & support.in_holdout(s))
    art = _copy(tls.artifact, tmp_path)
    samples = np.concatenate([np.load(art / SAMPLES_FILE), _sample_at(model, names[k], X, k)])
    _reseal(tls.dataset_dir, art, samples)
    with pytest.raises(ContractError, match="holdout_point"):
        verify_depth_supervision(tls.dataset_dir, art)


def test_sample_whose_ray_crosses_the_holdout_is_refused(tls, tmp_path):
    """The point itself is outside the holdout; the camera is on the other side of it."""
    manifest = Manifest.load_dataset(tls.dataset_dir)
    model = colmap_io.read_model(tls.dataset_dir / "sparse" / "0")
    support = Support.of_dataset(tls.dataset_dir, manifest)
    cloud = manifest.T_local_from_tls.apply(read_ply(tls.cloud).xyz)
    s, ok = support.locate(cloud)
    lo, hi = support.holdout[0]
    names = tls.artifact.record.images
    by_name = model.image_by_name()
    found = None
    for k, n in enumerate(names):
        s_cam, cam_ok = support.locate(by_name[n].center.reshape(1, 3))
        if not cam_ok[0] or support.in_holdout(s_cam)[0]:
            continue
        beyond = (s > hi + 1.0) if s_cam[0] < lo else (s < lo - 1.0)
        im = by_name[n]
        cam = model.cameras[im.camera_id]
        uv, z = colmap_io.project(cam.K(), im.cam_from_world, cloud)
        vis = colmap_io.visible_mask(uv, z, cam.width - 1, cam.height - 1, near=0.5)
        hit = np.flatnonzero(vis & ok & beyond & ~support.in_holdout(s))
        if len(hit):
            found = (k, n, cloud[hit[0]])
            break
    assert found, "the scene should have a view looking across the holdout"
    k, n, X = found
    art = _copy(tls.artifact, tmp_path)
    samples = np.concatenate([np.load(art / SAMPLES_FILE), _sample_at(model, n, X, k)])
    _reseal(tls.dataset_dir, art, samples)
    with pytest.raises(ContractError, match="holdout_ray"):
        verify_depth_supervision(tls.dataset_dir, art)


def test_sfm_sample_inside_the_holdout_is_refused(video, tmp_path):
    manifest = Manifest.load_dataset(video.dataset_dir)
    model = colmap_io.read_model(video.dataset_dir / "sparse" / "0")
    support = Support.of_dataset(video.dataset_dir, manifest)
    pts = manifest.T_local_from_tls.apply(video.survey.points_tls)
    s, ok = support.locate(pts)
    names = video.artifact.record.images
    k, _, X = _visible_from(model, names, pts, ok & support.in_holdout(s))
    art = _copy(video.artifact, tmp_path)
    samples = np.concatenate([np.load(art / SAMPLES_FILE), _sample_at(model, names[k], X, k)])
    _reseal(video.dataset_dir, art, samples)
    with pytest.raises(ContractError, match="holdout"):
        verify_depth_supervision(video.dataset_dir, art)


def test_held_out_image_cannot_supply_supervision(tls, tmp_path):
    manifest = Manifest.load_dataset(tls.dataset_dir)
    test_image = manifest.test_images()[0]
    art = _copy(tls.artifact, tmp_path)
    data = json.loads((art / RECORD_FILE).read_text())
    data["images"][0] = test_image
    (art / RECORD_FILE).write_text(json.dumps(data))
    with pytest.raises(ContractError, match="held-out image cannot supply supervision"):
        verify_depth_supervision(tls.dataset_dir, art)


def test_image_excluded_by_the_holdout_cannot_supply_supervision(video, tmp_path):
    """images_excluded=True: the group that straddles the holdout is not a training group."""
    manifest = Manifest.load_dataset(video.dataset_dir)
    dropped = sorted(set(manifest.all_images()) - set(manifest.train_images()))
    assert dropped
    art = _copy(video.artifact, tmp_path)
    data = json.loads((art / RECORD_FILE).read_text())
    data["images"][0] = dropped[0]
    (art / RECORD_FILE).write_text(json.dumps(data))
    with pytest.raises(ContractError, match="not training images"):
        verify_depth_supervision(video.dataset_dir, art)


def test_sfm_tracks_through_held_out_frames_never_become_samples(video):
    """A depth triangulated with a held-out view carries that view: excluded at the point."""
    assert video.artifact.record.exclusions.get("track_touches_non_training_image", 0) > 0


def test_unlocatable_support_fails_closed(tls, tmp_path):
    """A point beyond the end of the axis has no chainage; with a holdout declared, refuse."""
    manifest = Manifest.load_dataset(tls.dataset_dir)
    model = colmap_io.read_model(tls.dataset_dir / "sparse" / "0")
    support = Support.of_dataset(tls.dataset_dir, manifest)
    cl = support.centerline
    t_end = cl.tangent_at(cl.s_end)
    names = tls.artifact.record.images
    by_name = model.image_by_name()
    # a point 15 m straight ahead of a camera: off the axis, or past its end, located nowhere
    found = None
    for idx, name in enumerate(names):
        im = by_name[name]
        cam = model.cameras[im.camera_id]
        X = im.center + im.world_from_cam.R @ np.array([0.0, 0.0, 15.0])
        _, ok = support.locate(X.reshape(1, 3))
        uv, z = colmap_io.project(cam.K(), im.cam_from_world, X.reshape(1, 3))
        if not ok[0] and colmap_io.visible_mask(uv, z, cam.width - 1, cam.height - 1)[0]:
            found = (idx, name, X)
            break
    assert found, f"no unlocated point found (axis tangent {t_end})"
    k, n, X = found
    art = _copy(tls.artifact, tmp_path)
    samples = np.concatenate([np.load(art / SAMPLES_FILE), _sample_at(model, n, X, k)])
    _reseal(tls.dataset_dir, art, samples)
    with pytest.raises(ContractError, match="unlocated_support"):
        verify_depth_supervision(tls.dataset_dir, art)


def test_holdout_without_a_centerline_refuses_build_and_verification(tls, tmp_path):
    ds = tmp_path / "ds"
    shutil.copytree(tls.dataset_dir, ds)
    manifest = Manifest.load_dataset(ds)
    manifest.centerline = None
    manifest.save_dataset(ds)
    with pytest.raises(ContractError, match="no centerline"):
        build_tls_projection(ds, tls.cloud, tmp_path / "out")
    with pytest.raises(ContractError, match=r"no centerline|no longer matches"):
        verify_depth_supervision(ds, tls.artifact.path)


def test_tls_depth_cannot_supervise_an_image_only_dataset(video, tls, tmp_path):
    with pytest.raises(ContractError, match="TLS-assisted"):
        build_tls_projection(video.dataset_dir, video.tls, tmp_path / "x")
    # and an sfm artifact relabelled as TLS is refused by the verifier, not just the builder
    art = _copy(video.artifact, tmp_path)
    data = json.loads((art / RECORD_FILE).read_text())
    data["source_kind"] = "tls_projection"
    (art / RECORD_FILE).write_text(json.dumps(data))
    with pytest.raises(ContractError, match="TLS-assisted"):
        verify_depth_supervision(video.dataset_dir, art)


def test_sfm_tracks_need_an_image_only_dataset(tls, tmp_path):
    with pytest.raises(ContractError, match="has none"):
        build_sfm_tracks(tls.dataset_dir, tmp_path / "x")


def test_tls_builder_never_reads_holdout_geometry(tls):
    ex = tls.artifact.record.exclusions
    assert ex.get("cloud_holdout_point", 0) > 0
    assert ex.get("holdout_ray", 0) > 0


def test_cloud_with_an_unknown_frame_is_refused(tls, tmp_path):
    pc = read_ply(tls.cloud)
    p = write_ply(PointCloud(pc.xyz[:1000], frame="UNKNOWN"), tmp_path / "c.ply", xyz_dtype="f8")
    with pytest.raises(ContractError, match="declares frame"):
        build_tls_projection(tls.dataset_dir, p, tmp_path / "x")


# ================================================================ init separation (AD-1)


def test_changing_init_neither_rewrites_nor_invalidates_depth_evidence(tls, tmp_path):
    ds = tmp_path / "ds"
    shutil.copytree(tls.dataset_dir, ds)
    before = sha256_tree(tls.artifact.path)
    init = read_ply(ds / "init_points.ply")
    write_ply(
        PointCloud(init.xyz[: len(init) // 2] + 0.01, frame="LOCAL_METRIC"), ds / "init_points.ply"
    )
    model = colmap_io.read_model(ds / "sparse" / "0")
    model.points3D = dict(list(model.points3D.items())[:10])
    colmap_io.write_model(model, ds / "sparse" / "0")
    v = verify_depth_supervision(ds, tls.artifact.path)
    assert sha256_tree(tls.artifact.path) == before == v.artifact_sha256


def test_building_depth_evidence_does_not_touch_init_or_the_dataset_hash(tmp_path):
    sc = tls_scene(tmp_path / "scene")
    ds = sc.dataset_dir
    init_before = sha256_file(ds / "init_points.ply")
    sparse_before = sha256_tree(ds / "sparse")
    hash_before = sha256_tree(ds, DATASET_HASH_PATTERNS)
    v = build_tls_projection(ds, sc.cloud)  # default location: inside the dataset
    assert v.path.parent == ds / "supervision" / "depth"
    assert sha256_file(ds / "init_points.ply") == init_before
    assert sha256_tree(ds / "sparse") == sparse_before
    assert sha256_tree(ds, DATASET_HASH_PATTERNS) == hash_before


def test_artifact_is_written_once(tls):
    with pytest.raises(ContractError, match="written once"):
        build_tls_projection(tls.dataset_dir, tls.cloud, tls.artifact.path)
