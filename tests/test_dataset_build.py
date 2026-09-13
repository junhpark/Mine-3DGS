"""Phase 0C §33 — leakage and dataset contract."""

from __future__ import annotations

import json

import numpy as np
import pytest
from minegs.core.centerline import Centerline
from minegs.core.errors import ContractError
from minegs.core.manifest import Manifest
from minegs.core.pointcloud import read_ply
from minegs.dataset.build_config import DatasetBuildConfig
from minegs.dataset.materialize import build_dataset, holdout_mask
from minegs.eval.protocol import Claim, Protocol, judge
from minegs.ingest.common import colmap_io

HOLDOUT = (20.0, 26.0)


def _local_centerline(res):
    m = res.manifest
    cl = Centerline.from_csv(res.dataset_dir / m.centerline.file, "TLS_GLOBAL", m.centerline.source)
    return cl.transformed(m.T_local_from_tls, "LOCAL_METRIC")


# 24. test groups are excluded from initialization groups
def test_test_groups_not_in_init(dataset_small):
    m = dataset_small.manifest
    assert m.split.test_groups and not set(m.initialization.groups) & set(m.split.test_groups)
    assert set(m.initialization.groups) == set(m.split.train_groups)


# 25. holdout chainage points are removed from the init cloud, by their own coordinates
def test_holdout_points_removed_from_init(dataset_small):
    init = read_ply(dataset_small.dataset_dir / "init_points.ply")
    s, _ = _local_centerline(dataset_small).project(init.xyz)
    assert not holdout_mask(s, [HOLDOUT]).any()
    assert dataset_small.report["initialization"]["points_removed_by_holdout"] > 0


# 26. points just outside the range are retained
def test_points_just_outside_holdout_retained(dataset_small):
    init = read_ply(dataset_small.dataset_dir / "init_points.ply")
    s, _ = _local_centerline(dataset_small).project(init.xyz)
    assert ((s > HOLDOUT[1]) & (s < HOLDOUT[1] + 2.0)).sum() > 50
    assert ((s < HOLDOUT[0]) & (s > HOLDOUT[0] - 2.0)).sum() > 50


# 27. multiple holdout intervals
def test_multiple_holdout_intervals(staging_small, build_config_small, tmp_path):
    cfg = build_config_small.model_copy(deep=True)
    cfg.geometry_holdout.ranges_m = [(10.0, 14.0), (30.0, 33.0)]
    res = build_dataset(staging_small.staging_dir, tmp_path / "ds", cfg)
    init = read_ply(res.dataset_dir / "init_points.ply")
    s, _ = _local_centerline(res).project(init.xyz)
    assert not holdout_mask(s, [(10.0, 14.0), (30.0, 33.0)]).any()
    assert ((s > 14.0) & (s < 30.0)).sum() > 100
    assert res.manifest.initialization.excluded_chainage_ranges_m == [(10.0, 14.0), (30.0, 33.0)]


# 28. sparse points cannot reintroduce holdout geometry
def test_sparse_points_do_not_reintroduce_holdout(dataset_small):
    model = colmap_io.read_model(dataset_small.dataset_dir / "sparse" / "0")
    pts = model.points_xyz()
    assert len(pts) > 0
    s, _ = _local_centerline(dataset_small).project(pts)
    assert not holdout_mask(s, [HOLDOUT]).any()
    init = read_ply(dataset_small.dataset_dir / "init_points.ply")
    # every sparse point is one of the init points (same leak-free set)
    from scipy.spatial import cKDTree

    d, _ = cKDTree(init.xyz).query(pts)
    assert np.max(d) < 1e-4  # float32 PLY rounding at ~50 m magnitude is ~4e-6


# 29. manifest image set == COLMAP image set
def test_manifest_images_equal_colmap_images(dataset_small):
    model = colmap_io.read_model(dataset_small.dataset_dir / "sparse" / "0")
    assert set(dataset_small.manifest.all_images()) == {im.name for im in model.images.values()}
    for n in dataset_small.manifest.all_images():
        assert (dataset_small.dataset_dir / "images" / n).is_file()


# 30. every image belongs to exactly one capture group
def test_every_image_in_exactly_one_group(dataset_small):
    names = dataset_small.manifest.all_images()
    assert len(names) == len(set(names)) == 30


# 31. init_points.ply is LOCAL_METRIC
def test_init_points_frame(dataset_small):
    init = read_ply(dataset_small.dataset_dir / "init_points.ply")
    assert init.frame == "LOCAL_METRIC" and np.max(np.abs(init.xyz)) < 100


# 32. a valid dataset has no consistency issues
def test_manifest_consistency_clean(dataset_small):
    m = Manifest.load_dataset(dataset_small.dataset_dir)
    assert m.consistency_issues() == []


# 33. an intentionally leaking manifest is detected
def test_leaking_manifest_detected(dataset_small):
    m = Manifest.load_dataset(dataset_small.dataset_dir)
    leaky = m.model_copy(deep=True)
    leaky.initialization.groups = list(m.capture_groups)
    assert any("initialization uses test groups" in i for i in leaky.consistency_issues())
    assert Claim.RENDER_QUALITY not in judge(leaky).claims


# 34. reconstruction-only dataset gains no performance claim
def test_reconstruction_dataset_claims_nothing(staging_small, build_config_small, tmp_path):
    cfg = build_config_small.model_copy(
        deep=True, update={"geometry_holdout": None, "centerline": None}
    )
    cfg.split.test_every = None
    res = build_dataset(staging_small.staging_dir, tmp_path / "ds", cfg)
    j = judge(res.manifest)
    assert j.protocols == [Protocol.RECONSTRUCTION] and j.claims == [Claim.GEOMETRY_DIAGNOSTIC]
    assert res.manifest.split.test_groups == [] and res.manifest.centerline is None
    assert set(res.manifest.initialization.groups) == set(res.manifest.capture_groups)


# 35. novel-view dataset gains the render claim only when leak-free
def test_novel_view_claim_only_when_leak_free(dataset_small):
    j = judge(dataset_small.manifest)
    assert Protocol.NOVEL_VIEW in j.protocols and Claim.RENDER_QUALITY in j.claims
    leaky = dataset_small.manifest.model_copy(deep=True)
    leaky.initialization.groups = list(leaky.capture_groups)
    assert Claim.RENDER_QUALITY not in judge(leaky).claims


# 36. geometry holdout gains the metric claim only when leak-free
def test_geometry_claim_only_when_leak_free(dataset_small):
    j = judge(dataset_small.manifest)
    assert Protocol.GEOMETRY_HOLDOUT in j.protocols and Claim.GEOMETRY_ACCURACY in j.claims
    leaky = dataset_small.manifest.model_copy(deep=True)
    leaky.split.geometry_holdout.points_excluded = False
    assert Claim.GEOMETRY_ACCURACY not in judge(leaky).claims
    leaky2 = dataset_small.manifest.model_copy(deep=True)
    leaky2.initialization.excluded_chainage_ranges_m = []
    assert Claim.VOLUME_ACCURACY not in judge(leaky2).claims


# --- contract extras: provenance, transactional publication, holdout needs a centerline


def test_provenance_names_every_input(dataset_small, staging_small):
    prov = dataset_small.manifest.provenance
    paths = {a.path for a in prov.source_assets}
    assert prov.config_hash and len(prov.config_hash) == 64
    assert str(staging_small.source_placeholder) in paths
    for name in ("inventory.json", "pano_mapping.json", "extraction_manifest.json"):
        assert any(p.endswith(name) for p in paths), name
    assert any(p.endswith("scan_000.ply") for p in paths)
    assert any(p.endswith("image_000.png") for p in paths)
    assert "camera_convention.json" in paths
    assert any(p.endswith("centerline_source.csv") for p in paths)
    assert all(len(a.sha256) == 64 for a in prov.source_assets)
    resolved = json.loads((dataset_small.dataset_dir / "build_config.json").read_text())
    assert resolved["frames"]["source_mode"] == "explicit_identity"
    assert resolved["camera_convention"]["label"] == "cam(+X,-Y,-Z)"
    assert (dataset_small.dataset_dir / "camera_convention.json").is_file()


def test_build_is_transactional(staging_small, build_config_small, tmp_path, monkeypatch):
    out = tmp_path / "ds"
    import minegs.dataset.materialize as mat

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(mat, "sanity_checks", boom)
    with pytest.raises(RuntimeError):
        build_dataset(staging_small.staging_dir, out, build_config_small)
    assert not out.exists() and not (tmp_path / ".ds.minegs-partial").exists()
    monkeypatch.undo()
    res = build_dataset(staging_small.staging_dir, out, build_config_small)
    with pytest.raises(ContractError, match="--overwrite"):
        build_dataset(staging_small.staging_dir, out, build_config_small)
    # a failed overwrite keeps the previous dataset intact
    monkeypatch.setattr(mat, "sanity_checks", boom)
    with pytest.raises(RuntimeError):
        build_dataset(staging_small.staging_dir, out, build_config_small, overwrite=True)
    assert Manifest.load_dataset(out).dataset_id == res.manifest.dataset_id


def test_holdout_requires_centerline():
    with pytest.raises(Exception, match="needs a centerline"):
        DatasetBuildConfig.model_validate(
            {
                "dataset_id": "x",
                "source_frame": {"mode": "explicit_identity"},
                "camera": {"mode": "e57_pinhole", "R_e57cam_from_cam": np.eye(3).tolist()},
                "geometry_holdout": {"ranges_m": [[1.0, 2.0]]},
            }
        )


def test_no_hash_staging_is_refused(staging_small, tmp_path):
    import shutil

    from minegs.dataset.staging_input import load_staging

    shutil.copytree(staging_small.staging_dir, tmp_path / "s")
    p = tmp_path / "s" / "extraction_manifest.json"
    d = json.loads(p.read_text())
    d["source_sha256"], d["hash_skipped_reason"] = None, "requested with --no-hash"
    p.write_text(json.dumps(d))
    with pytest.raises(ContractError, match="no source SHA-256"):
        load_staging(tmp_path / "s")


def test_unregistered_staging_is_refused(staging_small, tmp_path):
    import shutil

    from minegs.dataset.staging_input import load_staging

    shutil.copytree(staging_small.staging_dir, tmp_path / "s")
    p = tmp_path / "s" / "extraction_manifest.json"
    d = json.loads(p.read_text())
    d["registration"], d["output_frame"] = "unregistered", "SCANNER"
    p.write_text(json.dumps(d))
    with pytest.raises(ContractError, match="SOURCE frame"):
        load_staging(tmp_path / "s")


def test_explicit_split_must_assign_every_group(staging_small, build_config_small, tmp_path):
    cfg = build_config_small.model_copy(
        deep=True, update={"geometry_holdout": None, "centerline": None}
    )
    cfg.split.test_every = None
    cfg.split.train_groups, cfg.split.test_groups = ["S000", "S001"], ["S002"]
    with pytest.raises(ContractError, match="unassigned"):
        build_dataset(staging_small.staging_dir, tmp_path / "ds", cfg)
