"""Phase 0C §33 — leakage and dataset contract."""

from __future__ import annotations

import json
from pathlib import Path

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


# --- review round 2: bytes actually consumed, holdout extent, safe overwrite


def _copy_staging(staging_small, tmp_path, name="s"):
    import shutil

    shutil.copytree(staging_small.staging_dir, tmp_path / name)
    return tmp_path / name


def _no_holdout(cfg):
    return cfg.model_copy(deep=True, update={"geometry_holdout": None, "centerline": None})


def test_modified_scan_bytes_are_refused(staging_small, build_config_small, tmp_path):
    """BLOCKER 1: provenance must describe the bytes consumed, so a changed scan stops the build."""
    stg = _copy_staging(staging_small, tmp_path)
    with open(stg / "scans" / "scan_001.ply", "r+b") as f:
        f.seek(-8, 2)
        f.write(b"\xff" * 8)  # corrupt the last vertex in place
    with pytest.raises(ContractError, match=r"scan scan_001 .* does not match the digest"):
        build_dataset(stg, tmp_path / "ds", _no_holdout(build_config_small))
    assert not (tmp_path / "ds").exists() and not (tmp_path / ".ds.minegs-partial").exists()


def test_modified_image_bytes_are_refused(staging_small, build_config_small, tmp_path):
    stg = _copy_staging(staging_small, tmp_path)
    img = stg / "images" / "image_004.png"
    from PIL import Image as PILImage

    im = PILImage.open(img).convert("RGB")
    im.transpose(PILImage.Transpose.FLIP_LEFT_RIGHT).save(img)  # a plausible-looking edit
    with pytest.raises(ContractError, match=r"image image_004 .* does not match the digest"):
        build_dataset(stg, tmp_path / "ds", _no_holdout(build_config_small))


def test_provenance_records_the_verified_digest(dataset_small, staging_small):
    from minegs.core.provenance import sha256_file

    by_name = {Path(a.path).name: a.sha256 for a in dataset_small.manifest.provenance.source_assets}
    for name in ("scan_000.ply", "scan_003.ply"):
        assert by_name[name] == sha256_file(staging_small.staging_dir / "scans" / name)
    for name in ("image_000.png", "image_017.png"):
        assert by_name[name] == sha256_file(staging_small.staging_dir / "images" / name)


def test_contradictory_mapping_reports_are_refused(staging_small, tmp_path):
    from minegs.dataset.staging_input import load_staging

    stg = _copy_staging(staging_small, tmp_path)
    pm = stg / "pano_mapping.json"
    rep = json.loads(pm.read_text())
    # image_008 belongs to scan_001; the standalone report now claims otherwise
    rec = next(m for m in rep["mappings"] if m["image_id"] == "image_008")
    assert rec["scan_id"] == "scan_001"
    rec["scan_id"], rec["station_id"] = "scan_000", "S000"
    pm.write_text(json.dumps(rep))
    with pytest.raises(ContractError, match="contradict each other"):
        load_staging(stg)


@pytest.mark.parametrize("ranges, side", [([(-3.0, 4.0)], "lower"), ([(55.0, 64.0)], "upper")])
def test_holdout_must_lie_fully_inside_the_centerline(
    staging_small, build_config_small, tmp_path, ranges, side
):
    """BLOCKER 4: a range overhanging the line would claim chainage the survey does not have."""
    cfg = build_config_small.model_copy(deep=True)
    cfg.geometry_holdout.ranges_m = ranges
    with pytest.raises(ContractError, match="not fully inside the centerline extent"):
        build_dataset(staging_small.staging_dir, tmp_path / "ds", cfg)
    # exactly on the ends is inside
    ext = (0.0, 60.0)
    cfg.geometry_holdout.ranges_m = (
        [(ext[0], ext[0] + 3.0)] if side == "lower" else [(ext[1] - 3.0, ext[1])]
    )
    build_dataset(staging_small.staging_dir, tmp_path / "ok", cfg)


def test_overwrite_never_deletes_a_foreign_directory(staging_small, build_config_small, tmp_path):
    """MAJOR 4: --overwrite replaces only a dataset this tool wrote, holding nothing else."""
    cfg = _no_holdout(build_config_small)
    foreign = tmp_path / "photos"
    foreign.mkdir()
    (foreign / "holiday.jpg").write_bytes(b"precious")
    with pytest.raises(ContractError, match="does not own"):
        build_dataset(staging_small.staging_dir, foreign, cfg, overwrite=True)
    assert (foreign / "holiday.jpg").read_bytes() == b"precious"
    out = tmp_path / "ds"
    build_dataset(staging_small.staging_dir, out, cfg)
    (out / "notes.txt").write_text("field notes")
    with pytest.raises(ContractError, match="did not write"):
        build_dataset(staging_small.staging_dir, out, cfg, overwrite=True)
    assert (out / "notes.txt").exists()
    (out / "notes.txt").unlink()
    build_dataset(staging_small.staging_dir, out, cfg, overwrite=True)
    assert Manifest.load_dataset(out).dataset_id == cfg.dataset_id


# --- review round 3: camera-critical cross-artifact agreement, recursive overwrite ownership


def _edit_standalone(staging_small, tmp_path, mutate):
    """Copy a staging tree and change only pano_mapping.json."""
    stg = _copy_staging(staging_small, tmp_path)
    pm = stg / "pano_mapping.json"
    rep = json.loads(pm.read_text())
    mutate(rep)
    pm.write_text(json.dumps(rep))
    return stg


def _asset(rep, image_id="image_005"):
    return next(a for a in rep["images"] if a["image_id"] == image_id)


@pytest.mark.parametrize(
    ("mutate", "expect"),
    [
        # BLOCKER: camera-critical ImageAsset content, which 0C reads from the standalone file
        (
            lambda r: _asset(r)["vendor_metadata"].__setitem__("focalLength", 0.009),
            "vendor_metadata",
        ),
        (
            lambda r: _asset(r)["vendor_metadata"].__setitem__("pose_translation", [1.0, 2.0, 3.0]),
            "vendor_metadata",
        ),
        (lambda r: _asset(r).__setitem__("representation", "spherical"), "representation"),
        (lambda r: _asset(r).__setitem__("width", 999), "width"),
        (lambda r: _asset(r).__setitem__("height", 999), "height"),
        (lambda r: _asset(r).__setitem__("sha256", "0" * 64), "sha256"),
        (lambda r: _asset(r).__setitem__("source_index", 99), "source_index"),
        (lambda r: r.__setitem__("scan_count", 999), "scan_count"),
        (lambda r: r.__setitem__("image_root_sha256", "f" * 64), "image_root_sha256"),
    ],
)
def test_camera_metadata_must_agree_across_the_two_artifacts(
    staging_small, tmp_path, mutate, expect
):
    """A field-by-field allowlist would go stale; the whole record is compared."""
    from minegs.dataset.staging_input import load_staging

    stg = _edit_standalone(staging_small, tmp_path, mutate)
    with pytest.raises(ContractError, match="contradict each other") as e:
        load_staging(stg)
    assert expect in str(e.value)


def test_cross_artifact_comparison_is_symmetric(staging_small, tmp_path):
    """An extra record on either side is detected, not just a differing shared one."""
    from minegs.dataset.staging_input import load_staging

    dropped = _edit_standalone(staging_small, tmp_path / "drop", lambda r: r["images"].pop(3))
    with pytest.raises(ContractError, match=r"missing from pano_mapping\.json"):
        load_staging(dropped)

    def add(r):
        extra = json.loads(json.dumps(r["images"][0]))
        extra["image_id"] = "image_999"
        r["images"].append(extra)

    added = _edit_standalone(staging_small, tmp_path / "add", add)
    with pytest.raises(ContractError, match=r"missing from extraction_manifest\.json"):
        load_staging(added)

    def drop_mapping(r):
        r["mappings"] = [m for m in r["mappings"] if m["image_id"] != "image_002"]

    m_dropped = _edit_standalone(staging_small, tmp_path / "mdrop", drop_mapping)
    with pytest.raises(ContractError, match=r"mapping \['image_002'\] missing from pano_mapping"):
        load_staging(m_dropped)


def test_an_untouched_tree_still_loads(staging_small, tmp_path):
    """The strict comparison must not reject the extractor's own output."""
    from minegs.dataset.staging_input import load_staging

    assert load_staging(_copy_staging(staging_small, tmp_path)).source_sha256


def test_overwrite_detects_a_nested_foreign_file(staging_small, build_config_small, tmp_path):
    """MAJOR: --overwrite removes the whole tree, so a stray file under images/ is at risk too."""
    from minegs.dataset.materialize import owned_relpaths

    cfg = _no_holdout(build_config_small)
    out = tmp_path / "ds"
    res = build_dataset(staging_small.staging_dir, out, cfg)
    # a freshly built dataset is entirely owned — nothing the builder writes reads as foreign
    actual = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    assert actual - owned_relpaths(out) == set()

    nested = out / "images" / "field_notes.txt"
    nested.write_text("survey notes nobody backed up")
    with pytest.raises(ContractError, match=r"images/field_notes\.txt"):
        build_dataset(staging_small.staging_dir, out, cfg, overwrite=True)
    assert nested.read_text() == "survey notes nobody backed up"
    nested.unlink()

    deep = out / "sparse" / "0" / "extra" / "notes.md"
    deep.parent.mkdir()
    deep.write_text("x")
    with pytest.raises(ContractError, match="did not write"):
        build_dataset(staging_small.staging_dir, out, cfg, overwrite=True)
    assert deep.exists()
    deep.unlink()
    deep.parent.rmdir()

    # ...and a clean generated dataset is still replaceable
    again = build_dataset(staging_small.staging_dir, out, cfg, overwrite=True)
    assert again.manifest.dataset_id == res.manifest.dataset_id
    assert Manifest.load_dataset(out).all_images() == res.manifest.all_images()


def test_overwrite_refuses_a_dataset_whose_manifest_cannot_be_read(
    staging_small, build_config_small, tmp_path
):
    cfg = _no_holdout(build_config_small)
    out = tmp_path / "ds"
    build_dataset(staging_small.staging_dir, out, cfg)
    (out / "manifest.json").write_text("{ not json")
    with pytest.raises(ContractError, match="cannot be read as a dataset"):
        build_dataset(staging_small.staging_dir, out, cfg, overwrite=True)
    assert (out / "init_points.ply").exists()
