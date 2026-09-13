"""Phase 0C §34 — synthetic Golden Gate: known projection, known convention."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from minegs.core.frames import rot_z
from minegs.dataset.calibrate import calibrate_camera_convention, select_spread
from minegs.dataset.cameras import CameraCalibration, convention_label
from minegs.dataset.golden_gate import REPORT_FILE, TLS_SAMPLE_FILE, run_golden_gate
from minegs.dataset.staging_input import load_staging

TRUTH = "cam(+X,-Y,-Z)"


# 37. the correct convention has the low reprojection residual
def test_correct_convention_scores_best(calibration_small):
    cal, _ = calibration_small
    assert cal.status == "selected" and cal.convention.label == TRUTH
    assert cal.scoring == "rgb_residual" and cal.best_score < 25.0


# 38. x/y/z flips are clearly worse
def test_flips_score_worse(calibration_small):
    cal, _ = calibration_small
    scores = {c.label: c.rgb_residual for c in cal.candidates}
    for label in ("cam(-X,+Y,-Z)", "cam(+X,+Y,+Z)", "cam(-X,-Y,+Z)"):
        assert scores[label] > scores[TRUTH] + 20.0, label


# 39. a 90° wrong orientation is detected
def test_ninety_degree_error_detected(calibration_small):
    cal, _ = calibration_small
    scores = {c.label: c.rgb_residual for c in cal.candidates}
    truth = np.diag([1.0, -1.0, -1.0])
    for R90 in (rot_z(90), rot_z(-90)):
        label = convention_label(truth @ R90)
        assert scores[label] > scores[TRUTH] + 20.0, label
    assert cal.margin > 20.0


# 40. the selected convention is stable across spatially separated stations
def test_convention_stable_across_stations(calibration_small, golden_gate_small):
    cal, _ = calibration_small
    assert len(cal.sampled_station_ids) == 3 and len(set(cal.sampled_station_ids)) == 3
    assert all(s.best_label == TRUTH for s in cal.stations)
    report, _ = golden_gate_small
    assert all(s["best_convention"] == TRUTH for s in report["stations"])
    assert report["sampled_stations"][0] != report["sampled_stations"][-1]


# 41. overlay files are generated
def test_overlays_generated(golden_gate_small):
    report, out = golden_gate_small
    assert report["overlays"] and all((out / p).is_file() for p in report["overlays"])
    assert any(p.endswith("_depth.png") for p in report["overlays"])
    assert any(p.endswith("_tlsrgb.png") for p in report["overlays"])
    assert (out / TLS_SAMPLE_FILE).is_file()


# 42. the report round-trips
def test_report_roundtrip(golden_gate_small):
    report, out = golden_gate_small
    back = json.loads((out / REPORT_FILE).read_text())
    assert back == json.loads(json.dumps(report))
    for key in (
        "dataset_id",
        "T_tls_from_source",
        "T_tls_from_local",
        "camera_convention",
        "sampled_stations",
        "sampled_images",
        "coordinate_bounds_local_metric",
        "init_point_bounds_local_metric",
        "input_hashes",
        "structural_result",
        "real_data_validation_status",
    ):
        assert key in back, key
    assert back["structural_result"] == "pass" and back["structural_problems"] == []
    assert back["init_points_in_holdout"] == 0


# 43. input hashes and config are recorded
def test_report_records_inputs(golden_gate_small, dataset_small):
    report, _ = golden_gate_small
    h = report["input_hashes"]
    assert {
        "manifest.json",
        "build_config.json",
        "camera_convention.json",
        "init_points.ply",
        "source_e57",
    } <= set(h)
    assert all(len(v) == 64 for v in h.values())
    assert report["config_hash"] == dataset_small.manifest.provenance.config_hash


# 44. a structural pass never claims real-data validation
def test_structural_pass_is_not_real_data_validation(golden_gate_small):
    report, _ = golden_gate_small
    assert report["structural_result"] == "pass"
    assert report["real_data_validation_status"] == "pending_human_inspection"
    assert report["human_inspection_checklist"]
    text = json.dumps(report).lower()
    assert "g2 pass" not in text and "validated" not in text


# --- extras: nothing is hard-coded; ambiguity is refused; a wrong build is caught


def test_calibration_recovers_a_different_convention(tmp_path):
    from minegs.core.synthetic_staging import StagingSpec, generate_staging

    r = generate_staging(
        tmp_path / "s",
        StagingSpec(
            length_m=45,
            station_spacing_m=15,
            points_per_m=3000,
            image_size=48,
            R_e57cam_from_cam=tuple(map(tuple, np.eye(3))),
        ),
    )
    cal = calibrate_camera_convention(load_staging(r.staging_dir))
    assert cal.status == "selected" and cal.convention.label == "cam(+X,+Y,+Z)"


def test_ambiguous_calibration_is_refused_by_the_builder(calibration_small, tmp_path):
    cal, _ = calibration_small
    amb = cal.model_copy(
        update={"status": "ambiguous", "convention": None, "notes": ["margin too small"]}
    )
    p = tmp_path / "amb.json"
    amb.save(p)
    from minegs.core.errors import ContractError

    with pytest.raises(ContractError, match="ambiguous"):
        CameraCalibration.load(p).require_selected(str(p))


def test_golden_gate_catches_a_wrong_explicit_convention(
    staging_small, build_config_small, tmp_path
):
    from minegs.dataset.materialize import build_dataset

    cfg = build_config_small.model_copy(
        deep=True, update={"geometry_holdout": None, "centerline": None}
    )
    cfg.camera.convention_file = None
    cfg.camera.R_e57cam_from_cam = np.eye(3).tolist()  # declared, wrong
    res = build_dataset(staging_small.staging_dir, tmp_path / "ds", cfg)
    from minegs.core.errors import ContractError

    # the gate fails loudly — but only after the diagnostics are on disk
    with pytest.raises(ContractError, match="structural_result=fail"):
        run_golden_gate(res.dataset_dir, staging_small.staging_dir, tmp_path / "gg")
    report = json.loads((tmp_path / "gg" / REPORT_FILE).read_text())
    assert report["structural_result"] == "fail" and report["overlays"]
    assert any("evidence picks cam(+X,-Y,-Z)" in p for p in report["structural_problems"])
    same = run_golden_gate(
        res.dataset_dir, staging_small.staging_dir, tmp_path / "gg2", raise_on_fail=False
    )
    assert same["structural_result"] == "fail"


def test_select_spread_uses_position_not_index():
    pos = {
        "S003": np.array([30.0, 0, 0]),
        "S000": np.array([0.0, 0, 0]),
        "S001": np.array([90.0, 0, 0]),
        "S002": np.array([60.0, 0, 0]),
    }
    assert select_spread(pos, 3) == ["S000", "S002", "S001"]  # first, middle, last by position
    assert select_spread(pos, 2) == ["S000", "S001"]


# --- review round 2: source-bound calibration, >= 3 stations, RGB required, exact inputs


def _staging(tmp_path, name, **spec):
    from minegs.core.synthetic_staging import StagingSpec, generate_staging

    base = {"points_per_m": 2000, "image_size": 40}
    base.update(spec)
    return generate_staging(tmp_path / name, StagingSpec(**base))


def test_calibration_from_another_source_is_refused(staging_small, build_config_small, tmp_path):
    """BLOCKER 2: a valid artifact measured on a different survey is evidence about that survey."""
    from minegs.core.errors import ContractError
    from minegs.dataset.materialize import build_dataset

    other = _staging(tmp_path, "other", length_m=45, station_spacing_m=15, seed=7)
    cal = calibrate_camera_convention(load_staging(other.staging_dir))
    assert (
        cal.status == "selected"
        and cal.source_sha256 != load_staging(staging_small.staging_dir).source_sha256
    )
    foreign = tmp_path / "foreign_convention.json"
    cal.save(foreign)
    cfg = build_config_small.model_copy(
        deep=True, update={"geometry_holdout": None, "centerline": None}
    )
    cfg.camera.convention_file = str(foreign)
    with pytest.raises(ContractError, match="measured on source"):
        build_dataset(staging_small.staging_dir, tmp_path / "ds", cfg)
    assert not (tmp_path / "ds").exists()


@pytest.mark.parametrize("length_m, n_expected", [(20.0, 1), (30.0, 2)])
def test_fewer_than_three_stations_cannot_select(tmp_path, length_m, n_expected):
    """BLOCKER 3: one or two stations never yield a calibrated convention."""
    from minegs.core.errors import ContractError

    r = _staging(tmp_path, "few", length_m=length_m, station_spacing_m=15)
    assert len(r.station_poses) == n_expected
    cal = calibrate_camera_convention(load_staging(r.staging_dir))
    assert cal.status == "insufficient" and cal.convention is None
    assert len(cal.sampled_station_ids) == n_expected
    p = tmp_path / "cal.json"
    cal.save(p)
    with pytest.raises(ContractError, match="insufficient"):
        CameraCalibration.load(p).require_selected(str(p))


def test_requesting_fewer_than_three_stations_is_refused(staging_small):
    from minegs.core.errors import ContractError

    tree = load_staging(staging_small.staging_dir)
    for n in (1, 2):
        with pytest.raises(ContractError, match="at least 3"):
            calibrate_camera_convention(tree, n_stations=n)


def test_scan_without_rgb_is_refused_for_calibration(staging_small, tmp_path):
    """MAJOR 3: there is no colour-free scoring that can clear the threshold, so refuse."""
    import shutil

    from minegs.core.errors import ContractError
    from minegs.core.pointcloud import PointCloud, read_ply, write_ply
    from minegs.core.provenance import sha256_file

    shutil.copytree(staging_small.staging_dir, tmp_path / "s")
    em_path = tmp_path / "s" / "extraction_manifest.json"
    em = json.loads(em_path.read_text())
    for so in em["scan_outputs"]:
        ply = tmp_path / "s" / "scans" / Path(so["path"]).name
        pc = read_ply(ply)
        write_ply(PointCloud(pc.xyz, None, frame="SOURCE"), ply, xyz_dtype="f8")
        so["sha256"], so["attributes"] = sha256_file(ply), []
    em_path.write_text(json.dumps(em))
    with pytest.raises(ContractError, match="no RGB"):
        calibrate_camera_convention(load_staging(tmp_path / "s"))


def test_golden_gate_refuses_a_reextracted_tree(dataset_small, staging_small, tmp_path):
    """MAJOR 2: same E57, different bytes — a consistent but different tree is not the input."""
    import shutil

    from minegs.core.errors import ContractError
    from minegs.core.provenance import sha256_file

    shutil.copytree(staging_small.staging_dir, tmp_path / "s")
    scan = next((tmp_path / "s" / "scans").glob("scan_000.ply"))
    with open(scan, "ab") as f:
        f.write(b"\0")  # a trailing byte a PLY reader ignores, a digest does not
    em_path = tmp_path / "s" / "extraction_manifest.json"
    em = json.loads(em_path.read_text())
    for so in em["scan_outputs"]:
        if so["scan_id"] == "scan_000":
            so["sha256"] = sha256_file(scan)
    em_path.write_text(json.dumps(em))
    load_staging(tmp_path / "s")  # internally consistent, and still not the build's input
    with pytest.raises(ContractError, match=r"not the (file|artifact) this dataset was built from"):
        run_golden_gate(dataset_small.dataset_dir, tmp_path / "s", tmp_path / "gg")
    # scan bytes changed with the artifacts left intact: caught at consumption
    shutil.copytree(staging_small.staging_dir, tmp_path / "s3")
    with open(tmp_path / "s3" / "scans" / "scan_000.ply", "ab") as f:
        f.write(b"\0")
    with pytest.raises(ContractError, match="does not match the digest"):
        run_golden_gate(dataset_small.dataset_dir, tmp_path / "s3", tmp_path / "gg3")
    # an internally consistent tree whose artifacts were re-written after the build: the
    # cross-artifact check is happy, the provenance digests are not
    shutil.copytree(staging_small.staging_dir, tmp_path / "s2")
    for name in ("pano_mapping.json", "extraction_manifest.json"):
        f = tmp_path / "s2" / name
        f.write_text(f.read_text().replace("Skybox 0", "Skybox 9"))
    load_staging(tmp_path / "s2")  # consistent
    with pytest.raises(ContractError, match="not the artifact this dataset was built from"):
        run_golden_gate(dataset_small.dataset_dir, tmp_path / "s2", tmp_path / "gg2")
