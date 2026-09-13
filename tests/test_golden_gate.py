"""Phase 0C §34 — synthetic Golden Gate: known projection, known convention."""

from __future__ import annotations

import json

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
    report = run_golden_gate(res.dataset_dir, staging_small.staging_dir, tmp_path / "gg")
    assert report["structural_result"] == "fail"
    assert any("evidence picks cam(+X,-Y,-Z)" in p for p in report["structural_problems"])


def test_select_spread_uses_position_not_index():
    pos = {
        "S003": np.array([30.0, 0, 0]),
        "S000": np.array([0.0, 0, 0]),
        "S001": np.array([90.0, 0, 0]),
        "S002": np.array([60.0, 0, 0]),
    }
    assert select_spread(pos, 3) == ["S000", "S002", "S001"]  # first, middle, last by position
    assert select_spread(pos, 2) == ["S000", "S001"]
