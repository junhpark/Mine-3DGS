"""Phase 3 — the synthetic image/360 end-to-end structural gate (§Phase 3 C4).

One drift, reconstructed twice. The scanner path is the Phase 2 chain: a genuine E57, the real
extractor, the real dataset builder and its golden gate. The image path never sees the scanner:
panoramas of the same tunnel go through the real selection, the real ring crop, the real mask
writer and the real frame-set digest checks, are reconstructed by a stand-in SfM into a frame
whose scale nothing has measured, are brought into TLS_GLOBAL by the real registration against
survey control, and are materialised by the real image-only dataset builder — which re-derives
that the initialisation is the reconstruction's own points rather than believing the manifest.
Both then run the same training, depth, surface, geometry, sections and volume stages, and the
two are compared over the domain both of them observed.

Three things are substituted, each for want of hardware, each recorded: the trainer, the depth
renderer, and — on the image path — SfM and the frame decoder. **This is not G2.** No number
below is evidence about a mine, and `real_data_validation_status` stays
`pending_human_inspection` on the image-only gate because a person has not looked yet.

T1–T30 are the adversarial cases the phase directive names. Each breaks something a
well-meaning implementation would let through, on this workflow rather than on a fixture built
to make the point.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from minegs.core.pointcloud import PointCloud, read_ply, write_ply
from minegs.core.synthetic_staging import StagingSpec, generate_staging
from minegs.dataset.from_sfm import PROVENANCE_DIR, SfmDatasetConfig
from minegs.e2e.models import Phase3Inputs, Stage
from minegs.e2e.phase3 import phase3_specs
from minegs.e2e.report import build_report
from minegs.e2e.runner import E2EConfig, Workflow
from minegs.ingest.video.build import copy_extractor

from e57_survey import write_survey_e57
from test_e2e import stand_in_trainer
from video_survey import RING, TunnelDepthRenderer, generate_video_survey, stand_in_sfm

pye57 = pytest.importorskip("pye57", reason="the comparison needs a scanner survey to compare to")

GATE_SPEC = StagingSpec(length_m=60.0, station_spacing_m=12.0, points_per_m=3000, image_size=64)
HOLDOUT = (20.0, 26.0)
DATASET_ID = "syn_p3_gate"
THRESHOLDS = {"max_rmse_m": 0.10, "min_inlier_ratio": 0.6, "min_correspondences": 6}


# ---------------------------------------------------------------- the survey, both ways


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """One tunnel: its points, its design axis, the E57 of it and the held-out TLS cloud."""
    root = tmp_path_factory.mktemp("p3_scene")
    result = generate_staging(root / "synthetic", GATE_SPEC)
    e57 = write_survey_e57(result.staging_dir, root / "raw" / "gate_survey.e57")
    centerline_csv = root / "centerline_source.csv"
    result.centerline_source.to_csv(centerline_csv)
    tls = write_ply(
        PointCloud(result.points_source.xyz, rgb=result.points_source.rgb, frame="TLS_GLOBAL"),
        root / "raw" / "tls_full.ply",
        xyz_dtype="f8",
    )
    survey = generate_video_survey(
        root / "video", result.centerline_source, result.points_source.xyz
    )
    return SimpleNamespace(
        root=root,
        scene=result,
        e57=e57,
        centerline_csv=centerline_csv,
        tls=tls,
        survey=survey,
    )


def phase3_config(scene, root: Path, **over) -> E2EConfig:
    dataset = SfmDatasetConfig(
        dataset_id=over.pop("dataset_id", DATASET_ID),
        source="video360",
        geometry_holdout_m=over.pop("holdout", [HOLDOUT]),
        centerline_file=str(scene.centerline_csv),
        centerline_source="design",
        init_voxel_m=0.05,
        init_max_points=120_000,
        sparse_max_points=20_000,
        test_groups=over.pop("test_groups", []),
    )
    p3 = Phase3Inputs(
        kind="video360",
        video=str(scene.survey.video),
        frameset_dir=str(root / "frameset"),
        sfm_dir=str(root / "sfm"),
        registration_dir=str(root / "registration"),
        blur_threshold=over.pop("blur_threshold", 1.0),
        hamming_threshold=over.pop("hamming_threshold", 2),
        ring=RING,
        pano_convention=scene.survey.convention,
        nadir_el_deg=over.pop("nadir_el_deg", -35.0),
        basis=over.pop("basis", "known_target"),
        targets_sfm=str(scene.survey.targets_sfm),
        targets_tls=str(scene.survey.targets_tls),
        target_ranges_m=over.pop("target_ranges_m", scene.survey.control_ranges_m),
        icp_target_ply=over.pop("icp_target_ply", None),
        icp_ranges_m=over.pop("icp_ranges_m", None),
        icp_target_is_whole_reference=over.pop("icp_whole", False),
        registration_thresholds=over.pop("thresholds", dict(THRESHOLDS)),
        dataset=dataset,
    )
    return E2EConfig(
        dataset_dir=str(root / "dataset"),
        tls_reference_ply=str(scene.tls),
        phase3=p3,
        stride=2,
        interval_m=1.0,
        thickness_m=0.5,
        angle_bins=72,
        max_dist_m=1.0,
        **over,
    )


def run_phase3(scene, root: Path, *, through: Stage = Stage.REPORT, **over):
    cfg = phase3_config(scene, root, **over)
    wf = Workflow(root / "wf", cfg)
    local = read_ply(scene.tls).xyz  # moved into LOCAL_METRIC once the dataset exists
    state = wf.execute(
        phase3_specs(),
        through=Stage.DATASET if through is Stage.DATASET else through,
        sfm=stand_in_sfm(scene.survey),
        frame_extractor=copy_extractor(scene.survey.frames_dir),
        trainer=stand_in_trainer,
        renderer=_renderer_for(wf, local, through),
    )
    return SimpleNamespace(root=root, wf=wf, state=state, config=cfg)


def _renderer_for(wf: Workflow, tls_xyz: np.ndarray, through: Stage):
    """A renderer bound to the reference cloud in the dataset's own metric frame.

    Built lazily: the origin it subtracts is decided by the dataset stage, which has not run
    yet when the workflow is launched.
    """

    class _Lazy(TunnelDepthRenderer):
        def __init__(self) -> None:
            super().__init__(np.zeros((0, 3)))

        def render(self, checkpoint, cameras, images, expected_step=None):
            from minegs.core.manifest import Manifest

            ds = Path(wf.state.stages[Stage.DATASET].outputs["dataset_dir"])
            m = Manifest.load_dataset(ds, strict_layout=False)
            self.points = tls_xyz - np.asarray(m.T_tls_from_local.t)
            return super().render(checkpoint, cameras, images, expected_step)

    return _Lazy()


@pytest.fixture(scope="module")
def gate(scene, tmp_path_factory):
    """The image-only chain, from a video file, driven by the workflow the CLI drives."""
    return run_phase3(scene, tmp_path_factory.mktemp("p3_gate"))


# ---------------------------------------------------------------- the chain ran, and says so


def outputs(state, stage: Stage) -> dict:
    return dict(state.stages[stage].outputs)


def test_every_stage_succeeded_on_the_image_only_path(gate):
    for stage in Stage:
        rec = gate.state.stages[stage]
        assert rec.usable, f"{stage.value}: {rec.status.value} {rec.failure_reason}"


def test_the_reconstruction_is_the_survey_and_says_which_parts_were_not_real(gate):
    ing = outputs(gate.state, Stage.INGEST)
    assert ing["mode"] == "image_reconstruction"
    assert ing["source_sha256"] and ing["source_file_name"] == "drift_360.mp4"
    assert ing["frame_extraction_real"] is False  # copy_extractor, not a decoder
    assert ing["real_sfm_execution"] is False  # stand-in, not COLMAP
    assert ing["sfm_metric_state"] == "arbitrary_scale"
    assert ing["output_frame"] == "TLS_GLOBAL"
    assert ing["n_images"] == ing["n_frames_kept"] * RING.n_yaw
    assert ing["n_masks"] == ing["n_images"]


def test_registration_recovers_the_scale_nothing_told_it(gate):
    from video_survey import T_SFM_FROM_TLS

    ing = outputs(gate.state, Stage.INGEST)
    assert ing["registration_basis"] == "known_target"
    assert ing["registration_scale"] == pytest.approx(1.0 / T_SFM_FROM_TLS.s, rel=2e-3)
    assert ing["registration_claim_allowed"] is True
    assert ing["registration_claim_refusals"] == []


def test_the_dataset_is_image_only_and_passed_its_own_gate(gate):
    ds = outputs(gate.state, Stage.DATASET)
    assert ds["init_source"] == "sfm_sparse"
    assert ds["golden_gate_kind"] == "image_sfm_registered"
    assert ds["golden_gate_passed"] is True
    assert ds["real_data_validation_status"] == "pending_human_inspection"
    assert "volume_accuracy" in ds["claims"]
    assert ds["geometry_holdout_ranges_m"] == [list(HOLDOUT)]


def test_the_holdout_is_absent_from_the_initialisation_not_only_declared(gate):
    ds = Path(outputs(gate.state, Stage.DATASET)["dataset_dir"])
    prov = json.loads((ds / PROVENANCE_DIR / "init_provenance.json").read_text())
    assert prov["n_holdout_points_excluded"] > 0

    from minegs.core.centerline import Centerline
    from minegs.core.manifest import Manifest

    m = Manifest.load_dataset(ds, strict_layout=False)
    cl = Centerline.from_csv(ds / m.centerline.file, "TLS_GLOBAL", m.centerline.source)
    init = read_ply(ds / m.initialization.file)
    s, _ = cl.project(init.xyz + np.asarray(m.T_tls_from_local.t))
    lo, hi = HOLDOUT
    assert not ((s > lo) & (s < hi)).any()


def test_the_report_says_what_was_substituted_and_claims_nothing(gate):
    report = build_report(gate.state)
    assert report.maturity.structural_status == "implemented_and_structurally_tested"
    assert report.maturity.real_data_validation_status == "not_validated"
    assert report.maturity.human_visual_review_status == "pending"
    notes = " ".join(report.maturity.notes)
    assert "SfM was substituted" in notes
    assert "frame extraction was substituted" in notes
    assert report.training.real_gpu_execution is False
    assert report.reconstruction.real_renderer_execution is False
    meta = report.source.capture_metadata
    assert meta["real_sfm_execution"] is False
    assert meta["registration_basis"] == "known_target"


def test_volume_was_computed_over_the_holdout_on_both_series(gate):
    sv = outputs(gate.state, Stage.SECTIONS_VOLUME)
    assert sv["requested_ranges_m"] == [list(HOLDOUT)]
    assert sv["paired_valid_count"] > 0
    assert sv["predicted_volume_m3"] is not None
    assert sv["reference_volume_m3"] is not None


# ---------------------------------------------------------------- the same drift, with a scanner


def tls_build_config(scene, root: Path) -> Path:
    """The scanner path's build config: same tunnel, same axis, same holdout."""
    spec = {
        "dataset_id": "syn_p3_tls",
        "source_frame": {"mode": "explicit_identity", "note": "synthetic survey frame"},
        "camera": {"mode": "e57_pinhole", "convention_file": str(root / "convention.json")},
        "split": {"test_every": 3},
        "centerline": {"file": str(scene.centerline_csv), "frame": "SOURCE"},
        "geometry_holdout": {"ranges_m": [list(HOLDOUT)]},
        "initialization": {"voxel_m": 0.05, "max_points": 120_000, "sparse_max_points": 20_000},
        "capture_epoch": {"id": "ep1"},
    }
    path = root / "build_config.json"
    path.write_text(json.dumps(spec, indent=2))
    return path


@pytest.fixture(scope="module")
def tls_gate(scene, tmp_path_factory):
    """The scanner path over the same drift, so there is something to compare against."""
    from minegs.e2e.stages import default_specs

    root = tmp_path_factory.mktemp("p3_tls")
    cfg = E2EConfig(
        source_e57=str(scene.e57),
        staging_dir=str(root / "staging"),
        build_config=str(tls_build_config(scene, root)),
        dataset_dir=str(root / "dataset"),
        tls_reference_ply=str(scene.tls),
        stride=2,
        interval_m=1.0,
        thickness_m=0.5,
        angle_bins=72,
        max_dist_m=1.0,
    )
    wf = Workflow(root / "wf", cfg)
    state = wf.execute(
        default_specs(),
        through=Stage.REPORT,
        renderer=_renderer_for(wf, read_ply(scene.tls).xyz, Stage.REPORT),
        trainer=stand_in_trainer,
    )
    return SimpleNamespace(root=root, wf=wf, state=state, config=cfg)


def comparison(gate, tls_gate):
    from minegs.eval.compare import compare_paths, path_context, require_same_holdout
    from minegs.eval.sections import load_section_input

    def side(g):
        sv = outputs(g.state, Stage.SECTIONS_VOLUME)
        ctx = path_context(
            outputs(g.state, Stage.DATASET)["dataset_dir"],
            e2e_report=outputs(g.state, Stage.REPORT)["report_json_path"],
        )
        pred, _ = load_section_input(sv["sections_predicted_path"])
        ref, _ = load_section_input(sv["sections_reference_path"])
        return ctx, pred, ref

    tls_ctx, tls_pred, tls_ref = side(tls_gate)
    img_ctx, img_pred, img_ref = side(gate)
    return compare_paths(
        tls_pred=tls_pred,
        tls_ref=tls_ref,
        image_pred=img_pred,
        image_ref=img_ref,
        ranges=require_same_holdout(tls_ctx, img_ctx),
        tls_context=tls_ctx,
        image_context=img_ctx,
        comparison_id="p3-gate",
    )


def test_the_two_paths_are_compared_over_the_domain_both_observed(gate, tls_gate):
    rep = comparison(gate, tls_gate)
    assert rep.requested_intervals_m == [HOLDOUT]
    assert rep.common_length_m > 0
    for side in (rep.tls_assisted, rep.image_only):
        assert side.paired.volume.integrated_intervals_m == rep.common_intervals_m
    assert rep.volume_difference_m3 is not None
    assert rep.section_median_difference_m2 is not None
    assert rep.image_only.source == "video360"
    assert rep.image_only.initialization_source == "sfm_sparse"
    assert rep.tls_assisted.initialization_source == "tls"


def test_the_comparison_refuses_to_be_read_as_a_measurement(gate, tls_gate):
    rep = comparison(gate, tls_gate)
    assert rep.real_execution is False
    assert rep.image_only.execution["real_sfm_execution"] is False
    assert rep.image_only.execution["real_frame_extraction"] is False
    assert rep.tls_assisted.execution["real_gpu_execution"] is False
    assert "NOT VALIDATED" in rep.maturity_statement
    assert "PENDING" in rep.maturity_statement
    assert any("structural evidence" in n for n in rep.notes)
