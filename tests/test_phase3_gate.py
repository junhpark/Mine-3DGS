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
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from minegs.core.errors import ContractError
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
        # The gate keeps the stricter setting so the build-time enforcement is exercised; the
        # library default is the weaker declaration (§T35).
        holdout_images_excluded=over.pop("images_excluded", True),
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
        targets_sfm=over.pop("targets_sfm", str(scene.survey.targets_sfm)),
        targets_tls=over.pop("targets_tls", str(scene.survey.targets_tls)),
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
        through=through,
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


# ================================================================ T1-T30, the adversarial set
#
# Each of these breaks something a well-meaning implementation would let through. Where it can
# be done on the gate's own artifacts it is, on a copy, because a refusal proven on a fixture
# built to be refused proves less than one proven on the thing that actually ran.


@pytest.fixture
def frameset(gate, tmp_path):
    src = Path(outputs(gate.state, Stage.INGEST)["frameset_dir"])
    dst = tmp_path / "frameset"
    shutil.copytree(src, dst)
    from minegs.ingest.video.models import check_frameset, load_frameset

    rec, root = load_frameset(dst)
    return SimpleNamespace(rec=rec, dir=root, check=lambda: check_frameset(*load_frameset(dst)))


@pytest.fixture
def sfm_copy(gate, tmp_path):
    src = Path(outputs(gate.state, Stage.INGEST)["sfm_dir"])
    dst = tmp_path / "sfm"
    shutil.copytree(src, dst)
    return dst


@pytest.fixture
def dataset_copy(gate, tmp_path):
    src = Path(outputs(gate.state, Stage.DATASET)["dataset_dir"])
    dst = tmp_path / "dataset"
    shutil.copytree(src, dst)
    return dst


def repaint(path: Path) -> None:
    """Change one pixel. The file is still a valid image and is no longer the same evidence."""
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB")).copy()
    arr[0, 0] = (255 - arr[0, 0, 0], 0, 0)
    Image.fromarray(arr).save(path)


# ---------------------------------------------------------------- T1-T2: the frame boundary


def test_t1_sfm_internal_cannot_be_declared_into_tls_global():
    from minegs.dataset.build_config import SourceFrameConfig
    from minegs.dataset.frames import resolve_tls_from_source

    cfg = SourceFrameConfig(mode="explicit_identity", note="whatever the operator believes")
    resolve_tls_from_source(cfg, "SOURCE")  # the scanner door still opens
    with pytest.raises(ContractError, match="cannot promote SFM_INTERNAL"):
        resolve_tls_from_source(cfg, "SFM_INTERNAL")


def test_t2_lengths_are_refused_while_the_reconstruction_is_unitless():
    from minegs.core.errors import FrameError
    from minegs.core.frames import require_metric_frame

    require_metric_frame("TLS_GLOBAL", "the reference")
    require_metric_frame("LOCAL_METRIC", "the surface")
    with pytest.raises(FrameError, match="unitless"):
        require_metric_frame("SFM_INTERNAL", "the reconstruction")


# ---------------------------------------------------------------- T3-T6: the frame set


def test_t3_a_rejected_frame_cannot_reach_the_reconstruction(frameset):
    rejected = [d.name for d in frameset.rec.selection.decisions if not d.keep]
    kept = {Path(i.name).stem for i in frameset.rec.images}
    for name in rejected:
        assert Path(name).stem not in {k.rsplit("_", 1)[0] for k in kept}
    # And the layout, not only the record: `images/` holds crops of kept frames and nothing else.
    parents = {c.parent_frame for c in frameset.rec.crops}
    assert parents == {d.name for d in frameset.rec.selection.decisions if d.keep}


def test_t4_an_image_edited_after_the_record_is_refused(frameset):
    repaint(frameset.dir / "images" / frameset.rec.images[0].name)
    with pytest.raises(ContractError):
        frameset.check()


def test_t5_an_image_added_after_the_record_is_refused(frameset):
    first = frameset.dir / "images" / frameset.rec.images[0].name
    shutil.copy2(first, first.parent / "smuggled.png")
    with pytest.raises(ContractError):
        frameset.check()


def test_t6_a_frame_set_never_writes_into_a_directory_that_is_not_its_own(tmp_path):
    from minegs.ingest.video.build import prepare_frameset_dir

    out = tmp_path / "fs"
    out.mkdir()
    (out / "thesis.tex").write_text("not ours")
    with pytest.raises(ContractError, match="did not write"):
        prepare_frameset_dir(out, overwrite=True)
    (out / "thesis.tex").unlink()
    (out / "frameset.json").write_text("{}")
    with pytest.raises(ContractError, match="already holds a frame set"):
        prepare_frameset_dir(out, overwrite=False)


# ---------------------------------------------------------------- T7-T9: the 360 path


def test_t7_a_360_frame_set_without_a_ring_is_refused(scene, tmp_path):
    from minegs.ingest.video.build import build_frameset, copy_extractor

    with pytest.raises(ContractError, match="ring crop spec"):
        build_frameset(
            tmp_path / "fs",
            kind="video360",
            video=scene.survey.video,
            pano_convention=scene.survey.convention,
            extractor=copy_extractor(scene.survey.frames_dir),
        )


def test_t8_a_360_frame_set_will_not_borrow_the_scanners_panorama_convention(scene, tmp_path):
    from minegs.ingest.common.geometry import PanoConvention
    from minegs.ingest.video.build import build_frameset, copy_extractor

    common = dict(
        kind="video360",
        video=scene.survey.video,
        ring=RING,
        extractor=copy_extractor(scene.survey.frames_dir),
    )
    with pytest.raises(ContractError, match="explicit panorama convention"):
        build_frameset(tmp_path / "a", **common)
    with pytest.raises(ContractError, match="E57Embedded"):
        build_frameset(tmp_path / "b", pano_convention=PanoConvention(), **common)


def test_t9_every_crop_of_one_panorama_gets_its_own_depth_map_name(frameset):
    stems = [Path(i.name).stem for i in frameset.rec.images]
    assert len(set(stems)) == len(stems)
    # and the crop still binds to bytes, not to its path
    repaint(frameset.dir / "images" / frameset.rec.crops[0].name)
    with pytest.raises(ContractError):
        frameset.check()


# ---------------------------------------------------------------- T10: masks


def test_t10_a_mask_is_bound_to_its_image_by_name_and_by_bytes(frameset):
    from minegs.ingest.video.models import expected_mask_name

    for mask in frameset.rec.masks:
        assert mask.mask_file == expected_mask_name(mask.image)
        assert (frameset.dir / "masks" / mask.mask_file).is_file()
    repaint(frameset.dir / "masks" / frameset.rec.masks[0].mask_file)
    with pytest.raises(ContractError):
        frameset.check()


# ---------------------------------------------------------------- T11-T13: the reconstruction


def test_t11_a_substituted_backend_cannot_report_itself_as_a_real_run(gate, sfm_copy):
    from minegs.ingest.video.sfm.models import load_sfm

    rec, _ = load_sfm(sfm_copy)
    assert rec.real_sfm_execution is False
    assert rec.commands == []  # nothing was executed, so nothing is recorded as executed
    assert rec.frame == "SFM_INTERNAL"
    assert rec.metric_state == "arbitrary_scale"


def test_t12_a_model_edited_after_the_record_is_refused(sfm_copy):
    from minegs.ingest.video.sfm.models import check_sfm, load_sfm

    rec, root = load_sfm(sfm_copy)
    check_sfm(rec, root)
    points = root / rec.model_dir / "points3D.txt"
    points.write_text(points.read_text() + "9999 0.0 0.0 0.0 128 128 128 0.0\n")
    with pytest.raises(ContractError):
        check_sfm(rec, root)


def test_t13_a_registration_of_another_reconstruction_is_refused(gate, tmp_path):
    from minegs.eval.register.models import check_registration, load_registration

    rec, _ = load_registration(outputs(gate.state, Stage.INGEST)["registration_dir"])
    check_registration(rec, rec.sfm_model_sha256)
    with pytest.raises(ContractError, match="belongs to the reconstruction"):
        check_registration(rec, "0" * 64)


# ---------------------------------------------------------------- T14-T20: registration


def test_t14_registration_without_correspondences_does_not_assume_metric(scene, tmp_path):
    with pytest.raises(ContractError, match="needs correspondences"):
        run_phase3(scene, tmp_path / "wf", through=Stage.INGEST, targets_sfm=None, targets_tls=None)


def test_t15_no_thresholds_means_no_claim(scene, tmp_path):
    g = run_phase3(scene, tmp_path / "wf", through=Stage.INGEST, thresholds=None)
    ing = outputs(g.state, Stage.INGEST)
    assert ing["registration_claim_allowed"] is False
    assert any("thresholds" in r for r in ing["registration_claim_refusals"])


def test_t16_a_whole_reference_icp_is_diagnostic_whatever_the_basis(scene, tmp_path):
    g = run_phase3(
        scene,
        tmp_path / "wf",
        through=Stage.INGEST,
        basis="known_target",
        icp_target_ply=str(scene.tls),
        icp_whole=True,
    )
    ing = outputs(g.state, Stage.INGEST)
    assert ing["registration_claim_allowed"] is False
    assert any("whole TLS reference" in r for r in ing["registration_claim_refusals"])


def test_t17_icp_against_geometry_nobody_recorded_the_extent_of_is_refused(scene, tmp_path):
    g = run_phase3(
        scene,
        tmp_path / "wf",
        through=Stage.INGEST,
        icp_target_ply=str(scene.tls),
        icp_ranges_m=None,
    )
    ing = outputs(g.state, Stage.INGEST)
    assert ing["registration_claim_allowed"] is False
    assert any("which chainage it covers" in r for r in ing["registration_claim_refusals"])


def holdout_subset_ply(scene, path: Path) -> Path:
    """The part of the reference that lies inside the evaluation holdout, and only that."""
    cloud = read_ply(scene.tls)
    s, _ = scene.scene.centerline_source.project(cloud.xyz)
    inside = (s >= HOLDOUT[0]) & (s <= HOLDOUT[1])
    return write_ply(PointCloud(cloud.xyz[inside], frame="TLS_GLOBAL"), path, xyz_dtype="f8")


def test_t18_support_that_overlaps_the_holdout_costs_the_claim(scene, tmp_path):
    subset = holdout_subset_ply(scene, tmp_path / "holdout_only.ply")
    g = run_phase3(
        scene,
        tmp_path / "wf",
        through=Stage.DATASET,
        icp_target_ply=str(subset),
        icp_ranges_m=[HOLDOUT],
    )
    ing = outputs(g.state, Stage.INGEST)
    assert ing["registration_support_ranges_m"] == [list(HOLDOUT)]
    ds = outputs(g.state, Stage.DATASET)
    assert "volume_accuracy" not in ds["claims"]
    assert "geometry_accuracy" not in ds["claims"]
    assert any("holdout" in r for r in ds["refusals"])


def test_t19_a_robust_fit_that_gave_up_on_its_inliers_cannot_carry_a_claim():
    from minegs.eval.register.models import IcpRecord, QualityGate, SupportRecord, decide_claim

    allowed, refusals, _ = decide_claim(
        basis="known_target",
        initial_support=SupportRecord(kind="targets"),
        icp_support=None,
        icp=IcpRecord(used=False),
        gate=QualityGate(thresholds=dict(THRESHOLDS), passed=True),
        ransac_fallback_used=True,
    )
    assert allowed is False
    assert any("fell back" in r for r in refusals)


def test_t20_a_registration_that_does_not_permit_a_claim_stops_the_protocol(dataset_copy):
    from minegs.core.manifest import Manifest
    from minegs.eval.protocol import Claim, judge

    m = Manifest.load_dataset(dataset_copy, strict_layout=False)
    assert judge(m).allows(Claim.VOLUME_ACCURACY)
    m.registration.claim_allowed = False
    m.registration.claim_refusals = ["the operator has not finished the survey control"]
    j = judge(m)
    assert not j.allows(Claim.VOLUME_ACCURACY)
    assert not j.allows(Claim.GEOMETRY_ACCURACY)
    assert any("does not permit" in r for r in j.refusals)


# ---------------------------------------------------------------- T21-T25: the dataset


def test_t21_an_initialisation_from_another_instrument_is_refused(scene, dataset_copy):
    from minegs.core.manifest import Manifest
    from minegs.dataset.from_sfm import check_image_only_dataset

    check_image_only_dataset(dataset_copy)
    m = Manifest.load_dataset(dataset_copy, strict_layout=False)
    tls = read_ply(scene.tls)
    write_ply(
        PointCloud(tls.xyz[:2000] - np.asarray(m.T_tls_from_local.t), frame="LOCAL_METRIC"),
        dataset_copy / m.initialization.file,
    )
    with pytest.raises(ContractError, match="did not come from that reconstruction"):
        check_image_only_dataset(dataset_copy)


def test_t22_a_holdout_declared_in_chainage_needs_an_axis_to_be_declared_against(
    gate, scene, tmp_path
):
    from minegs.dataset.from_sfm import SfmDatasetConfig, build_dataset_from_sfm

    ing = outputs(gate.state, Stage.INGEST)
    cfg = SfmDatasetConfig(
        dataset_id="no_axis",
        source="video360",
        geometry_holdout_m=[HOLDOUT],
        centerline_file=None,
        centerline_bin_m=10_000.0,  # too coarse to extract an axis from
        init_voxel_m=0.05,
    )
    with pytest.raises(ContractError):
        build_dataset_from_sfm(
            ing["frameset_dir"], ing["sfm_dir"], ing["registration_dir"], tmp_path / "ds", cfg
        )


def test_t23_the_provenance_beside_the_dataset_must_be_the_record_that_was_measured(dataset_copy):
    from minegs.dataset.from_sfm import check_image_only_dataset

    reg = dataset_copy / PROVENANCE_DIR / "registration.json"
    doc = json.loads(reg.read_text())
    doc["scale"] = doc["scale"] * 1.05
    reg.write_text(json.dumps(doc, indent=2))
    with pytest.raises(ContractError, match="not the ones that were measured"):
        check_image_only_dataset(dataset_copy)


def test_t24_a_video_dataset_without_a_registration_has_no_metric_claims(dataset_copy):
    from minegs.core.manifest import Manifest
    from minegs.eval.protocol import Claim, judge

    m = Manifest.load_dataset(dataset_copy, strict_layout=False)
    m.registration = None
    j = judge(m)
    assert not j.allows(Claim.VOLUME_ACCURACY)
    assert any("registration" in r for r in j.refusals)


def test_t25_a_360_frame_set_cannot_be_built_as_a_plain_video_dataset(gate, tmp_path):
    from minegs.dataset.from_sfm import SfmDatasetConfig, build_dataset_from_sfm

    ing = outputs(gate.state, Stage.INGEST)
    cfg = SfmDatasetConfig(dataset_id="wrong_source", source="video", init_voxel_m=0.05)
    with pytest.raises(ContractError, match="video360"):
        build_dataset_from_sfm(
            ing["frameset_dir"], ing["sfm_dir"], ing["registration_dir"], tmp_path / "ds", cfg
        )


# ---------------------------------------------------------------- T26-T28: the structural gate


def test_t26_each_gate_refuses_the_dataset_it_has_no_evidence_for(gate, tls_gate, tmp_path):
    from minegs.dataset.golden_gate_sfm import run_image_only_gate

    tls_ds = outputs(tls_gate.state, Stage.DATASET)["dataset_dir"]
    with pytest.raises(ContractError, match="image-only gate"):
        run_image_only_gate(tls_ds, tmp_path / "out", raise_on_fail=False)


def test_t27_a_reconstruction_whose_poses_and_points_are_not_in_one_space_fails(scene, tmp_path):
    """A bad reconstruction, not a tampered dataset.

    Moving the dataset's points or poses after the fact is caught earlier, by the check that
    re-derives them from the reconstruction the dataset names. What that check cannot catch is
    an SfM whose *own* output is incoherent: the dataset reproduces it faithfully, every digest
    agrees, and only looking through the cameras shows that they see nothing. That is the gate
    this test is about, so the incoherence is put where a real one would be — in the
    reconstruction itself.
    """
    from minegs.core.frames import quat_to_rotmat
    from minegs.ingest.common import colmap_io

    from video_survey import stand_in_sfm

    honest = stand_in_sfm(scene.survey)

    def adrift(images_dir, work_dir, opts):
        # The poses drift away from the points, not the other way round: the points are what
        # the registration measures itself against, and moving them would leave nothing to
        # measure rather than a reconstruction that is wrong in the way this gate is for.
        run = honest(images_dir, work_dir, opts)
        model = colmap_io.read_model(work_dir / "sparse/0")
        shift = np.array([500.0, 0.0, 0.0])
        for iid, im in model.images.items():
            R_cw = quat_to_rotmat(im.qvec)
            model.images[iid] = colmap_io.Image(
                im.id,
                im.qvec,
                im.tvec - R_cw @ shift,
                im.camera_id,
                im.name,
                im.xys,
                im.point3D_ids,
            )
        colmap_io.write_model(model, work_dir / "sparse/0")
        return run

    root = tmp_path / "wf"
    cfg = phase3_config(scene, root, dataset_id="adrift")
    wf = Workflow(root / "wf", cfg)
    with pytest.raises(ContractError) as excinfo:
        wf.execute(
            phase3_specs(),
            through=Stage.DATASET,
            sfm=adrift,
            frame_extractor=copy_extractor(scene.survey.frames_dir),
        )
    assert "image-only golden gate" in str(excinfo.value)
    report = json.loads(next((root / "wf").rglob("golden_gate_sfm.json")).read_text())
    assert report["structural_result"] == "fail"
    assert any("one space" in p for p in report["problems"])


def test_t28_no_code_path_records_a_human_review_as_passed(gate, dataset_copy, tmp_path):
    from minegs.dataset.golden_gate_sfm import run_image_only_gate

    report = run_image_only_gate(
        dataset_copy, tmp_path / "out", reference_ply=None, raise_on_fail=False
    )
    assert report["real_data_validation_status"] == "pending_human_inspection"
    written = json.loads((tmp_path / "out" / "golden_gate_sfm.json").read_text())
    assert written["real_data_validation_status"] == "pending_human_inspection"
    # And nowhere in the package does anything write a human's verdict for them.
    forbidden = (
        'human_visual_review_status="pass"',
        '"human_visual_review_status": "pass"',
        'real_data_validation_status="validated"',
        '"real_data_validation_status": "validated"',
    )
    offenders = [
        f"{f}: {phrase}"
        for f in Path("minegs").rglob("*.py")
        for phrase in forbidden
        if phrase in f.read_text()
    ]
    assert offenders == []


# ---------------------------------------------------------------- T29-T30: the comparison


def test_t29_two_paths_cut_on_different_grids_are_not_compared(gate, tls_gate):
    from minegs.eval.compare import compare_paths, path_context, require_same_holdout
    from minegs.eval.sections import load_section_input

    tls_ctx = path_context(outputs(tls_gate.state, Stage.DATASET)["dataset_dir"])
    img_ctx = path_context(outputs(gate.state, Stage.DATASET)["dataset_dir"])
    sv = outputs(gate.state, Stage.SECTIONS_VOLUME)
    tsv = outputs(tls_gate.state, Stage.SECTIONS_VOLUME)
    img_pred, _ = load_section_input(sv["sections_predicted_path"])
    img_ref, _ = load_section_input(sv["sections_reference_path"])
    tls_pred, _ = load_section_input(tsv["sections_predicted_path"])
    tls_ref, _ = load_section_input(tsv["sections_reference_path"])
    for record in (img_pred, img_ref):
        record.series.angle_bins = 36
        for section in record.series.sections:
            section.radii_m = section.radii_m[:36]
    with pytest.raises(ContractError, match="not cut on the same grid"):
        compare_paths(
            tls_pred=tls_pred,
            tls_ref=tls_ref,
            image_pred=img_pred,
            image_ref=img_ref,
            ranges=require_same_holdout(tls_ctx, img_ctx),
            tls_context=tls_ctx,
            image_context=img_ctx,
            comparison_id="mismatched",
        )


def test_t30_two_paths_evaluated_over_different_holdouts_are_not_compared(gate, tls_gate):
    from minegs.eval.compare import path_context, require_same_holdout

    tls_ctx = path_context(outputs(tls_gate.state, Stage.DATASET)["dataset_dir"])
    img_ctx = path_context(outputs(gate.state, Stage.DATASET)["dataset_dir"])
    require_same_holdout(tls_ctx, img_ctx)
    img_ctx["holdout_ranges_m"] = [(30.0, 36.0)]
    with pytest.raises(ContractError, match="different geometry holdouts"):
        require_same_holdout(tls_ctx, img_ctx)
    with pytest.raises(ContractError, match="declares none"):
        require_same_holdout(tls_ctx, {**img_ctx, "holdout_ranges_m": []})


def test_a_360_survey_delivered_as_a_folder_of_panoramas_still_gets_its_ring(scene, tmp_path):
    """What the source *is* decides how frames arrive; `--kind` says what the pictures are.

    Reading the directory case off `--kind` meant a 360 capture that arrived already extracted
    — a camera's own export, or someone else's `ffmpeg` — could only be ingested as
    `image_set`, which drops the ring crops that are the entire 360 path.
    """
    from minegs.cli.main import app
    from minegs.ingest.video.models import check_frameset, load_frameset
    from typer.testing import CliRunner

    out = tmp_path / "fs"
    result = CliRunner().invoke(
        app,
        [
            "ingest",
            "video",
            "frameset",
            str(scene.survey.frames_dir),
            str(out),
            "--kind",
            "video360",
            "--n-yaw",
            "4",
            "--fov-deg",
            "90",
            "--size",
            "32",
            "--pano-source",
            "Configured",
            "--blur",
            "0.0",
            "--hamming",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    rec, root = load_frameset(out)
    check_frameset(rec, root)
    assert rec.kind == "video360"
    assert rec.extraction.method == "none"  # nothing was decoded; the frames were handed over
    assert len(rec.crops) == 4 * len(rec.selection.decisions)

    missing = CliRunner().invoke(
        app, ["ingest", "video", "frameset", str(tmp_path / "nope"), str(tmp_path / "x")]
    )
    assert missing.exit_code != 0
    assert "neither a video file nor a directory" in missing.output


# ================================================================ T31-T35, round 2
#
# The four paths a reviewer found around the claim boundary. Each one passed every check in
# the first round of this gate.


def registration_of(dataset_dir: Path):
    from minegs.eval.register.models import RegistrationRecord

    path = Path(dataset_dir) / PROVENANCE_DIR / "registration.json"
    return RegistrationRecord.load(path), path


def test_t31_a_registration_that_declares_a_claim_it_did_not_derive_is_refused(gate, tmp_path):
    """`claim_allowed` is a conclusion, and a conclusion in JSON is four characters from true.

    The earlier check compared the SfM model digest and asked one question about unknown
    support. Everything `decide_claim` reached — the verdict, the refusals, the support extent —
    was read back and believed.
    """
    from minegs.eval.register.models import check_registration, load_registration

    src = Path(outputs(gate.state, Stage.INGEST)["registration_dir"])
    dst = tmp_path / "reg"
    shutil.copytree(src, dst)
    rec, _ = load_registration(dst)
    check_registration(rec, rec.sfm_model_sha256)  # honest record, re-derives cleanly

    def edited(**over):
        doc = json.loads((dst / "registration.json").read_text())
        doc.update(over)
        (dst / "registration.json").write_text(json.dumps(doc, indent=2))
        return load_registration(dst)[0]

    # the thresholds are gone and the gate honestly says so — but the claim still says yes
    no_gate = edited(
        quality_gate={
            "thresholds": None,
            "passed": False,
            "reasons": ["no thresholds configured"],
        }
    )
    with pytest.raises(ContractError, match="not derived from the evidence"):
        check_registration(no_gate, no_gate.sfm_model_sha256)

    # the refusals are deleted but the evidence that produced them is not
    silent = edited(
        quality_gate=rec.quality_gate.model_dump(mode="json"),
        ransac_fallback_used=True,
        claim_allowed=True,
        claim_refusals=[],
    )
    with pytest.raises(ContractError, match="not derived from the evidence"):
        check_registration(silent, silent.sfm_model_sha256)

    # the gate verdict itself is rewritten to pass against thresholds it fails
    forged = edited(
        ransac_fallback_used=False,
        claim_allowed=True,
        claim_refusals=[],
        quality_gate={"thresholds": {"max_rmse_m": 1e-9}, "passed": True, "reasons": []},
    )
    with pytest.raises(ContractError, match="does not follow from its own diagnostics"):
        check_registration(forged, forged.sfm_model_sha256)


def test_t32_a_manifest_that_disagrees_with_its_registration_is_refused(dataset_copy):
    """The protocol judge reads the manifest, so the copy has to be the record.

    The digest check proves the *record* is unedited and says nothing about the manifest's
    copy of it: flipping `claim_allowed` there, or emptying the support ranges, sat beside a
    perfectly intact record and granted exactly what the measurement refused.
    """
    from minegs.core.manifest import Manifest
    from minegs.dataset.from_sfm import check_image_only_dataset

    check_image_only_dataset(dataset_copy)
    rec, _ = registration_of(dataset_copy)
    honest = {
        "claim_allowed": rec.claim_allowed,
        "claim_refusals": list(rec.claim_refusals),
        "support_ranges_m": rec.support_ranges_m,
        "scale": float(rec.scale),
        "rmse_m": float(rec.diagnostics.rmse_m),
    }

    def set_registration(**over):
        m = Manifest.load_dataset(dataset_copy, strict_layout=False)
        for k, v in over.items():
            setattr(m.registration, k, v)
        m.save_dataset(dataset_copy)

    # Any disagreement, either direction: the manifest is not allowed to be a second opinion.
    for over in (
        {"claim_allowed": not honest["claim_allowed"]},
        {"claim_refusals": ["a reason nobody measured"]},
        {"support_ranges_m": [(0.0, 5.0)]},
        {"scale": honest["scale"] * 1.02},
        {"rmse_m": 0.0},
    ):
        set_registration(**over)
        with pytest.raises(ContractError, match=r"not the registration it names|measured scale"):
            check_image_only_dataset(dataset_copy)
        set_registration(**honest)
    check_image_only_dataset(dataset_copy)  # restored


def test_t33_replacing_both_clouds_with_tls_geometry_is_refused(scene, dataset_copy):
    """Two copies of a claim are not evidence for it.

    `init_points.ply` used to be compared against `sparse/0` — both inside the dataset. Put the
    same scanner geometry in both and they agree with each other perfectly, so a dataset whose
    initialisation came from a TLS passed a check named after refusing exactly that.
    """
    from minegs.core.manifest import Manifest
    from minegs.dataset.from_sfm import check_image_only_dataset
    from minegs.ingest.common import colmap_io

    check_image_only_dataset(dataset_copy)
    m = Manifest.load_dataset(dataset_copy, strict_layout=False)
    origin = np.asarray(m.T_tls_from_local.t)
    tls_local = read_ply(scene.tls).xyz[:3000] - origin

    write_ply(PointCloud(tls_local, frame="LOCAL_METRIC"), dataset_copy / m.initialization.file)
    model = colmap_io.read_model(dataset_copy / "sparse" / "0")
    model.points3D = {
        i + 1: colmap_io.Point3D(i + 1, tls_local[i], np.full(3, 128, np.uint8))
        for i in range(len(tls_local))
    }
    colmap_io.write_model(model, dataset_copy / "sparse" / "0")

    with pytest.raises(ContractError, match="did not come from that reconstruction"):
        check_image_only_dataset(dataset_copy)


def test_t34_a_path_that_did_not_say_what_ran_is_not_a_real_execution():
    """`real_execution` is per path, over a required list, and absence is not consent."""
    from minegs.eval.compare.paths import REQUIRED_EXECUTION, _path_execution

    full_tls = dict.fromkeys(REQUIRED_EXECUTION["tls_assisted"], True)
    full_img = dict.fromkeys(REQUIRED_EXECUTION["image_only"], True)
    assert _path_execution("tls_assisted", full_tls)[0] is True
    assert _path_execution("image_only", full_img)[0] is True

    # one required stage never reported
    for label, full in (("tls_assisted", full_tls), ("image_only", full_img)):
        for key in REQUIRED_EXECUTION[label]:
            partial = {k: v for k, v in full.items() if k != key}
            real, missing, _ = _path_execution(label, partial)
            assert real is False and missing == [key]

    # and one path's truth cannot cover the other's substitution
    assert _path_execution("tls_assisted", {**full_tls, "real_gpu_execution": False})[0] is False
    assert _path_execution("image_only", full_img)[0] is True


def test_t34b_one_paths_real_stage_cannot_erase_the_others_substitution(gate, tls_gate):
    from minegs.eval.compare import REQUIRED_EXECUTION, compare_paths, path_context
    from minegs.eval.sections import load_section_input

    def side(g):
        sv = outputs(g.state, Stage.SECTIONS_VOLUME)
        pred, _ = load_section_input(sv["sections_predicted_path"])
        ref, _ = load_section_input(sv["sections_reference_path"])
        return path_context(outputs(g.state, Stage.DATASET)["dataset_dir"]), pred, ref

    tls_ctx, tls_pred, tls_ref = side(tls_gate)
    img_ctx, img_pred, img_ref = side(gate)
    # the TLS trainer was substituted; the image path claims every stage of its own ran
    tls_ctx["execution"] = dict.fromkeys(REQUIRED_EXECUTION["tls_assisted"], False)
    img_ctx["execution"] = dict.fromkeys(REQUIRED_EXECUTION["image_only"], True)
    rep = compare_paths(
        tls_pred=tls_pred,
        tls_ref=tls_ref,
        image_pred=img_pred,
        image_ref=img_ref,
        ranges=[HOLDOUT],
        tls_context=tls_ctx,
        image_context=img_ctx,
        comparison_id="overwrite",
    )
    assert rep.real_execution is False
    assert any("TLS-assisted path substituted" in n for n in rep.notes)
    assert rep.tls_assisted.execution["real_gpu_execution"] is False


def test_t35_an_image_exclusion_nobody_can_check_is_not_an_exclusion(scene, tmp_path):
    """`images_excluded=True` has to be true of the dataset, not only of the reader.

    `train_images()` drops holdout groups at read time — but only those whose chainage is
    known, and it keeps a group it cannot place. So a manifest could record the extrapolation
    test while a group sitting inside the holdout was trained on.
    """
    from minegs.core.manifest import Manifest

    excluded = run_phase3(
        scene, tmp_path / "wf", through=Stage.DATASET, dataset_id="excl", images_excluded=True
    )
    ds = Path(outputs(excluded.state, Stage.DATASET)["dataset_dir"])
    m = Manifest.load_dataset(ds, strict_layout=False)
    assert m.split.geometry_holdout.images_excluded is True
    lo, hi = HOLDOUT
    spans = {g: m.capture_groups[g].span() for g in m.split.train_groups}
    assert all(s is not None for s in spans.values()), spans
    assert not [g for g, (a, b) in spans.items() if a <= hi and b >= lo], spans
    # and the manifest says so structurally: train_images() has nothing left to filter
    assert sorted(m.train_images()) == sorted(m.images_of(m.split.train_groups))

    kept = run_phase3(
        scene, tmp_path / "wf2", through=Stage.DATASET, dataset_id="kept", images_excluded=False
    )
    m2 = Manifest.load_dataset(
        Path(outputs(kept.state, Stage.DATASET)["dataset_dir"]), strict_layout=False
    )
    assert m2.split.geometry_holdout.images_excluded is False
    assert [
        g
        for g, s in ((g, m2.capture_groups[g].span()) for g in m2.split.train_groups)
        if s and s[0] <= hi and s[1] >= lo
    ], "the reconstruction test keeps those images"


def test_t36_a_registration_with_nothing_to_measure_refuses_instead_of_writing_nulls(
    scene, tmp_path
):
    """A residual that is not a number is not a small residual.

    With no inlier anywhere, the diagnostics come back NaN — and NaN does not survive a JSON
    round trip, so the record was written and then could not be read: the failure surfaced two
    stages later as a schema error about a null float. It is refused where it happens, as what
    it is.
    """
    from minegs.ingest.common import colmap_io

    from video_survey import stand_in_sfm

    honest = stand_in_sfm(scene.survey)

    def nowhere(images_dir, work_dir, opts):
        run = honest(images_dir, work_dir, opts)
        model = colmap_io.read_model(work_dir / "sparse/0")
        model.points3D = {
            pid: colmap_io.Point3D(pid, pt.xyz + np.array([1e6, 0.0, 0.0]), pt.rgb)
            for pid, pt in model.points3D.items()
        }
        colmap_io.write_model(model, work_dir / "sparse/0")
        return run

    root = tmp_path / "wf"
    wf = Workflow(root / "wf", phase3_config(scene, root, dataset_id="nowhere"))
    with pytest.raises(ContractError, match="are not numbers"):
        wf.execute(
            phase3_specs(),
            through=Stage.INGEST,
            sfm=nowhere,
            frame_extractor=copy_extractor(scene.survey.frames_dir),
        )


# ================================================================ T37-T38, round 2
#
# A capture group is a stretch of drift, not a point on it; and a gate with a hole in it is
# not a weaker gate.


def plain_video_dataset(scene, root: Path, *, holdout, images_excluded, group_size=3):
    """The traverse path: one camera per frame, groups of consecutive frames.

    Built through the library rather than the workflow, because what is under test is the
    builder's arithmetic over a group's members and the workflow would only carry it.
    """
    from minegs.dataset.from_sfm import SfmDatasetConfig, build_dataset_from_sfm
    from minegs.eval.register.run import register_sfm
    from minegs.ingest.video.build import build_frameset
    from minegs.ingest.video.sfm.run import run_sfm

    from video_survey import stand_in_sfm

    _, fs_dir = build_frameset(
        root / "fs",
        kind="image_set",
        image_dir=scene.survey.frames_dir,
        blur_threshold=0.0,
        hamming_threshold=0,
    )
    _, sfm_dir = run_sfm(fs_dir, root / "sfm", executor=stand_in_sfm(scene.survey))
    _, reg_dir = register_sfm(
        sfm_dir,
        root / "reg",
        basis="known_target",
        targets_sfm=scene.survey.targets_sfm,
        targets_tls=scene.survey.targets_tls,
        reference_ply=scene.tls,
        thresholds=dict(THRESHOLDS),
    )
    cfg = SfmDatasetConfig(
        dataset_id="traverse",
        source="video",
        group_size=group_size,
        geometry_holdout_m=[holdout],
        holdout_images_excluded=images_excluded,
        centerline_file=str(scene.centerline_csv),
        centerline_source="design",
        init_voxel_m=0.05,
    )
    return build_dataset_from_sfm(fs_dir, sfm_dir, reg_dir, root / "ds", cfg)


def test_t37_a_group_that_straddles_the_holdout_boundary_is_excluded(scene, tmp_path):
    """A group's midpoint is not the group.

    The traverse path makes a capture group out of consecutive frames, so it covers a stretch
    of drift. Recording one chainage for it — the mean — hid the case that matters: a group
    whose middle sits outside the holdout while one end reaches inside. Under the mean rule
    that group stayed in training with frames the holdout was supposed to keep out.
    """
    straddle = (10.0, 14.0)
    manifest, _ = plain_video_dataset(
        scene, tmp_path / "excl", holdout=straddle, images_excluded=True
    )
    groups = manifest.capture_groups
    assert all(g.type == "trajectory_segment" for g in groups.values())

    # every group covers a real stretch, and it is the members' own extent
    spans = {gid: g.chainage_range_m for gid, g in groups.items()}
    assert all(s is not None and s[1] > s[0] for s in spans.values()), spans

    straddlers = [
        gid
        for gid, (lo, hi) in spans.items()
        if hi >= straddle[0]
        and lo <= straddle[1]
        and not (straddle[0] <= groups[gid].chainage_m <= straddle[1])
    ]
    assert straddlers, f"the fixture no longer produces a straddling group: {spans}"
    for gid in straddlers:
        assert gid not in manifest.split.train_groups, (
            f"{gid} spans {spans[gid]}, reaches into the holdout {straddle}, and its midpoint "
            f"{groups[gid].chainage_m} is outside it — the case a single chainage hid"
        )
    assert manifest.split.train_groups, "something has to be left to train on"

    # and the reconstruction test keeps them, which is what makes it a different experiment
    kept, _ = plain_video_dataset(scene, tmp_path / "kept", holdout=straddle, images_excluded=False)
    assert set(straddlers) <= set(kept.split.train_groups)


def test_t38_an_incomplete_quality_gate_cannot_carry_a_claim(scene, tmp_path):
    """`{}` is not a lenient gate. It is a gate that ran and judged nothing.

    Reading the thresholds with `.get` and skipping whatever was absent meant an empty
    dictionary passed everything: a registration with a 9.9 m residual over three
    correspondences came out `passed=True` and carried a metric claim. So did a dictionary
    holding one limit, or only a misspelt key that no check reads.
    """
    from minegs.eval.register.models import REQUIRED_THRESHOLDS, evaluate_gate

    from video_survey import stand_in_sfm

    full = dict(THRESHOLDS)
    assert sorted(full) == sorted(REQUIRED_THRESHOLDS)

    for label, thresholds in (
        ("empty", {}),
        ("unknown only", {"min_scale": 0.1}),
        ("partial", {"max_rmse_m": 100.0}),
        ("partial pair", {"max_rmse_m": 100.0, "min_inlier_ratio": 0.0}),
        ("complete but misspelt", {**full, "max_rmse": 1.0}),
    ):
        g = run_phase3(
            scene,
            tmp_path / label.replace(" ", "_"),
            through=Stage.INGEST,
            thresholds=thresholds,
        )
        ing = outputs(g.state, Stage.INGEST)
        assert ing["registration_claim_allowed"] is False, label
        assert ing["registration_claim_refusals"], label

    # the same numbers with the complete gate do carry one
    ok = run_phase3(scene, tmp_path / "complete", through=Stage.INGEST, thresholds=full)
    assert outputs(ok.state, Stage.INGEST)["registration_claim_allowed"] is True

    # and a record carrying a hand-written verdict over an incomplete gate does not survive
    # being re-read, because the gate is re-derived (T31)
    from minegs.eval.register.models import IcpRecord
    from minegs.eval.register.run import register_sfm
    from minegs.ingest.video.build import build_frameset
    from minegs.ingest.video.sfm.run import run_sfm

    _, fs_dir = build_frameset(
        tmp_path / "fs2",
        kind="image_set",
        image_dir=scene.survey.frames_dir,
        blur_threshold=0.0,
        hamming_threshold=0,
    )
    _, sfm_dir = run_sfm(fs_dir, tmp_path / "sfm2", executor=stand_in_sfm(scene.survey))
    rec, _ = register_sfm(
        sfm_dir,
        tmp_path / "reg2",
        basis="known_target",
        targets_sfm=scene.survey.targets_sfm,
        targets_tls=scene.survey.targets_tls,
        reference_ply=scene.tls,
        thresholds=full,
    )
    assert evaluate_gate(rec.diagnostics, IcpRecord(used=False), {}).passed is False
