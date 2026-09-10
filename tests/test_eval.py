import json

import numpy as np
import pytest
from minegs.core.errors import ContractError, ProtocolViolation
from minegs.core.frames import SE3, Sim3, rot_z
from minegs.core.manifest import Manifest
from minegs.eval.change import diff_sections
from minegs.eval.geometry import compare_clouds
from minegs.eval.protocol import Claim, Protocol, judge, require
from minegs.eval.register import align_correspondences, diagnose, icp_point_to_point, umeyama
from minegs.eval.render.metrics import psnr, ssim
from minegs.eval.sections import extract_sections
from minegs.eval.surface.mesh import mesh_volume, sample_mesh_surface
from minegs.eval.volume import compare_to_design, integrate_sections


def test_protocol_judgement_on_synthetic(synthetic):
    j = judge(synthetic.manifest)
    assert Protocol.NOVEL_VIEW in j.protocols and Protocol.GEOMETRY_HOLDOUT in j.protocols
    assert (
        j.allows(Claim.RENDER_QUALITY)
        and j.allows(Claim.GEOMETRY_ACCURACY)
        and j.allows(Claim.VOLUME_ACCURACY)
    )
    assert j.refusals == []


def test_protocol_refuses_reconstruction_geometry_claims(synthetic):
    d = synthetic.manifest.model_dump(mode="json")
    d["split"] = {"train_groups": list(d["capture_groups"]), "test_groups": []}
    d["initialization"]["groups"] = [
        g for g, v in d["capture_groups"].items() if v["type"] == "tls_station"
    ]
    d["initialization"]["excluded_chainage_ranges_m"] = []
    m = Manifest.from_dict(d)
    j = judge(m)
    assert j.primary == Protocol.RECONSTRUCTION
    assert not j.allows(Claim.GEOMETRY_ACCURACY) and not j.allows(Claim.RENDER_QUALITY)
    assert j.allows(Claim.GEOMETRY_DIAGNOSTIC)
    with pytest.raises(ProtocolViolation, match="geometry_accuracy"):
        require(m, Claim.GEOMETRY_ACCURACY)


def test_protocol_refuses_without_scale_basis_or_leaky_init(synthetic):
    d = synthetic.manifest.model_dump(mode="json")
    d["scale"] = None
    j = judge(Manifest.from_dict(d))
    assert not j.allows(Claim.GEOMETRY_ACCURACY) and any("scale.basis" in r for r in j.refusals)
    d = synthetic.manifest.model_dump(mode="json")
    d["initialization"]["groups"] = d["initialization"]["groups"] + d["split"]["test_groups"]
    j = judge(Manifest.from_dict(d))
    assert not j.allows(Claim.RENDER_QUALITY)
    d = synthetic.manifest.model_dump(mode="json")
    d["split"]["geometry_holdout"]["points_excluded"] = False
    j = judge(Manifest.from_dict(d))
    assert not j.allows(Claim.GEOMETRY_ACCURACY)
    d = synthetic.manifest.model_dump(mode="json")
    d["source"] = "video"
    d["registration"] = None
    assert not judge(Manifest.from_dict(d)).allows(Claim.GEOMETRY_ACCURACY)


def test_umeyama_and_ransac(rng):
    src = rng.normal(size=(30, 3))
    T = Sim3(1.02, rot_z(15), [1, 2, 3])
    dst = T.apply(src)
    est = umeyama(src, dst)
    assert np.isclose(est.s, 1.02) and np.allclose(est.apply(src), dst, atol=1e-9)
    dst_out = dst.copy()
    dst_out[:5] += 3.0  # outliers
    est2, mask = align_correspondences(src, dst_out, inlier_m=0.05)
    assert mask.sum() == 25 and np.isclose(est2.s, 1.02, atol=1e-6)


def test_icp_recovers_pose_and_diagnostics(synthetic):
    src = synthetic.tls_points_global.xyz[::5] - synthetic.tls_points_global.centroid()
    # ICP is the *refine* step (§7): initial Sim3 from targets is already within cm / ~1 deg.
    # A smooth cylinder wall is near-degenerate along its axis, so a far start slides.
    T = SE3(rot_z(1.0), [0.1, -0.05, 0.02])
    tgt = T.apply(src)
    res = icp_point_to_point(src, tgt, SE3.identity(), max_dist_m=1.0, max_iters=200, tol=1e-9)
    assert res.T.allclose(T, atol=1e-3)
    assert res.rmse_m < 1e-3 and res.inlier_ratio > 0.99
    diag = diagnose(res.T, src, tgt, inlier_m=0.05)
    assert diag.inlier_ratio > 0.99 and diag.scale == 1.0
    reg = diag.to_manifest()
    assert reg.rmse_m < 1e-3 and len(reg.transform) == 4


def test_geometry_metrics_bidirectional(rng, synthetic):
    ref = synthetic.tls_points_global.xyz
    pred = ref + rng.normal(0, 0.01, ref.shape)
    rep = compare_clouds(pred, ref, max_dist_m=0.5)
    assert 0.005 < rep.accuracy.median_m < 0.02
    assert 0.005 < rep.completeness.median_m < 0.02
    assert rep.f_score["0.05"] > 0.99
    # a hole in pred shows up in completeness only
    s, _ = synthetic.centerline_tls.project(pred)
    rep2 = compare_clouds(pred[(s < 30) | (s > 40)], ref, max_dist_m=0.5)
    assert rep2.accuracy.median_m < 0.02
    assert rep2.completeness.clipped_ratio > 0.08
    assert rep2.completeness.p95_m == 0.5


def test_sections_volume_design_against_analytic(synthetic):
    r = 2.5
    ser = extract_sections(
        synthetic.tls_points_global.xyz,
        synthetic.centerline_tls,
        interval_m=2.0,
        thickness_m=0.5,
        angle_bins=72,
    )
    assert ser.valid_count() == len(ser.sections)
    assert np.nanmean(ser.areas()) == pytest.approx(np.pi * r**2, rel=0.01)
    vol = integrate_sections(ser, "centerline:design")
    expected = np.pi * r**2 * (vol.end_chainage_m - vol.start_chainage_m)
    assert vol.volume_m3 == pytest.approx(expected, rel=0.01)
    assert vol.missing_section_count == 0 and vol.section_interval_m == 2.0 and vol.reference_axis
    dc = compare_to_design(ser, 2.4)
    assert dc.overbreak_m3 == pytest.approx(
        np.pi * (r**2 - 2.4**2) * (vol.end_chainage_m - vol.start_chainage_m), rel=0.02
    )
    assert dc.underbreak_m3 == pytest.approx(0.0, abs=0.5)
    # json roundtrip of the series (what the CLI writes)
    from minegs.eval.sections.sections import SectionSeries

    back = SectionSeries.model_validate(json.loads(ser.model_dump_json()))
    assert back.valid_count() == ser.valid_count()


def test_sections_report_missing_not_interpolated(synthetic):
    xyz = synthetic.tls_points_global.xyz
    s, _ = synthetic.centerline_tls.project(xyz)
    ser = extract_sections(
        xyz[(s < 20) | (s > 30)],
        synthetic.centerline_tls,
        interval_m=2.0,
        thickness_m=0.5,
        angle_bins=72,
    )
    missing = [sec.chainage_m for sec in ser.sections if not sec.valid]
    assert missing and all(20 <= m <= 30 for m in missing)
    vol = integrate_sections(ser, "x")
    assert vol.missing_section_count == len(missing)


def test_change_between_epochs(synthetic):
    xyz = synthetic.tls_points_global.xyz
    ser_a = extract_sections(xyz, synthetic.centerline_tls, 2.0, 0.5, 72)
    # epoch b: same tunnel, radius scaled by 4 % between 40 and 60 m
    s, _ = synthetic.centerline_tls.project(xyz)
    R, o = synthetic.centerline_tls.frames_at(s)
    local = np.einsum("nji,nj->ni", R, xyz - o)
    grow = np.where((s > 40) & (s < 60), 1.04, 1.0)
    local[:, 1:] *= grow[:, None]
    xyz_b = o + np.einsum("nij,nj->ni", R, local)
    ser_b = extract_sections(xyz_b, synthetic.centerline_tls, 2.0, 0.5, 72)
    rep = diff_sections(ser_a, ser_b, "ep1", "ep2", "cl")
    assert rep.delta_volume_m3 == pytest.approx(np.pi * 2.5**2 * (1.04**2 - 1) * 20, rel=0.1)
    with pytest.raises(ContractError):
        diff_sections(
            ser_a, extract_sections(xyz, synthetic.centerline_tls, 3.0, 0.5, 72), "a", "b", "cl"
        )


def test_mesh_volume_cube():
    v = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        float,
    )
    f = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )
    assert mesh_volume(v, f) == pytest.approx(1.0)
    pts = sample_mesh_surface(v, f, 1000)
    assert pts.shape == (1000, 3) and pts.min() >= -1e-9 and pts.max() <= 1 + 1e-9


def test_render_metrics(rng):
    a = rng.integers(0, 255, (64, 64, 3)).astype(np.uint8)
    assert psnr(a, a) == np.inf and ssim(a, a) == pytest.approx(1.0)
    b = np.clip(a.astype(int) + rng.integers(-10, 10, a.shape), 0, 255).astype(np.uint8)
    assert 25 < psnr(a, b) < 40 and 0.5 < ssim(a, b) < 1.0
