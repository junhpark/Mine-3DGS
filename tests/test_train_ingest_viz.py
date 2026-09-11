import json

import numpy as np
import pytest
from minegs.core.errors import ContractError, NoGpuError, NotYetImplementedError
from minegs.core.frames import SE3, Sim3, rot_z
from minegs.core.pointcloud import PointCloud, read_ply, write_ply
from minegs.ingest.common import colmap_io
from minegs.ingest.common.equirect import RingCropSpec
from minegs.ingest.common.geometry import PanoConvention
from minegs.ingest.e57.pose_to_colmap import StationPose, stations_to_colmap
from minegs.ingest.e57.tiles import tile_pipeline
from minegs.ingest.video.dedup_blur import blur_score, dhash, hamming
from minegs.ingest.video.frames import extract_command
from minegs.ingest.video.masks import nadir_mask_for_crop
from minegs.ingest.video.rig import rig_config_json, rig_from_ring
from minegs.ingest.video.sfm import SfMOptions, get_sfm_backend
from minegs.train.backends import get_backend
from minegs.train.backends.gsplat import TRAINER_ENV, gsplat_normalization
from minegs.train.profiles import load_profile
from minegs.train.runner import RunConfig, get_runner
from minegs.train.runner import sync as rsync
from minegs.train.runner.base import DATASET_HASH_PATTERNS, RunnerConfig
from minegs.train.staging import select_images, stage_dataset
from minegs.viz.export import write_splat
from minegs.viz.overlay import calibrate_convention, render_overlay


def test_profiles_and_capabilities():
    light, heavy = load_profile("light"), load_profile("heavy")
    be = get_backend("gsplat")
    assert be.check_profile(light) == []
    # heavy requires depth_loss, which this adapter cannot deliver under TLS staging (Phase 4)
    assert be.check_profile(heavy) == ["depth_loss"]
    assert heavy.required_capabilities() == ["antialiasing", "appearance_embedding", "depth_loss"]
    assert be.capabilities().has("depth_loss") is False
    with pytest.raises(ContractError):
        load_profile("nope")
    with pytest.raises(NotYetImplementedError):
        get_backend("pgsr")
    with pytest.raises(ContractError, match="non-commercial"):
        get_backend("inria")


def test_gsplat_command_keeps_local_metric(synthetic, tmp_path):
    be = get_backend("gsplat")
    cmd = be.build_command(
        synthetic.dataset_dir, tmp_path / "out", load_profile("light"), check_trainer=False
    )
    assert "--no-normalize_world_space" in cmd.argv and cmd.T_local_from_internal.is_identity()
    assert cmd.argv[1].endswith("simple_trainer.py") and cmd.argv[2] == "default"
    assert "--max_images" not in cmd.argv and "--absgrad" not in cmd.argv
    assert "--depth_loss" not in cmd.argv


def test_normalize_world_space_true_is_refused(synthetic, tmp_path):
    """Phase 0A/0D contract: BACKEND_INTERNAL == LOCAL_METRIC, no partial support (§3)."""
    be = get_backend("gsplat")
    prof = load_profile("light")
    prof.backend_args["normalize_world_space"] = True
    with pytest.raises(ContractError, match="normalize_world_space=true is not enabled"):
        be.build_command(synthetic.dataset_dir, tmp_path / "out", prof, check_trainer=False)
    # false stays the supported path: no normalisation, identity transform, explicit flag
    prof.backend_args["normalize_world_space"] = False
    cmd = be.build_command(synthetic.dataset_dir, tmp_path / "out", prof, check_trainer=False)
    assert cmd.T_local_from_internal.is_identity()
    assert "--no-normalize_world_space" in cmd.argv and "--normalize_world_space" not in cmd.argv


def test_depth_loss_is_refused_under_tls_staging(synthetic, tmp_path):
    """TLS staging clears COLMAP tracks that upstream depth supervision needs (Phase 4)."""
    be = get_backend("gsplat")
    heavy = load_profile("heavy")
    with pytest.raises(ContractError, match="observation tracks"):
        be.build_command(synthetic.dataset_dir, tmp_path / "out", heavy, check_trainer=False)
    # the refusal explains itself wherever the capability gap is reported
    with pytest.raises(ContractError, match=r"deferred\s+to Phase 4"):
        be.resolve_requests(heavy)
    # a raw backend_args override cannot smuggle the flag past the capability check
    sneaky = load_profile("light")
    sneaky.backend_args["depth_loss"] = True
    with pytest.raises(ContractError, match="observation tracks"):
        be.build_command(synthetic.dataset_dir, tmp_path / "out", sneaky, check_trainer=False)
    # light (depth_loss requested as optional=false) is unaffected
    light_cmd = be.build_command(
        synthetic.dataset_dir, tmp_path / "out", load_profile("light"), check_trainer=False
    )
    assert "--depth_loss" not in light_cmd.argv and light_cmd.T_local_from_internal.is_identity()


def test_refused_options_cannot_be_smuggled_by_spelling(synthetic, tmp_path):
    """tyro accepts --depth-loss and --depth_loss alike, so both spellings must be refused."""
    be = get_backend("gsplat")
    ds, out = synthetic.dataset_dir, tmp_path / "out"
    for key in ("depth_loss", "depth-loss"):
        prof = load_profile("light")
        prof.backend_args[key] = True
        with pytest.raises(ContractError, match="observation tracks"):
            be.build_command(ds, out, prof, check_trainer=False)
    for key in ("normalize_world_space", "normalize-world-space"):
        prof = load_profile("light")
        prof.backend_args.pop("normalize_world_space", None)
        prof.backend_args[key] = True
        with pytest.raises(ContractError, match="normalize_world_space=true is not enabled"):
            be.build_command(ds, out, prof, check_trainer=False)
    # two spellings of one option is itself a contract error, not a last-one-wins merge
    clash = load_profile("light")
    clash.backend_args["max-steps"] = 10
    clash.backend_args["max_steps"] = 20
    with pytest.raises(ContractError, match="two spellings"):
        be.build_command(ds, out, clash, check_trainer=False)


def test_declined_capabilities_are_not_enabled_opportunistically(synthetic, tmp_path):
    """requests[cap]=false means off, even though the backend supports it (ROADMAP invariant 11)."""
    be = get_backend("gsplat")
    light = load_profile("light")
    assert light.requests["appearance_embedding"] is False
    assert be.capabilities().has("appearance_embedding") and be.capabilities().has("bilateral_grid")
    cmd = be.build_command(synthetic.dataset_dir, tmp_path / "out", light, check_trainer=False)
    for flag in ("--app_opt", "--use_bilateral_grid", "--antialiased", "--depth_loss"):
        assert flag not in cmd.argv
    # opting in still works
    opt_in = load_profile("light")
    opt_in.requests["antialiasing"] = True
    assert (
        "--antialiased"
        in be.build_command(
            synthetic.dataset_dir, tmp_path / "out", opt_in, check_trainer=False
        ).argv
    )


def test_runner_refuses_heavy_profile_before_staging(synthetic, tmp_path):
    """`minegs train run --profile heavy` fails closed in prepare(), before any work.

    The assertion watches the directory a run would really create (``<dataset>/../runs/<id>``),
    so moving the capability check after mkdir/stage_dataset makes this test fail.
    """
    runs = synthetic.dataset_dir.parent / "runs"
    before = sorted(p.name for p in runs.iterdir()) if runs.exists() else []
    r = get_runner("local", RunnerConfig(runner="local", image="x@sha256:abc"))
    with pytest.raises(ContractError, match="depth_loss"):
        r.prepare(RunConfig(dataset_dir=str(synthetic.dataset_dir), profile="heavy"))
    after = sorted(p.name for p in runs.iterdir()) if runs.exists() else []
    assert after == before, "refusal must happen before any run directory or staging is created"


def test_gsplat_trainer_contract(synthetic, tmp_path, monkeypatch):
    be = get_backend("gsplat")
    prof = load_profile("light")
    monkeypatch.setenv(TRAINER_ENV, str(tmp_path / "nowhere" / "simple_trainer.py"))
    with pytest.raises(ContractError, match=r"simple_trainer\.py"):
        be.build_command(synthetic.dataset_dir, tmp_path / "out", prof)
    script = tmp_path / "simple_trainer.py"
    script.write_text("# stub")
    monkeypatch.setenv(TRAINER_ENV, str(script))
    cmd = be.build_command(synthetic.dataset_dir, tmp_path / "out", prof)
    assert cmd.argv[:2] == ["python", str(script)]
    # absgrad is a DefaultStrategy field -> --strategy.absgrad; refused with mcmc
    prof.requests["absgrad"] = True
    assert (
        "--strategy.absgrad" in be.build_command(synthetic.dataset_dir, tmp_path / "out", prof).argv
    )
    prof.backend_args["strategy"] = "mcmc"
    with pytest.raises(ContractError, match="absgrad"):
        be.build_command(synthetic.dataset_dir, tmp_path / "out", prof)
    prof.backend_args["strategy"] = "bogus"
    with pytest.raises(ContractError, match="strategy"):
        be.build_command(synthetic.dataset_dir, tmp_path / "out", prof)


def test_select_images_even_subset():
    imgs = [f"i{k}" for k in range(50)]
    assert select_images(imgs, None) == imgs and select_images(imgs, 100) == imgs
    sub = select_images(imgs, 5)
    assert sub == ["i0", "i12", "i24", "i37", "i49"]  # np.round half-to-even at 24.5
    with pytest.raises(ContractError):
        select_images(imgs, 0)


def test_stage_dataset_subset_and_tls_init(synthetic, tmp_path):
    m = synthetic.manifest
    st = stage_dataset(synthetic.dataset_dir, tmp_path / "staged", m, max_images=7)
    assert st.subset and len(st.images) == 7 and st.n_train_available == len(m.train_images())
    assert set(st.images) <= set(m.train_images()) and not set(st.images) & set(m.test_images())
    assert all((st.path / "images" / n).exists() for n in st.images)
    model = colmap_io.read_model(st.path / "sparse" / "0")
    assert sorted(im.name for im in model.images.values()) == sorted(st.images)
    assert st.init_source == "init_points.ply" and len(model.points3D) == st.init_points
    assert len(model.points3D) == len(read_ply(synthetic.dataset_dir / "init_points.ply"))
    assert all(len(im.point3D_ids) == 0 for im in model.images.values())
    # no subset, sfm init: keeps the original sparse points
    st2 = stage_dataset(
        synthetic.dataset_dir, tmp_path / "staged2", m, max_images=None, use_init_points=False
    )
    assert (
        not st2.subset
        and st2.init_source == "points3D.txt"
        and len(st2.images) == st.n_train_available
    )
    with pytest.raises(ContractError, match="unknown chunk"):
        stage_dataset(synthetic.dataset_dir, tmp_path / "s3", m, chunk_id="C99")


def test_dataset_hash_covers_images(synthetic, tmp_path):
    from minegs.core.provenance import sha256_tree

    assert "images/**/*" in DATASET_HASH_PATTERNS and "masks/**/*" in DATASET_HASH_PATTERNS
    before = sha256_tree(synthetic.dataset_dir, DATASET_HASH_PATTERNS)
    img = synthetic.dataset_dir / "images" / synthetic.manifest.all_images()[0]
    original = img.read_bytes()
    try:
        img.write_bytes(original + b"\x00")
        assert sha256_tree(synthetic.dataset_dir, DATASET_HASH_PATTERNS) != before
    finally:
        img.write_bytes(original)
    assert sha256_tree(synthetic.dataset_dir, DATASET_HASH_PATTERNS) == before


def test_runpod_runner_is_explicitly_unimplemented(synthetic):
    r = get_runner("runpod", RunnerConfig(runner="runpod", image="x@sha256:abc"))
    # light, not heavy: heavy is refused earlier for depth_loss, which would mask this path
    with pytest.raises(NotYetImplementedError, match="Phase 6"):
        r.submit(RunConfig(dataset_dir=str(synthetic.dataset_dir), profile="light"))


def test_gsplat_normalization_is_invertible(synthetic):
    T_int_from_local = gsplat_normalization(synthetic.dataset_dir / "sparse" / "0")
    model = colmap_io.read_model(synthetic.dataset_dir / "sparse" / "0")
    pts = model.points_xyz()
    assert np.allclose(
        T_int_from_local.inverse().apply(T_int_from_local.apply(pts)), pts, atol=1e-6
    )


def test_normalize_outputs_reexpresses_ply(synthetic, tmp_path):
    be = get_backend("gsplat")
    out = tmp_path / "backend_out" / "ply"
    n = 10
    extra = {
        "opacity": np.zeros(n),
        "scale_0": np.zeros(n),
        "scale_1": np.zeros(n),
        "scale_2": np.zeros(n),
        "f_dc_0": np.zeros(n),
        "f_dc_1": np.zeros(n),
        "f_dc_2": np.zeros(n),
    }
    extra.update({f"rot_{i}": (np.ones(n) if i == 0 else np.zeros(n)) for i in range(4)})
    pc = PointCloud(
        np.random.default_rng(0).normal(size=(n, 3)), extra=extra, frame="BACKEND_INTERNAL"
    )
    write_ply(pc, out / "point_cloud_6999.ply")
    T = Sim3(2.5, rot_z(10), [1, 2, 3])
    produced = be.normalize_outputs(out.parent, tmp_path / "run", T)
    back = read_ply(produced[0])
    assert back.frame == "LOCAL_METRIC" and np.allclose(back.xyz, T.apply(pc.xyz), atol=1e-5)
    assert np.allclose(back.extra["scale_0"], np.log(2.5), atol=1e-6)  # log-scale shifts with Sim3


def test_local_runner_refuses_without_gpu(synthetic, monkeypatch):
    import minegs.train.runner.local as local

    monkeypatch.setattr(local, "cuda_available", lambda: False)
    r = get_runner("local", RunnerConfig(runner="local", native=True))
    with pytest.raises(NoGpuError, match="Phase 6 and not implemented"):
        r.submit(RunConfig(dataset_dir=str(synthetic.dataset_dir), profile="light"))


def test_runner_prepare_writes_provenance(synthetic):
    r = get_runner("local", RunnerConfig(runner="local", image="x@sha256:abc"))
    run, manifest, _profile, record = r.prepare(
        RunConfig(dataset_dir=str(synthetic.dataset_dir), profile="light", chunk_id="C01")
    )
    assert run.run_id.startswith("gsplat_") and record.docker_digest == "sha256:abc"
    assert record.dataset_hash and record.provenance.parent_ids == [manifest.dataset_id]
    assert record.T_tls_from_local is not None and record.chunk_id == "C01"
    with pytest.raises(ContractError, match="unknown chunk"):
        r.prepare(RunConfig(dataset_dir=str(synthetic.dataset_dir), chunk_id="C99"))


def test_sync_never_pushes_raw(synthetic, tmp_path):
    argv = rsync.push_command(synthetic.dataset_dir, "remote:x")
    assert argv[:3] == ["rclone", "sync", str(synthetic.dataset_dir)] and "--include" in argv
    with pytest.raises(ContractError, match="raw"):
        rsync.push_command(synthetic.root / "raw", "remote:x")
    with pytest.raises(ContractError):
        rsync.push_command(tmp_path, "remote:x")


def test_stations_to_colmap_rig(tmp_path):
    spec = RingCropSpec(n_yaw=4, width=64, height=64)
    st = [
        StationPose("S01", SE3(rot_z(20), [100.0, 200.0, 5.0])),
        StationPose("S02", SE3(rot_z(40), [110.0, 200.0, 5.0])),
    ]
    T_local_from_tls = SE3.from_translation([-105.0, -200.0, -5.0])
    model, members = stations_to_colmap(st, spec, T_local_from_tls, as_rig=True)
    assert len(model.images) == 8 and members["S01"][0] == "S01_p0y00.jpg"
    assert np.allclose(model.images[1].center, [-5.0, 0.0, 0.0])
    assert len(model.frames) == 2 and len(model.frames[1].image_ids) == 4
    colmap_io.write_model(model, tmp_path / "sparse")
    assert (tmp_path / "sparse" / "rigs.txt").exists() and (
        tmp_path / "sparse" / "frames.txt"
    ).exists()
    back = colmap_io.read_model(tmp_path / "sparse")
    assert back.frames[2].rig_from_world.inverse().t.tolist() == pytest.approx(
        [5.0, 0.0, 0.0], abs=1e-8
    )


def test_video_rig_config_and_masks():
    spec = RingCropSpec(n_yaw=6, fov_deg=90, width=32, height=32)
    rig = rig_from_ring(spec)
    assert rig.sensors[0].sensor_from_rig is None and len(rig.sensors) == 6
    cfg = rig_config_json(spec)
    assert (
        cfg[0]["cameras"][0]["ref_sensor"] is True
        and "cam_from_rig_rotation" in cfg[0]["cameras"][1]
    )
    m = nadir_mask_for_crop(
        RingCropSpec(n_yaw=1, fov_deg=120, width=32, height=32, pitches_deg=[-45]).crops()[0],
        nadir_el_deg=-60,
    )
    assert m[0].mean() > m[-1].mean()  # bottom rows look down -> masked


def test_dedup_blur_and_ffmpeg_command(rng, tmp_path):
    sharp = rng.integers(0, 255, (64, 64, 3)).astype(np.uint8)
    blurry = np.full((64, 64, 3), 120, np.uint8)
    assert blur_score(sharp) > blur_score(blurry)
    assert hamming(dhash(sharp), dhash(sharp)) == 0
    argv = extract_command("v.mp4", tmp_path, fps=1.5)
    assert argv[0] == "ffmpeg" and "fps=1.5" in argv


def test_sfm_commands_and_pdal_pipeline(tmp_path):
    for mapper, tool in (("global", "global_mapper"), ("incremental", "mapper")):
        cmds = get_sfm_backend("colmap", mapper).commands(
            tmp_path / "img",
            tmp_path / "w",
            SfMOptions(mapper=mapper, fix_intrinsics=True, rig_config=tmp_path / "rig.json"),
        )
        assert any(tool in c for c in cmds) and any("rig_configurator" in c for c in cmds)
    with pytest.raises(NotYetImplementedError):
        get_sfm_backend("gluemap").commands(tmp_path, tmp_path, SfMOptions())
    pipe = tile_pipeline("scan.e57", tmp_path, 80, 15, 0.02)
    assert (
        pipe["pipeline"][0]["type"] == "readers.e57"
        and pipe["pipeline"][-1]["type"] == "writers.ply"
    )
    json.dumps(pipe)


def test_overlay_calibration_recovers_convention(rng):
    # scanner-frame wall points + a panorama rendered with a known convention
    n = 40000
    az = rng.uniform(0, 2 * np.pi, n)
    el = rng.uniform(-0.6, 0.6, n)
    # structure that is asymmetric under every reflection the 8 candidates can produce
    # (az -> -az, az -> pi - az) and under el -> -el, so the convention is identifiable
    r = 3 + 1.5 * (np.sin(az) + 0.6 * np.cos(az) + 0.4 * np.sin(3 * az) > 0.4) + 0.5 * (el > 0.2)
    xyz = np.column_stack(
        [r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)]
    )
    truth = PanoConvention(-1, False, 180.0)
    from minegs.viz.overlay import range_image

    W, H = 400, 200
    ri = range_image(xyz, W, H, truth)
    ri = np.where(np.isnan(ri), 3.0, ri)
    pano = np.repeat(
        (255 * (ri - ri.min()) / (ri.max() - ri.min())).astype(np.uint8)[..., None], 3, axis=2
    )
    best, _scores = calibrate_convention(pano, xyz, refine_offset=False)
    assert (best.az_sign, best.el_flip, best.az_offset_deg) == (-1, False, 180.0)
    ov = render_overlay(pano, xyz, best)
    assert ov.shape == pano.shape


def test_splat_export(tmp_path, rng):
    n = 8
    extra = {
        "opacity": rng.normal(size=n),
        "f_dc_0": np.zeros(n),
        "f_dc_1": np.zeros(n),
        "f_dc_2": np.zeros(n),
    }
    extra.update({f"scale_{i}": np.full(n, -3.0) for i in range(3)})
    extra.update({f"rot_{i}": (np.ones(n) if i == 0 else np.zeros(n)) for i in range(4)})
    p = write_splat(PointCloud(rng.normal(size=(n, 3)), extra=extra), tmp_path / "a.splat")
    assert p.stat().st_size == 32 * n
