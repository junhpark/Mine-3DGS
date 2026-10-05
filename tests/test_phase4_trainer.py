"""Phase 4 C2 — heavy profile, trainer adapter command, staging for the pinned parser, and the
requested-vs-actual evidence contract. Nothing here needs torch or a GPU.

The adapter's execution (hook, Runner subclass, depth gradient) is tested against a stand-in
upstream in ``test_phase4_trainer_torch.py``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from minegs.core.errors import ContractError, NotYetImplementedError
from minegs.core.provenance import sha256_tree
from minegs.ingest.common import colmap_io
from minegs.train.backends import get_backend
from minegs.train.backends.gsplat import (
    RENDERER_ASSUMED,
    read_trainer_config,
    strategy_name,
)
from minegs.train.profiles import BUILTIN, load_profile
from minegs.train.runner import RunConfig, get_runner
from minegs.train.runner.base import RunnerConfig, RunRecord, check_trainer_evidence
from minegs.train.staging import STAGED_HASH_PATTERNS, stage_dataset
from minegs.train.supervision.build import build_tls_projection
from minegs.train.trainers.advanced_gs import ADAPTER_EVIDENCE, adapter_sha256

from phase4_scene import tls_scene

#: Every option v1.5.3's simple_trainer Config accepts (C0 audit). An emitted upstream flag
#: outside this set is not a pinned flag.
V153_CONFIG_FIELDS = {
    "disable_viewer",
    "ckpt",
    "compression",
    "render_traj_path",
    "data_dir",
    "data_factor",
    "result_dir",
    "test_every",
    "patch_size",
    "global_scale",
    "normalize_world_space",
    "camera_model",
    "port",
    "batch_size",
    "steps_scaler",
    "max_steps",
    "eval_steps",
    "save_steps",
    "save_ply",
    "ply_steps",
    "disable_video",
    "init_type",
    "init_num_pts",
    "init_extent",
    "sh_degree",
    "sh_degree_interval",
    "init_opa",
    "init_scale",
    "ssim_lambda",
    "near_plane",
    "far_plane",
    "strategy",
    "packed",
    "sparse_grad",
    "visible_adam",
    "antialiased",
    "random_bkgd",
    "means_lr",
    "scales_lr",
    "opacities_lr",
    "quats_lr",
    "sh0_lr",
    "shN_lr",
    "opacity_reg",
    "scale_reg",
    "pose_opt",
    "pose_opt_lr",
    "pose_opt_reg",
    "pose_noise",
    "app_opt",
    "app_embed_dim",
    "app_opt_lr",
    "app_opt_reg",
    "use_bilateral_grid",
    "bilateral_grid_shape",
    "depth_loss",
    "depth_lambda",
    "tb_every",
    "tb_save_image",
    "lpips_net",
    "with_ut",
    "with_eval3d",
    "use_fused_bilagrid",
}


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    sc = tls_scene(tmp_path_factory.mktemp("p4c2"))
    sc.supervision = build_tls_projection(
        sc.dataset_dir, sc.cloud, tmp_path_factory.mktemp("p4c2a") / "dsup"
    )
    return sc


def _upstream_flags(argv: list[str]) -> list[str]:
    tail = argv[argv.index("--") + 1 :] if "--" in argv else argv[2:]
    return [t for t in tail if t.startswith("--")]


# ================================================================ profiles


def test_heavy_and_its_ablations_differ_only_in_the_two_requests():
    assert {"heavy", "heavy-base", "heavy-appearance", "heavy-depth"} <= set(BUILTIN)
    heavy = load_profile("heavy").model_dump()
    for name, app, depth in (
        ("heavy-base", False, False),
        ("heavy-appearance", True, False),
        ("heavy-depth", False, True),
        ("heavy", True, True),
    ):
        p = load_profile(name).model_dump()
        assert p["requests"]["appearance_embedding"] is app
        assert p["requests"]["depth_loss"] is depth
        for k in ("backend", "default_runner", "max_images", "data_factor", "max_steps"):
            assert p[k] == heavy[k], (name, k)
        reqs = {
            k: v
            for k, v in p["requests"].items()
            if k not in ("appearance_embedding", "depth_loss")
        }
        assert reqs == {
            k: v
            for k, v in heavy["requests"].items()
            if k not in ("appearance_embedding", "depth_loss")
        }
        args = dict(p["backend_args"])
        want = {k: v for k, v in heavy["backend_args"].items() if depth or k != "depth_lambda"}
        assert args == want, name


def test_heavy_is_the_declared_phase4_configuration():
    h = load_profile("heavy")
    assert h.default_runner == "local"
    assert h.max_images is None and h.data_factor == 2 and h.max_steps == 30000
    assert h.requests == {
        "depth_loss": True,
        "appearance_embedding": True,
        "bilateral_grid": False,
        "antialiasing": False,
    }
    assert h.backend_args["strategy"] == "mcmc" and h.backend_args["sh_degree"] == 3
    assert h.backend_args["normalize_world_space"] is False


# ================================================================ the command


def test_baseline_light_argv_is_unchanged(scene, tmp_path):
    """Phase 4 must not move the baseline: the exact argv Phase 0D built."""
    be = get_backend("gsplat")
    cmd = be.build_command(Path("/s"), Path("/o"), load_profile("light"), trainer=Path("/t.py"))
    assert cmd.argv == [
        "python",
        "/t.py",
        "default",
        "--data_dir",
        "/s",
        "--result_dir",
        "/o",
        "--data_factor",
        "4",
        "--max_steps",
        "7000",
        "--no-normalize_world_space",
        "--disable_viewer",
        "--init_type",
        "sfm",
        "--sh_degree",
        "3",
        "--eval_steps",
        "7000",
        "--save_steps",
        "7000",
        "--save_ply",
    ]
    assert cmd.trainer["entrypoint"] == "upstream"
    assert cmd.expected_config["test_every"] == 8 and cmd.expected_config["data_factor"] == 4


def test_heavy_runs_the_adapter_with_pinned_flags_only(scene, tmp_path):
    be = get_backend("gsplat")
    v = scene.supervision
    cmd = be.build_command(
        Path("/s"),
        Path("/o"),
        load_profile("heavy"),
        trainer=Path("/t.py"),
        depth_supervision_dir=Path("/s/supervision/depth/x"),
        depth_supervision_sha256=v.artifact_sha256,
    )
    argv = cmd.argv
    assert argv[:3] == ["python", "-m", "minegs.train.trainers.advanced_gs"]
    assert argv[argv.index("--trainer") + 1] == "/t.py"
    assert argv[argv.index("--depth-supervision-sha256") + 1] == v.artifact_sha256
    assert "--mcmc-metric-compensation" in argv
    tail = argv[argv.index("--") + 1 :]
    assert tail[0] == "mcmc"
    flags = _upstream_flags(argv)
    names = {f.lstrip("-").replace("-", "_").removeprefix("no_").split(".")[0] for f in flags}
    assert names <= V153_CONFIG_FIELDS, names - V153_CONFIG_FIELDS
    assert "--app_opt" in flags and "--antialiased" not in flags
    assert "--use_bilateral_grid" not in flags and "--depth_loss" not in flags
    assert tail[tail.index("--test_every") + 1] == "1000000"
    assert cmd.trainer["adapter_sha256"] == adapter_sha256()
    exp = cmd.expected_config
    assert exp["strategy"] == "MCMCStrategy" and exp["app_opt"] is True
    assert exp["depth_loss"] is False and exp["depth_lambda"] == 0.01


def test_heavy_without_its_supervision_is_refused(scene):
    be = get_backend("gsplat")
    with pytest.raises(ContractError, match="no verified DepthSupervisionRecord"):
        be.build_command(Path("/s"), Path("/o"), load_profile("heavy"), trainer=Path("/t.py"))


def test_supervision_given_to_a_profile_without_depth_is_refused(scene):
    be = get_backend("gsplat")
    with pytest.raises(ContractError, match="would be ignored"):
        be.build_command(
            Path("/s"),
            Path("/o"),
            load_profile("heavy-appearance"),
            trainer=Path("/t.py"),
            depth_supervision_dir=Path("/x"),
            depth_supervision_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    "edit,match",
    [
        ({"backend_args": {"init_type": "random"}}, "world origin"),
        ({"backend_args": {"camera_model": "fisheye"}}, "renderer reproduces"),
        ({"backend_args": {"near_plane": 0.2}}, "renderer reproduces"),
        ({"backend_args": {"strategy.noise_lr": 1e5}}, "wrong units"),
        ({"backend_args": {"scale_reg": 0.0}}, "wrong units"),
        ({"backend_args": {"use_fused_bilagrid": True}}, "request bilateral_grid"),
        ({"requests": {"pose_refinement": True}}, "drift apart"),
        ({"backend_args": {"depth_loss": True}}, "upstream depth_loss is refused"),
        ({"backend_args": {"normalize_world_space": True}}, "seven conditions"),
    ],
)
def test_heavy_refusals(scene, edit, match):
    prof = load_profile("heavy")
    prof.backend_args.update(edit.get("backend_args", {}))
    prof.requests.update(edit.get("requests", {}))
    with pytest.raises(ContractError, match=match):
        get_backend("gsplat").build_command(
            Path("/s"),
            Path("/o"),
            prof,
            trainer=Path("/t.py"),
            depth_supervision_dir=Path("/x"),
            depth_supervision_sha256="0" * 64,
        )


def test_depth_lambda_without_depth_is_refused():
    prof = load_profile("heavy-base")
    prof.backend_args["depth_lambda"] = 0.1
    with pytest.raises(ContractError, match="depth_lambda without"):
        get_backend("gsplat").build_command(Path("/s"), Path("/o"), prof, trainer=Path("/t.py"))


def test_renderer_assumed_values_may_be_restated_but_not_changed():
    prof = load_profile("light")
    prof.backend_args.update({"near_plane": 0.01, "camera_model": "pinhole"})
    cmd = get_backend("gsplat").build_command(Path("/s"), Path("/o"), prof, trainer=Path("/t"))
    for k, v in RENDERER_ASSUMED.items():
        assert cmd.expected_config[k] == v


def test_normalize_refusal_states_the_phase4_reality():
    from minegs.train.backends.gsplat import NORMALIZE_REFUSAL

    assert "deferred" not in NORMALIZE_REFUSAL and "Phase 4 (" not in NORMALIZE_REFUSAL
    assert "parser.transform" in NORMALIZE_REFUSAL and "seven conditions" in NORMALIZE_REFUSAL


def test_runpod_still_fails_closed_for_heavy(scene):
    r = get_runner("runpod", RunnerConfig(runner="runpod"))
    with pytest.raises(NotYetImplementedError, match="RunPod"):
        r.submit(
            RunConfig(
                dataset_dir=str(scene.dataset_dir),
                profile="heavy",
                depth_supervision=str(scene.supervision.path),
            )
        )


def test_runner_refuses_heavy_without_supervision_before_any_directory(scene, tmp_path):
    run_dir = tmp_path / "never"
    r = get_runner("local", RunnerConfig(runner="local", native=True))
    with pytest.raises(ContractError, match="no DepthSupervisionRecord was given"):
        r.prepare(
            RunConfig(dataset_dir=str(scene.dataset_dir), profile="heavy", run_dir=str(run_dir))
        )
    assert not run_dir.exists()


def test_runner_verifies_supervision_before_any_directory(scene, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(scene.supervision.path, bad)
    s = np.load(bad / "samples.npy")
    s["depth_m"][0] += 1
    np.save(bad / "samples.npy", s, allow_pickle=False)
    run_dir = tmp_path / "never"
    r = get_runner("local", RunnerConfig(runner="local", native=True))
    with pytest.raises(ContractError, match="changed after the record"):
        r.prepare(
            RunConfig(
                dataset_dir=str(scene.dataset_dir),
                profile="heavy",
                run_dir=str(run_dir),
                depth_supervision=str(bad),
            )
        )
    assert not run_dir.exists()


# ================================================================ staging for the pinned parser


def test_staging_writes_a_binary_model_equal_to_the_text_one(scene, tmp_path):
    st = stage_dataset(scene.dataset_dir, tmp_path / "s", max_images=6)
    txt = colmap_io.read_model(st.path / "sparse" / "0")
    bin_ = colmap_io.read_model_binary(st.path / "sparse" / "0")
    assert sorted(txt.images) == sorted(bin_.images) and len(txt.points3D) == len(bin_.points3D)
    for iid, im in txt.images.items():
        assert bin_.images[iid].name == im.name
        assert np.allclose(bin_.images[iid].tvec, im.tvec, atol=1e-8)
    a = next(iter(txt.points3D.values()))
    assert np.allclose(bin_.points3D[a.id].xyz, a.xyz, atol=1e-6)


def test_staging_provides_images_at_the_training_factor(scene, tmp_path):
    from PIL import Image

    st = stage_dataset(scene.dataset_dir, tmp_path / "s", max_images=4, data_factor=2)
    assert st.downscale["folder"] == "images_2" and st.downscale["mode"] == "minegs_png"
    for n in st.images:
        with (
            Image.open(st.path / "images" / n) as full,
            Image.open(st.path / "images_2" / n) as small,
        ):
            assert small.size == (round(full.size[0] / 2), round(full.size[1] / 2))
    # the staged-tree hash covers what the trainer reads, the downscaled copies included
    assert any(p.startswith("images_") for p in STAGED_HASH_PATTERNS)
    assert stage_dataset(scene.dataset_dir, tmp_path / "f1", max_images=4).downscale is None


def test_staging_refuses_a_downscale_upstream_does_not_define(scene, tmp_path):
    ds = tmp_path / "ds"
    shutil.copytree(scene.dataset_dir, ds)
    model = colmap_io.read_model(ds / "sparse" / "0")
    im = next(iter(model.images.values()))
    old = im.name
    im.name = Path(old).with_suffix(".jpeg").name
    (ds / "images" / old).rename(ds / "images" / im.name)
    colmap_io.write_model(model, ds / "sparse" / "0")
    from minegs.core.manifest import Manifest

    m = Manifest.load_dataset(ds, strict_layout=False)
    for g in m.capture_groups.values():
        g.members = [im.name if x == old else x for x in g.members]
    m.save_dataset(ds)
    with pytest.raises(ContractError, match="not defined"):
        stage_dataset(ds, tmp_path / "s", data_factor=2)


def test_staging_clamps_init_colours_for_appearance(scene, tmp_path):
    st = stage_dataset(scene.dataset_dir, tmp_path / "s", max_images=4, clamp_init_rgb=True)
    rgb = colmap_io.read_model_binary(st.path / "sparse" / "0").points_rgb()
    assert rgb.min() >= 1 and rgb.max() <= 254
    assert st.init_rgb_clamped is not None
    plain = stage_dataset(scene.dataset_dir, tmp_path / "p", max_images=4)
    assert plain.init_rgb_clamped is None


def test_staging_without_init_points_keeps_only_staged_tracks(scene, tmp_path):
    st = stage_dataset(scene.dataset_dir, tmp_path / "s", max_images=3, use_init_points=False)
    m = colmap_io.read_model_binary(st.path / "sparse" / "0")
    for p in m.points3D.values():
        assert set(np.asarray(p.image_ids).tolist()) <= set(m.images)


# ================================================================ cfg.yml and the evidence contract


def _cfg_yml(path: Path, **over) -> Path:
    import yaml

    cls = type("MCMCStrategy", (), {})
    cls.__module__ = "gsplat.strategy.mcmc"
    st = cls()
    st.__dict__.update({"noise_lr": over.pop("noise_lr", 5e5), "cap_max": 1000000})
    cfg = {
        "normalize_world_space": False,
        "depth_loss": False,
        "data_factor": 2,
        "max_steps": 10,
        "app_opt": True,
        "use_bilateral_grid": False,
        "antialiased": False,
        "pose_opt": False,
        "init_type": "sfm",
        "sh_degree": 3,
        "test_every": 1000000,
        "camera_model": "pinhole",
        "near_plane": 0.01,
        "far_plane": 1e10,
        "with_ut": False,
        "with_eval3d": False,
        "pose_noise": 0.0,
        "patch_size": None,
        "depth_lambda": 0.01,
        "scale_reg": over.pop("scale_reg", 0.01),
        "bilateral_grid_shape": (16, 16, 8),
        "strategy": st,
    }
    cfg.update(over)
    path.write_text(yaml.dump(cfg))
    return path


def test_cfg_yml_is_read_as_data_without_executing_tags(tmp_path):
    cfg = read_trainer_config(_cfg_yml(tmp_path / "cfg.yml"))
    assert strategy_name(cfg) == "MCMCStrategy"
    assert cfg["strategy"]["noise_lr"] == 5e5
    assert cfg["bilateral_grid_shape"] == [16, 16, 8]


def _record_and_evidence(scene, tmp_path, **cfg_over):
    """A heavy run's record, its staged tree, and evidence matching it exactly."""
    from minegs.train.runner.base import staged_metric_scale

    st = stage_dataset(scene.dataset_dir, tmp_path / "staged", max_images=6, data_factor=2)
    s = staged_metric_scale(st.path)
    cmd = get_backend("gsplat").build_command(
        st.path,
        tmp_path / "out",
        load_profile("heavy"),
        trainer=Path("/t.py"),
        depth_supervision_dir=Path("/x"),
        depth_supervision_sha256=scene.supervision.artifact_sha256,
    )
    cmd.expected_config["max_steps"] = 10
    cfg_over.setdefault("noise_lr", 5e5 * s * s)
    cfg_over.setdefault("scale_reg", 0.01 * s)
    cfg = read_trainer_config(_cfg_yml(tmp_path / "cfg.yml", **cfg_over))
    rec = RunRecord(
        run_id="r",
        dataset_id="d",
        dataset_hash="0" * 64,
        backend={"name": "gsplat"},
        profile={},
        runner="local",
        provenance={"git_commit": "x", "source_assets": [], "tool_versions": {}},
        trainer=cmd.trainer,
        expected_trainer_config=cmd.expected_config,
        depth_supervision=scene.supervision.summary(),
        staged={"n_images": 6},
    )
    adapter = {
        "adapter": {"sha256": adapter_sha256()},
        "optimised_images": ["a"] * 5,
        "mcmc_metric_compensation": {
            "s": s,
            "noise_lr_base": 5e5,
            "noise_lr": 5e5 * s * s,
            "scale_reg_base": 0.01,
            "scale_reg": 0.01 * s,
        },
        "depth_supervision": {
            "artifact_sha256": scene.supervision.artifact_sha256,
            "training_calls": 10,
            "n_samples_in_domain": 50,
            "steps_with_depth_term": 10,
            "images_with_samples": ["a"],
            "images_without_samples": [],
            "depth_lambda": 0.01,
        },
    }
    ev = SimpleNamespace(
        trainer_config_full=cfg,
        trainer_config_sha256="f" * 64,
        adapter_evidence=adapter,
        stats_gaussian_count=100,
    )
    return rec, ev, st.path


def test_matching_evidence_passes_and_is_recorded(scene, tmp_path):
    rec, ev, staged = _record_and_evidence(scene, tmp_path)
    check_trainer_evidence(rec, ev, staged, n_final=100)
    assert rec.trainer_config["app_opt"] is True and rec.trainer_config_sha256 == "f" * 64
    assert rec.metric_compensation["s_host"] > 0
    assert rec.depth_supervision["trainer"]["steps_with_depth_term"] == 10
    assert rec.optimised_images == 5


def test_the_adapters_own_image_count_is_recorded_not_a_host_estimate(scene, tmp_path):
    # Phase 5 hostile review: a chunk branch once re-paired the host formula
    # (n - ceil(n / test_every)) with every non-chunk run, overwriting what the trainer reported.
    rec, ev, staged = _record_and_evidence(scene, tmp_path)
    ev.adapter_evidence["optimised_images"] = ["a"] * 3
    check_trainer_evidence(rec, ev, staged, n_final=100)
    assert rec.optimised_images == 3 and rec.chunk is None


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r, e: e.trainer_config_full.update(app_opt=False), "app_opt: False"),
        (lambda r, e: e.trainer_config_full.update(antialiased=True), "antialiased: True"),
        (lambda r, e: e.trainer_config_full.update(depth_loss=True), "depth_loss: True"),
        (
            lambda r, e: e.trainer_config_full.update(normalize_world_space=True),
            "normalize_world_space",
        ),
        (lambda r, e: e.trainer_config_full.update(data_factor=4), "data_factor"),
        (lambda r, e: e.trainer_config_full.pop("test_every"), "test_every: missing"),
        (
            lambda r, e: e.trainer_config_full["strategy"].update(__tag__="x.DefaultStrategy"),
            "strategy",
        ),
        (lambda r, e: setattr(e, "trainer_config_sha256", None), "no cfg.yml"),
        (lambda r, e: setattr(e, "adapter_evidence", None), "no adapter evidence"),
        (lambda r, e: e.adapter_evidence["adapter"].update(sha256="0" * 64), "not the adapter"),
        (
            lambda r, e: e.adapter_evidence["depth_supervision"].update(artifact_sha256="1" * 64),
            "consumed depth supervision",
        ),
        (
            lambda r, e: e.adapter_evidence["depth_supervision"].update(steps_with_depth_term=0),
            "no training step applied",
        ),
        (lambda r, e: e.adapter_evidence.update(depth_supervision=None), "reports none"),
        (
            lambda r, e: e.adapter_evidence["mcmc_metric_compensation"].update(s=1.0),
            "trainer scale",
        ),
        (lambda r, e: e.trainer_config_full["strategy"].update(noise_lr=5e5), "strategy.noise_lr"),
        (lambda r, e: setattr(e, "stats_gaussian_count", 101), "diverged"),
        (lambda r, e: r.runtime.update(gsplat="1.5.2"), "pinned 1.5.3"),
        # Phase 4 C4: evidence that cannot have happened, or cannot be read
        (lambda r, e: setattr(e, "stats_gaussian_count", None), "cannot be read"),
        (
            lambda r, e: e.adapter_evidence["depth_supervision"].update(
                images_with_samples=["zzz"]
            ),
            "did not optimise",
        ),
        (
            lambda r, e: e.adapter_evidence["depth_supervision"].update(n_samples_in_domain=0),
            "no depth sample inside",
        ),
        (
            lambda r, e: e.adapter_evidence["depth_supervision"].update(steps_with_depth_term=11),
            "cannot have happened",
        ),
        (lambda r, e: setattr(r, "depth_supervision", None), "recorded none"),
    ],
)
def test_evidence_that_does_not_match_the_request_is_refused(scene, tmp_path, mutate, match):
    rec, ev, staged = _record_and_evidence(scene, tmp_path)
    mutate(rec, ev)
    with pytest.raises(ContractError, match=match):
        check_trainer_evidence(rec, ev, staged, n_final=100)


def test_substituted_training_cannot_become_real_by_declaration():
    """The record may say anything about a GPU; a substituted trainer is not real hardware."""
    from minegs.e2e.stages import _real_gpu_execution

    rec = SimpleNamespace(runtime={"gpu_model": "NVIDIA A100", "torch_cuda_available": True})
    assert _real_gpu_execution(rec, substituted=True) is False
    assert _real_gpu_execution(SimpleNamespace(runtime={}), substituted=False) is False


def test_capability_notes_explain_normal_loss_and_depth():
    be = get_backend("gsplat")
    assert "2DGS" in be.capability_notes["normal_loss"]
    prof = load_profile("light")
    prof.requests["normal_loss"] = True
    with pytest.raises(ContractError, match="simple_trainer_2dgs"):
        be.resolve_requests(prof)


def test_adapter_evidence_name_is_what_the_backend_collects(tmp_path):
    from minegs.train.backends.gsplat import GsplatBackend

    (tmp_path / ADAPTER_EVIDENCE).write_text('{"adapter": {"sha256": "x"}}')
    ev = GsplatBackend().collect_evidence(tmp_path, load_profile("heavy"))
    assert ev.adapter_evidence == {"adapter": {"sha256": "x"}}


def test_ci_runs_the_torch_tests():
    """In CI the adapter-execution tests must run, not skip: torch is installed there."""
    if os.environ.get("CI") == "true":
        import torch  # noqa: F401


def test_staged_hash_covers_the_supervision_copy(scene, tmp_path):
    from minegs.train.staging import stage_depth_supervision

    st = stage_dataset(scene.dataset_dir, tmp_path / "s", max_images=3)
    before = sha256_tree(st.path, STAGED_HASH_PATTERNS)
    dst = stage_depth_supervision(scene.supervision, st.path)
    assert sha256_tree(dst) == scene.supervision.artifact_sha256
    assert sha256_tree(st.path, STAGED_HASH_PATTERNS) != before


def test_heavy_runs_pass_the_phase1_render_gate_and_antialiased_ones_do_not(tmp_path, scene):
    """AD-11: nothing is bypassed. heavy's settings are reasoned render-neutral; antialiasing
    is still refused, which is why heavy turns it off."""
    from minegs.eval.surface.render import require_reproducible_render

    def record(profile):
        kw = {}
        if profile.requests.get("depth_loss"):
            kw = {"depth_supervision_dir": Path("/x"), "depth_supervision_sha256": "0" * 64}
        cmd = get_backend("gsplat").build_command(
            Path("/s"), Path("/o"), profile, trainer=Path("/t.py"), **kw
        )
        return SimpleNamespace(
            run_id="r", command=cmd.argv, profile=profile.model_dump(mode="json")
        )

    for name in ("heavy", "heavy-base", "heavy-appearance", "heavy-depth"):
        require_reproducible_render(record(load_profile(name)), tmp_path)
    aa = load_profile("heavy")
    aa.requests["antialiasing"] = True
    with pytest.raises(ContractError, match="antialiased"):
        require_reproducible_render(record(aa), tmp_path)


# ================================================================ C4: hostile review findings


def test_staging_refuses_an_image_that_is_not_its_cameras_size(scene, tmp_path):
    """Upstream rescales K by the first image's size ratio, so a mismatch trains on bent K."""
    from PIL import Image

    ds = tmp_path / "ds"
    shutil.copytree(scene.dataset_dir, ds)
    name = sorted((ds / "images").iterdir())[0]
    with Image.open(name) as im:
        big = im.resize((im.size[0] * 2, im.size[1]))
    big.save(name)
    with pytest.raises(ContractError, match="do not match the intrinsics"):
        stage_dataset(ds, tmp_path / "s")


def test_staging_refuses_a_factor_that_does_not_divide_the_camera(scene, tmp_path):
    cams = colmap_io.read_model(scene.dataset_dir / "sparse" / "0").cameras.values()
    factor = next(f for f in range(3, 50) if any(c.width % f or c.height % f for c in cams))
    with pytest.raises(ContractError, match=f"data_factor {factor} does not divide"):
        stage_dataset(scene.dataset_dir, tmp_path / "s", data_factor=factor)


def _sealed(scene, tmp_path, edit, semantics=None):
    """A copy of the verified artifact, edited, with its own record made consistent again."""
    import json

    from minegs.core.provenance import sha256_file

    art = tmp_path / "art"
    shutil.copytree(scene.supervision.path, art)
    s = np.load(art / "samples.npy")
    edit(s)
    np.save(art / "samples.npy", s, allow_pickle=False)
    rec = json.loads((art / "depth_supervision.json").read_text())
    rec["samples_sha256"] = sha256_file(art / "samples.npy")
    if semantics:
        rec["confidence_semantics"] = semantics
    (art / "depth_supervision.json").write_text(json.dumps(rec))
    return art, sha256_tree(art)


@pytest.mark.parametrize(
    ("edit", "semantics", "match"),
    [
        (lambda s: s["confidence"].__setitem__(0, np.inf), None, "non-finite confidence"),
        (lambda s: s["confidence"].__setitem__(0, 0.5), None, "neither 0 nor 1"),
        (lambda s: s["confidence"].__setitem__(0, 1e-9), "unit_interval_weight", "outside"),
        (lambda s: s["confidence"].__setitem__(0, 7.0), "unit_interval_weight", "outside"),
        (lambda s: s["depth_m"].__setitem__(0, 1e-40), None, "closer than"),
    ],
)
def test_the_adapter_refuses_values_on_its_own(scene, tmp_path, edit, semantics, match):
    """The host verifies first, but the adapter does not lean on that for finiteness."""
    from minegs.train.trainers.advanced_gs import AdapterError, load_supervision

    art, sha = _sealed(scene, tmp_path, edit, semantics)
    with pytest.raises(AdapterError, match=match):
        load_supervision(art, sha)


def test_a_runner_config_for_another_runner_is_refused(scene, tmp_path):
    from minegs.cli.main import app
    from typer.testing import CliRunner

    res = CliRunner().invoke(
        app,
        [
            "train",
            "run",
            str(scene.dataset_dir),
            "--profile",
            "heavy",
            "--depth-supervision",
            str(scene.supervision.path),
            "--config",
            str(Path(__file__).resolve().parents[1] / "configs" / "runner" / "runpod.yaml"),
        ],
    )
    assert res.exit_code != 0
    assert "names runner 'runpod'" in res.output


def test_a_cfg_yml_stating_a_key_twice_is_refused(tmp_path):
    """yaml.dump never writes a key twice; "last one wins" would let either value stand."""
    path = _cfg_yml(tmp_path / "cfg.yml")
    path.write_text(path.read_text() + "normalize_world_space: true\n")
    with pytest.raises(ContractError, match="duplicate key"):
        read_trainer_config(path)


def test_steps_scaler_is_refused():
    prof = load_profile("light")
    prof.backend_args = {**prof.backend_args, "steps_scaler": 3.0}
    with pytest.raises(ContractError, match="steps_scaler"):
        get_backend("gsplat").build_command(Path("/d"), Path("/o"), prof, trainer=Path("/t.py"))
