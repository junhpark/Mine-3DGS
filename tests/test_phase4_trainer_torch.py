"""Phase 4 C2 — the trainer adapter, executed (needs torch; CPU is enough).

The adapter runs the stand-in upstream in ``tests/fake_upstream`` exactly as it would run
gsplat v1.5.3's ``simple_trainer.py``: through ``runpy``, hooking ``gsplat.distributed.cli``,
replacing ``Runner`` with its subclass. The rasteriser there is a toy, so what is shown is the
wiring, the depth gradient and the evidence; real gsplat training is NOT PERFORMED here.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the adapter executes torch code")

from minegs.train.runner import RunConfig, get_runner  # noqa: E402
from minegs.train.runner import local as runner_local  # noqa: E402
from minegs.train.runner.base import RunnerConfig, RunStatus, load_record  # noqa: E402
from minegs.train.supervision.build import build_tls_projection  # noqa: E402
from minegs.train.trainers import advanced_gs  # noqa: E402
from minegs.train.trainers.depth_term import (  # noqa: E402
    add_gradient,
    depth_term_reference,
    depth_term_torch,
)

from phase4_scene import tls_scene  # noqa: E402

FAKE = Path(__file__).parent / "fake_upstream" / "simple_trainer.py"


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    sc = tls_scene(tmp_path_factory.mktemp("p4t"))
    sc.supervision = build_tls_projection(
        sc.dataset_dir, sc.cloud, tmp_path_factory.mktemp("p4ta") / "dsup"
    )
    return sc


@pytest.fixture
def upstream(monkeypatch):
    """The stand-in upstream as the trainer, and a CUDA device as far as the runner can tell."""
    monkeypatch.setenv("MINEGS_GSPLAT_TRAINER", str(FAKE))
    monkeypatch.setattr(runner_local, "cuda_available", lambda: True)
    # the native path launches `python`; make it this interpreter (the one with torch)
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}:{os.environ['PATH']}")


def _run(scene, tmp_path, profile, **kw):
    r = get_runner("local", RunnerConfig(runner="local", native=True))
    run_dir = tmp_path / "run"
    h = r.submit(
        RunConfig(
            dataset_dir=str(scene.dataset_dir),
            profile=profile,
            run_dir=str(run_dir),
            overrides={"max_steps": 6},
            **kw,
        )
    )
    status = h.wait(poll_s=0.05)
    return status, load_record(run_dir), run_dir


# ================================================================ the term and its gradient


def test_torch_term_equals_the_numpy_definition():
    rng = np.random.default_rng(0)
    H, W, M = 9, 13, 50
    ed = rng.uniform(1, 5, (H, W))
    ed[0, 0] = 0.0
    ui, vi = rng.uniform(0, W - 1, M), rng.uniform(0, H - 1, M)
    z, w = rng.uniform(1, 5, M), (rng.random(M) > 0.3).astype(float)
    ref, n = depth_term_reference([ed], [(ui, vi, z, w)], 2.5, 0.01)
    t, n2 = depth_term_torch(
        torch.tensor(ed)[None, ..., None],
        [tuple(torch.tensor(a) for a in (ui, vi, z, w))],
        2.5,
        0.01,
    )
    assert n == n2 == 1 and float(t) == pytest.approx(ref, rel=1e-12)


def test_injected_gradient_is_exactly_that_of_loss_plus_term():
    torch.manual_seed(0)
    p = torch.randn(5, dtype=torch.float64, requires_grad=True)
    H, W = 7, 11
    base = torch.linspace(1, 2, H * W, dtype=torch.float64).reshape(1, H, W, 1)
    rng = np.random.default_rng(1)
    s = [
        tuple(
            torch.tensor(a)
            for a in (
                rng.uniform(0, W - 1, 30),
                rng.uniform(0, H - 1, 30),
                rng.uniform(1, 4, 30),
                np.ones(30),
            )
        )
    ]

    def render():
        return (base * p[:3].sum()).repeat(1, 1, 1, 3), base * (2 + p[3] ** 2) + p[4]

    rgb, ed = render()
    term, _ = depth_term_torch(ed, s, 3.0, 0.01)
    ((rgb**2).mean() + term).backward()
    want = p.grad.clone()
    p.grad = None
    rgb, ed = render()
    term, _ = depth_term_torch(ed, s, 3.0, 0.01)
    out = add_gradient(rgb, term)
    loss = (out**2).mean() + out.mean() * 0  # colours consumed twice, as upstream does
    loss.backward()
    assert torch.equal(p.grad, want)
    assert float((out**2).mean()) == float((rgb**2).mean())  # the loss value is untouched


def test_zero_depth_gives_no_nan_gradient_and_empty_weights_give_zero():
    ed = torch.zeros(1, 5, 5, 1, requires_grad=True)
    s = [
        tuple(
            torch.tensor(a, dtype=torch.float64)
            for a in ([1.0, 2.0], [1.0, 3.0], [2.0, 2.0], [1.0, 1.0])
        )
    ]
    t, _ = depth_term_torch(ed, s, 1.0, 1.0)
    t.backward()
    assert not torch.isnan(ed.grad).any()
    s0 = [tuple(torch.tensor(a, dtype=torch.float64) for a in ([1.0], [1.0], [2.0], [0.0]))]
    t0, n0 = depth_term_torch(torch.ones(1, 5, 5, 1), s0, 1.0, 1.0)
    assert float(t0) == 0.0 and n0 == 0


# ================================================================ the adapter, end to end


def test_heavy_trains_through_the_adapter_with_its_depth_term(scene, tmp_path, upstream):
    status, rec, run_dir = _run(
        scene, tmp_path, "heavy", depth_supervision=str(scene.supervision.path)
    )
    assert status is RunStatus.SUCCEEDED, rec.failure_reason
    assert rec.trainer["entrypoint"] == "minegs_adapter"
    assert rec.trainer["upstream_trainer"] == str(FAKE)
    sup = rec.depth_supervision
    assert sup["artifact_sha256"] == scene.supervision.artifact_sha256
    assert sup["trainer"]["steps_with_depth_term"] == 6
    assert sup["trainer"]["images_with_samples"] > 0
    comp = rec.metric_compensation
    assert comp["trainer"]["s"] == pytest.approx(comp["s_host"], rel=1e-9)
    assert rec.trainer_config["strategy"] == "MCMCStrategy"
    assert rec.trainer_config["app_opt"] is True and rec.trainer_config["depth_loss"] is False
    assert rec.staged["downscale"]["folder"] == "images_2"
    assert (run_dir / "staged" / "sparse" / "0" / "images.bin").is_file()
    assert (run_dir / "trainer" / "cfg.yml").is_file()
    assert (run_dir / "trainer" / advanced_gs.ADAPTER_EVIDENCE).is_file()
    # the toy colour does not depend on positions, so every gradient on the means is the
    # depth term's: it reached the parameters on every step
    grads = json.loads((run_dir / "backend_out" / "fake_grad_log.json").read_text())
    assert len(grads) == 6 and all(g > 0 for g in grads)


def test_heavy_base_trains_with_no_depth_term(scene, tmp_path, upstream):
    status, rec, run_dir = _run(scene, tmp_path, "heavy-base")
    assert status is RunStatus.SUCCEEDED, rec.failure_reason
    assert rec.depth_supervision is None and rec.trainer["entrypoint"] == "minegs_adapter"
    assert rec.metric_compensation["trainer"]["noise_lr"] < 5e5  # metres, not unit scenes
    grads = json.loads((run_dir / "backend_out" / "fake_grad_log.json").read_text())
    assert all(g == 0 for g in grads)


def test_light_still_runs_upstream_directly(scene, tmp_path, upstream):
    status, rec, _ = _run(scene, tmp_path, "light")
    assert status is RunStatus.SUCCEEDED, rec.failure_reason
    assert rec.trainer["entrypoint"] == "upstream" and rec.metric_compensation is None
    assert rec.command[1] == str(FAKE)


def test_appearance_run_stages_finite_initial_colours(scene, tmp_path, upstream):
    """The stand-in refuses init colours at 0/255 under app_opt, as logit() would make them."""
    import shutil

    from minegs.core.pointcloud import PointCloud, read_ply, write_ply

    ds = tmp_path / "ds"
    shutil.copytree(scene.dataset_dir, ds)
    pc = read_ply(ds / "init_points.ply")
    rgb = np.array(pc.rgb, copy=True)
    rgb[:10] = 0
    rgb[10:20] = 255
    write_ply(PointCloud(pc.xyz, rgb, frame="LOCAL_METRIC"), ds / "init_points.ply")
    scene2 = type(scene)(dataset_dir=ds)
    status, rec, _ = _run(scene2, tmp_path, "heavy-appearance")
    assert status is RunStatus.SUCCEEDED, rec.failure_reason
    assert rec.staged["init_rgb_clamped"] > 0


# ================================================================ adapter refusals


@pytest.fixture(autouse=True)
def _isolated_imports():
    """In-process adapter runs put the stand-in's gsplat/datasets on sys.path; undo that."""
    path, mods = list(sys.path), set(sys.modules)
    yield
    sys.path[:] = path
    for m in set(sys.modules) - mods:
        if m.split(".")[0] in ("gsplat", "datasets", "simple_trainer"):
            del sys.modules[m]


def _adapter(tmp_path, upstream_argv, **extra):
    argv = [
        "--trainer",
        str(extra.pop("trainer", FAKE)),
        "--evidence",
        str(tmp_path / "ev.json"),
        *extra.pop("own", []),
        "--",
        *upstream_argv,
    ]
    return advanced_gs.run(argv)


def test_adapter_refuses_upstream_depth_loss(scene, tmp_path):
    from minegs.train.staging import stage_dataset

    st = stage_dataset(scene.dataset_dir, tmp_path / "s", max_images=4)
    with pytest.raises(advanced_gs.AdapterError, match="init points3D themselves"):
        _adapter(
            tmp_path,
            [
                "default",
                "--data_dir",
                str(st.path),
                "--result_dir",
                str(tmp_path / "o"),
                "--max_steps",
                "2",
                "--no-normalize_world_space",
                "--depth_loss",
                "--data_factor",
                "1",
            ],
        )


def test_adapter_refuses_a_trainer_that_bypasses_the_launcher(tmp_path):
    rogue = tmp_path / "simple_trainer.py"
    rogue.write_text("print('trained without gsplat.distributed.cli')\n")
    (tmp_path / "gsplat").mkdir()
    (tmp_path / "gsplat" / "__init__.py").write_text("")
    (tmp_path / "gsplat" / "distributed.py").write_text(
        "def cli(fn, a, verbose=False):\n    return fn(0, 0, 1, a)\n"
    )
    with pytest.raises(advanced_gs.AdapterError, match="never attached"):
        _adapter(tmp_path, ["default"], trainer=rogue)


def test_adapter_refuses_supervision_that_is_not_the_recorded_bytes(scene, tmp_path):
    with pytest.raises(advanced_gs.AdapterError, match="not the samples that were verified"):
        advanced_gs.load_supervision(scene.supervision.path, "0" * 64)


def test_adapter_drops_confidence_zero_and_keeps_the_rest(scene):
    t = advanced_gs.load_supervision(scene.supervision.path, scene.supervision.artifact_sha256)
    assert t.n_in_loss == scene.supervision.record.n_samples_in_loss


# ================================================================ C4: hostile review findings


def test_the_float32_term_the_adapter_runs_matches_the_definition():
    rng = np.random.default_rng(3)
    H, W, M = 9, 13, 60
    ed = rng.uniform(1, 5, (H, W))
    ui, vi = rng.uniform(0, W - 1, M), rng.uniform(0, H - 1, M)
    z, w = rng.uniform(1, 5, M), rng.uniform(1e-6, 1, M)
    ref, _ = depth_term_reference([ed], [(ui, vi, z, w)], 2.5, 0.01)
    t, _ = depth_term_torch(
        torch.tensor(ed, dtype=torch.float32)[None, ..., None],
        [tuple(torch.tensor(a, dtype=torch.float32) for a in (ui, vi, z, w))],
        2.5,
        0.01,
    )
    assert float(t) == pytest.approx(ref, rel=1e-5)


def test_confidence_is_absolute_and_the_smallest_weight_stays_finite():
    """An image of weight-0.001 samples counts a thousandth; nothing divides by a weight sum."""
    ed = torch.full((2, 5, 5, 1), 2.0, requires_grad=True)
    one = [torch.tensor(a, dtype=torch.float32) for a in ([1.0], [1.0], [1.0], [1.0])]

    def term(w):
        s = [tuple(one), (one[0], one[1], one[2], torch.tensor([w]))]
        return depth_term_torch(ed, s, 1.0, 1.0)[0]

    # each image's error is |1/2 - 1/1| = 0.5; the second is scaled by its weight
    assert float(term(1.0)) == pytest.approx(0.5)
    assert float(term(1e-3)) == pytest.approx(0.5 * (1 + 1e-3) / 2)
    t = term(1e-6)
    t.backward()
    assert torch.isfinite(ed.grad).all()


def test_a_sample_outside_the_rendered_image_is_refused():
    s = [tuple(torch.tensor(a) for a in ([4.5], [1.0], [2.0], [1.0]))]
    with pytest.raises(ValueError, match="outside the rendered 5x5 image"):
        depth_term_torch(torch.ones(1, 5, 5, 1, dtype=torch.float64), s, 1.0, 1.0)


def test_a_render_at_another_size_than_the_samples_were_kept_for_is_refused(monkeypatch):
    """Upstream sizes every camera by the first image's ratio; the render is the real size."""
    from types import SimpleNamespace

    K = np.array([[4.0, 0, 4.0], [0, 4.0, 3.0], [0, 0, 1]])
    monkeypatch.setattr(advanced_gs, "full_resolution_K", lambda _d: {1: K})

    class Base:
        def __init__(self, local_rank, world_rank, world_size, cfg):
            self.cfg = cfg
            self.scene_scale = 1.0
            self.parser = SimpleNamespace(
                image_names=["a.png"],
                camera_ids=[1],
                imsize_dict={1: (8, 6)},
                Ks_dict={1: K.copy()},
                camtoworlds=np.eye(4)[None],
            )
            self.trainset = SimpleNamespace(indices=[0])

        def rasterize_splats(self, camtoworlds, Ks, width, height, **kw):
            return torch.ones(1, height, width, 4, requires_grad=True), None, {}

    table = advanced_gs.SupervisionTable(
        path=Path("."),
        artifact_sha256="x",
        supervision_id="d",
        source_kind="tls_projection",
        confidence_semantics="binary_mask",
        n_samples=1,
        by_name={"a.png": tuple(np.array([v]) for v in (4.0, 3.0, 2.0, 1.0))},
    )
    cfg = SimpleNamespace(
        data_dir=".", patch_size=None, pose_opt=False, pose_noise=0.0, depth_lambda=0.1
    )
    runner_cls = advanced_gs.make_runner_class(Base, table, False, advanced_gs.AdapterState())
    r = runner_cls(0, 0, 1, cfg)
    ids = torch.tensor([0])
    r.rasterize_splats(None, None, 8, 6, image_ids=ids)  # the size the samples were kept for
    with pytest.raises(advanced_gs.AdapterError, match="renders at 7x6"):
        r.rasterize_splats(None, None, 7, 6, image_ids=ids)
