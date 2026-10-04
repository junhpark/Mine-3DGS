"""Phase 5 C2 — chunk-aware staging and training through the ordinary run path.

The trainer is the stand-in from the GPU-baseline tests; what is checked is the chunk binding,
the staging selection and the refusals, not training. Real GPU training: NOT PERFORMED.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from minegs.chunks.plan import build_chunk_plan
from minegs.chunks.run import train_chunks
from minegs.core.errors import ContractError
from minegs.core.manifest import Manifest
from minegs.core.pointcloud import read_ply
from minegs.ingest.common import colmap_io
from minegs.train.runner import RunConfig, get_runner
from minegs.train.runner import local as runner_local
from minegs.train.runner.base import RunnerConfig, RunStatus, load_record
from minegs.train.staging import stage_dataset
from minegs.train.supervision.support import Support, dataset_centerline

from phase5_scene import CORE_M, OVERLAP_M, long_tunnel
from test_gpu_baseline import FAKE_TRAINER


@pytest.fixture(scope="module")
def tunnel(tmp_path_factory):
    sc = long_tunnel(tmp_path_factory.mktemp("p5t"))
    sc.plan, sc.plan_path = build_chunk_plan(sc.dataset_dir, CORE_M, OVERLAP_M)
    sc.manifest = Manifest.load_dataset(sc.dataset_dir)
    return sc


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    script = tmp_path / "simple_trainer.py"
    script.write_text(FAKE_TRAINER)
    script.with_suffix(".cfg.json").write_text(json.dumps({"span": 50.0}))
    monkeypatch.setenv("MINEGS_GSPLAT_TRAINER", str(script))
    monkeypatch.setattr(runner_local, "cuda_available", lambda: True)

    def configure(**cfg):
        script.with_suffix(".cfg.json").write_text(json.dumps({"span": 50.0, **cfg}))

    return configure


def _runner():
    return get_runner("local", RunnerConfig(runner="local", native=True))


def _run(tunnel, tmp_path, chunk_id, name="r", **kw):
    h = _runner().submit(
        RunConfig(
            dataset_dir=str(tunnel.dataset_dir),
            profile="light",
            run_dir=str(tmp_path / "runs" / name),
            chunk_id=chunk_id,
            chunk_plan=str(tunnel.plan_path),
            overrides={"max_steps": 10, "max_images": None},
            **kw,
        )
    )
    return h, h.wait(poll_s=0.01)


# ================================================================ a chunk run


def test_a_chunk_run_is_bound_to_its_plan_and_stays_in_the_dataset_frame(tunnel, tmp_path, trainer):
    h, status = _run(tunnel, tmp_path, "K001")
    assert status is RunStatus.SUCCEEDED, load_record(h.run_dir).failure_reason
    rec = load_record(h.run_dir)
    c = tunnel.plan.chunk("K001")
    assert rec.chunk_id == "K001" and rec.schema_version == "1.3"
    assert rec.chunk["plan_id"] == tunnel.plan.plan_id
    assert rec.chunk["plan_digest"] == tunnel.plan.plan_digest
    assert tuple(rec.chunk["core_range_m"]) == c.core_range_m
    assert tuple(rec.chunk["support_range_m"]) == c.support_range_m
    assert rec.chunk["images"] == c.images and rec.chunk["capture_groups"] == c.capture_groups
    # one dataset: the dataset identity and frame, not a chunk-local one
    m = tunnel.manifest
    assert rec.dataset_id == m.dataset_id
    assert np.allclose(rec.T_tls_from_local, m.T_tls_from_local.to_list())
    assert rec.frame_of_outputs == "LOCAL_METRIC"
    assert np.allclose(rec.T_local_from_internal, np.eye(4))
    # staged exactly the chunk's images, and init only from its support
    staged = colmap_io.read_model(Path(h.run_dir) / "staged" / "sparse" / "0")
    assert sorted(im.name for im in staged.images.values()) == c.images
    assert rec.chunk["init_points_selected"] == len(staged.points3D) > 0
    total = rec.chunk["init_points_total"]
    assert rec.chunk["init_points_located"] + rec.chunk["init_points_unlocated"] == total
    assert rec.chunk["init_points_selected"] < total
    s, ok = Support(dataset_centerline(tunnel.dataset_dir, m)).locate(
        np.array([p.xyz for p in staged.points3D.values()])
    )
    lo, hi = c.support_range_m
    assert ok.all() and s.min() >= lo - 1e-6 and s.max() <= hi + 1e-6
    assert rec.staged["chunk"]["chunk_id"] == "K001"


def test_a_chunk_needs_its_plan_and_a_plan_needs_its_chunk(tunnel, tmp_path):
    r = _runner()
    for kw, match in (
        ({"chunk_id": "K001"}, "trained from a verified chunk plan"),
        ({"chunk_plan": str(tunnel.plan_path)}, "trained from a verified chunk plan"),
        ({"chunk_id": "K009", "chunk_plan": str(tunnel.plan_path)}, "not in plan"),
    ):
        run_dir = tmp_path / "never"
        with pytest.raises(ContractError, match=match):
            r.prepare(
                RunConfig(
                    dataset_dir=str(tunnel.dataset_dir),
                    profile="light",
                    run_dir=str(run_dir),
                    **kw,
                )
            )
        assert not run_dir.exists()


def test_a_plan_made_for_another_dataset_state_is_refused(tunnel, tmp_path):
    ds = tmp_path / "ds"
    shutil.copytree(tunnel.dataset_dir, ds)
    img = sorted((ds / "images").iterdir())[0]
    img.write_bytes(img.read_bytes() + b"\0")
    with pytest.raises(ContractError, match="dataset hash"):
        _runner().prepare(
            RunConfig(
                dataset_dir=str(ds),
                profile="light",
                run_dir=str(tmp_path / "never"),
                chunk_id="K000",
                chunk_plan=str(tunnel.plan_path),
            )
        )


# ================================================================ staging


def test_staging_takes_the_plan_s_images_and_refuses_anything_else(tunnel, tmp_path):
    c = tunnel.plan.chunk("K000")
    st = stage_dataset(tunnel.dataset_dir, tmp_path / "s", tunnel.manifest, chunk=c)
    assert sorted(st.images) == c.images and st.chunk["chunk_id"] == "K000"
    # the legacy manifest windows are not a plan
    with pytest.raises(ContractError, match="staged from a verified chunk plan"):
        stage_dataset(tunnel.dataset_dir, tmp_path / "s2", tunnel.manifest, chunk_id="C01")
    # a chunk whose image list is not its groups' training members
    bad = c.model_copy(update={"images": c.images[1:]})
    with pytest.raises(ContractError, match="staged whole or not at all"):
        stage_dataset(tunnel.dataset_dir, tmp_path / "s3", tunnel.manifest, chunk=bad)
    # a chunk naming an image the split does not train
    test_img = tunnel.manifest.test_images()[0]
    worse = c.model_copy(update={"images": sorted([*c.images, test_img])})
    with pytest.raises(ContractError, match="does not train on"):
        stage_dataset(tunnel.dataset_dir, tmp_path / "s4", tunnel.manifest, chunk=worse)
    # the SfM sparse-track init is not chunked
    with pytest.raises(ContractError, match="not chunked"):
        stage_dataset(
            tunnel.dataset_dir, tmp_path / "s5", tunnel.manifest, chunk=c, use_init_points=False
        )


def test_chunk_init_is_the_located_init_inside_the_support(tunnel, tmp_path):
    m = tunnel.manifest
    init = read_ply(tunnel.dataset_dir / m.initialization.file)
    s, ok = Support(dataset_centerline(tunnel.dataset_dir, m)).locate(init.xyz)
    for c in tunnel.plan.chunks:
        st = stage_dataset(tunnel.dataset_dir, tmp_path / c.chunk_id, m, chunk=c)
        lo, hi = c.support_range_m
        assert st.chunk["init_points_selected"] == int((ok & (s >= lo) & (s <= hi)).sum())
        assert st.init_points == st.chunk["init_points_selected"]


# ================================================================ the single-run path


def test_a_run_without_a_plan_is_unchanged(tunnel, tmp_path, trainer):
    """Chunking is opt-in: no chunk, no plan, the same staging and the same command."""
    from minegs.train.backends import get_backend
    from minegs.train.profiles import load_profile

    st = stage_dataset(tunnel.dataset_dir, tmp_path / "s", tunnel.manifest, max_images=None)
    m = tunnel.manifest
    assert sorted(st.images) == sorted(set(m.train_images()) - set(m.test_images()))
    assert st.chunk is None
    assert st.init_points == len(read_ply(tunnel.dataset_dir / m.initialization.file))
    h = _runner().submit(
        RunConfig(
            dataset_dir=str(tunnel.dataset_dir),
            profile="light",
            run_dir=str(tmp_path / "runs" / "plain"),
            overrides={"max_steps": 10},
        )
    )
    assert h.wait(poll_s=0.01) is RunStatus.SUCCEEDED
    rec = load_record(h.run_dir)
    assert rec.chunk is None and rec.chunk_id is None and "chunk" not in rec.staged
    prof = load_profile("light")
    prof.max_steps = 10
    want = get_backend("gsplat").build_command(
        Path(h.run_dir) / "staged", Path(h.run_dir) / "backend_out", prof
    )
    assert rec.command == want.argv


# ================================================================ orchestration


def test_train_chunks_runs_every_chunk_in_order(tunnel, tmp_path, trainer):
    done = train_chunks(
        tunnel.dataset_dir,
        tunnel.plan_path,
        "light",
        _runner(),
        runs_dir=tmp_path / "runs",
        overrides={"max_steps": 10, "max_images": None},
        poll_s=0.01,
    )
    assert [d["chunk_id"] for d in done] == ["K000", "K001", "K002"]
    assert all(d["status"] == "succeeded" for d in done)
    for d in done:
        rec = load_record(d["run_dir"])
        assert rec.chunk["chunk_id"] == d["chunk_id"]
        assert rec.chunk["plan_digest"] == tunnel.plan.plan_digest


def test_train_chunks_stops_at_the_first_failure(tunnel, tmp_path, trainer):
    trainer(exit_code=3)
    done = train_chunks(
        tunnel.dataset_dir,
        tunnel.plan_path,
        "light",
        _runner(),
        runs_dir=tmp_path / "runs",
        overrides={"max_steps": 10, "max_images": None},
        poll_s=0.01,
    )
    assert [d["chunk_id"] for d in done] == ["K000"] and done[0]["status"] == "failed"
    assert not (tmp_path / "runs" / "K001").exists()


def test_compare_runs_refuses_runs_of_different_chunks(tunnel, tmp_path, trainer):
    from minegs.eval.compare import RunInputs, compare_runs

    a, _ = _run(tunnel, tmp_path, "K000", name="a")
    b, _ = _run(tunnel, tmp_path, "K001", name="b")
    with pytest.raises(ContractError, match="different parts of the tunnel"):
        compare_runs(
            tunnel.dataset_dir,
            RunInputs(Path(a.run_dir), None, None),
            RunInputs(Path(b.run_dir), None, None),
            comparison_id="c",
        )


# ================================================================ depth supervision under chunking


@pytest.fixture(scope="module")
def supervision(tunnel, tmp_path_factory):
    from minegs.train.supervision.build import build_tls_projection

    return build_tls_projection(
        tunnel.dataset_dir, tunnel.cloud, tmp_path_factory.mktemp("p5d") / "dsup"
    )


def test_a_depth_chunk_reuses_the_global_artifact_unchanged(tunnel, supervision, tmp_path):
    """No child artifact: the verified global one is bound, and the chunk's images have samples."""
    _run_cfg, _m, _p, rec = _runner().prepare(
        RunConfig(
            dataset_dir=str(tunnel.dataset_dir),
            profile="heavy",
            run_dir=str(tmp_path / "r"),
            chunk_id="K001",
            chunk_plan=str(tunnel.plan_path),
            depth_supervision=str(supervision.path),
        )
    )
    assert rec.depth_supervision["artifact_sha256"] == supervision.artifact_sha256
    assert supervision.images_with_samples(tunnel.plan.chunk("K001").images)


def test_a_depth_chunk_with_no_samples_on_its_images_is_refused_before_training(
    tunnel, supervision, tmp_path, monkeypatch
):
    from minegs.train.supervision.depth import VerifiedDepthSupervision

    monkeypatch.setattr(VerifiedDepthSupervision, "images_with_samples", lambda self, n=None: [])
    run_dir = tmp_path / "never"
    with pytest.raises(ContractError, match="carries a depth sample"):
        _runner().prepare(
            RunConfig(
                dataset_dir=str(tunnel.dataset_dir),
                profile="heavy",
                run_dir=str(run_dir),
                chunk_id="K001",
                chunk_plan=str(tunnel.plan_path),
                depth_supervision=str(supervision.path),
            )
        )
    assert not run_dir.exists()
