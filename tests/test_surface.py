"""Phase 1A — the metric surface artifact boundary (§1.7).

Gaussian centres are not a surface. These tests cover the two halves of that contract: depth
maps become LOCAL_METRIC surface samples with a record of where they came from (T1-T5), and a
claim-bearing geometry evaluation will not run without such a record (T6-T8).

Structural only. Nothing here says the samples are accurate — no GPU, no rendered depth, no
real survey. `render_depths` stays unimplemented (Phase 1B).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from minegs.cli.main import app
from minegs.core.errors import ContractError
from minegs.core.frames import SE3
from minegs.core.manifest import Manifest
from minegs.core.pointcloud import read_ply, write_ply
from minegs.core.provenance import ProvenanceRecord, sha256_tree, stamp
from minegs.eval.surface.depth import build_depth_surface, depth_to_points
from minegs.eval.surface.models import (
    SURFACE_FILE,
    SURFACE_POINTS_FILE,
    SurfaceRecord,
    load_surface,
)
from minegs.ingest.common.colmap_io import Camera, Image, read_model
from minegs.train.runner.base import DATASET_HASH_PATTERNS, RunRecord, RunStatus
from typer.testing import CliRunner

runner = CliRunner()


def write_run(run_dir: Path, dataset_dir: Path, run_id: str = "run_test", **over) -> RunRecord:
    """A minimal succeeded run.json over *dataset_dir*, as `minegs train` would leave it."""
    m = Manifest.load_dataset(dataset_dir)
    rec = RunRecord(
        run_id=run_id,
        dataset_id=over.pop("dataset_id", m.dataset_id),
        dataset_hash=over.pop("dataset_hash", sha256_tree(dataset_dir, DATASET_HASH_PATTERNS)),
        backend={"name": "gsplat", "version": "test"},
        profile={"name": "light"},
        runner="local",
        status=over.pop("status", RunStatus.SUCCEEDED),
        provenance=ProvenanceRecord(),
        **over,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    rec.save(run_dir / "run.json")
    return rec


@pytest.fixture
def env(dataset_small, tmp_path):
    """A succeeded run over the built dataset plus one constant-depth map per camera view."""
    ds = dataset_small.dataset_dir
    model = read_model(ds / "sparse" / "0")
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    for im in model.images.values():
        cam = model.cameras[im.camera_id]
        np.save(
            depth_dir / (Path(im.name).stem + ".npy"),
            np.full((cam.height, cam.width), 5.0, np.float32),
        )
    run_dir = tmp_path / "runs" / "run_test"
    write_run(run_dir, ds)
    return SimpleNamespace(
        dataset_dir=ds,
        model=model,
        depth_dir=depth_dir,
        run_dir=run_dir,
        out=tmp_path / "surface" / "depth_v001",
    )


# ---------------------------------------------------------------- T1: backprojection geometry


def test_t1_constant_depth_backprojects_to_a_plane_at_that_range(tmp_path):
    """A 5 m constant depth map is a plane 5 m in front of the camera, in LOCAL_METRIC."""
    K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
    cam = Camera.pinhole(1, K, 64, 48)
    im = Image.from_world_from_cam(1, SE3.identity(), 1, "view.png")
    np.save(tmp_path / "view.npy", np.full((48, 64), 5.0, np.float32))

    pc = depth_to_points(tmp_path, {1: cam}, {1: im}, stride=1)

    assert pc.frame == "LOCAL_METRIC"
    assert len(pc) == 48 * 64
    assert np.allclose(pc.xyz[:, 2], 5.0)
    # the plane fills the frustum: width = (W - 1) / f * z
    assert np.isclose(np.ptp(pc.xyz[:, 0]), 63 / 100.0 * 5.0)
    assert np.isclose(np.ptp(pc.xyz[:, 1]), 47 / 100.0 * 5.0)

    # the pose is applied, not ignored: the same map from a camera 10 m along world z lands there
    moved = Image.from_world_from_cam(2, SE3(np.eye(3), np.array([0.0, 0.0, 10.0])), 1, "view.png")
    assert np.allclose(depth_to_points(tmp_path, {1: cam}, {2: moved}, stride=1).xyz[:, 2], 15.0)


# ---------------------------------------------------------------- T2: invalid depths


def test_t2_non_finite_and_non_positive_depths_are_dropped(tmp_path):
    K = np.array([[10.0, 0.0, 1.0], [0.0, 10.0, 1.5], [0.0, 0.0, 1.0]])
    cam = Camera.pinhole(1, K, 2, 3)
    im = Image.from_world_from_cam(1, SE3.identity(), 1, "view.png")
    np.save(tmp_path / "view.npy", np.array([[5.0, np.nan], [np.inf, 0.0], [-1.0, 7.0]]))

    pc = depth_to_points(tmp_path, {1: cam}, {1: im}, stride=1)

    assert sorted(pc.xyz[:, 2]) == [5.0, 7.0]
    assert np.isfinite(pc.xyz).all()
    # --max-depth uses the same filter
    near = depth_to_points(tmp_path, {1: cam}, {1: im}, stride=1, max_depth=6.0)
    assert sorted(near.xyz[:, 2]) == [5.0]


# ---------------------------------------------------------------- T3: missing depth maps


def test_t3_a_view_without_a_depth_map_fails_closed(env):
    """A skipped view is a hole in the surface, and a hole reads as missing geometry (§12)."""
    dropped = sorted(env.depth_dir.glob("*.npy"))[0]
    dropped.unlink()

    with pytest.raises(ContractError, match="expected"):
        build_depth_surface(env.depth_dir, env.dataset_dir, env.run_dir, env.out)

    assert not env.out.exists()
    assert not list(env.out.parent.glob(".*minegs-partial"))


# ---------------------------------------------------------------- T4: publication


def test_t4_surface_artifact_is_published_with_its_provenance(env):
    r = runner.invoke(
        app,
        [
            "eval",
            "surface-depth",
            str(env.depth_dir),
            str(env.dataset_dir),
            "--run-dir",
            str(env.run_dir),
            "--out",
            str(env.out),
            "--stride",
            "4",
        ],
    )
    assert r.exit_code == 0, r.output
    surface_dir = env.out
    rec, _ = load_surface(surface_dir)

    assert (surface_dir / SURFACE_FILE).is_file()
    assert (surface_dir / SURFACE_POINTS_FILE).is_file()
    assert not list(surface_dir.parent.glob(".*minegs-partial"))

    expected = sum(
        len(range(0, env.model.cameras[im.camera_id].height, 4))
        * len(range(0, env.model.cameras[im.camera_id].width, 4))
        for im in env.model.images.values()
    )
    manifest = Manifest.load_dataset(env.dataset_dir)
    assert rec.point_count == expected
    assert rec.depth_map_count == len(env.model.images)
    assert rec.dataset_id == manifest.dataset_id
    assert rec.run_id == "run_test"
    assert rec.method == "depth_backprojection"
    assert rec.frame == "LOCAL_METRIC"
    assert rec.unit == "m"
    assert rec.parameters["stride"] == 4

    reloaded, points = load_surface(surface_dir)
    assert reloaded.surface_id == rec.surface_id
    cloud = read_ply(points)
    assert cloud.frame == "LOCAL_METRIC"
    assert len(cloud) == rec.point_count
    # the artifact claims provenance, never accuracy
    assert "accuracy" not in json.loads((surface_dir / SURFACE_FILE).read_text())


# ---------------------------------------------------------------- T5: run / dataset identity


def test_t5_a_run_from_another_dataset_cannot_source_a_surface(env, tmp_path):
    foreign = tmp_path / "runs" / "foreign"
    write_run(foreign, env.dataset_dir, run_id="foreign", dataset_id="some_other_dataset")
    with pytest.raises(ContractError, match="some_other_dataset"):
        build_depth_surface(env.depth_dir, env.dataset_dir, foreign, env.out)

    stale = tmp_path / "runs" / "stale"
    write_run(stale, env.dataset_dir, run_id="stale", dataset_hash="0" * 64)
    with pytest.raises(ContractError, match="dataset_hash"):
        build_depth_surface(env.depth_dir, env.dataset_dir, stale, tmp_path / "surface2")

    unfinished = tmp_path / "runs" / "unfinished"
    write_run(unfinished, env.dataset_dir, run_id="unfinished", status=RunStatus.FAILED)
    with pytest.raises(ContractError, match="not succeeded"):
        build_depth_surface(env.depth_dir, env.dataset_dir, unfinished, tmp_path / "surface3")

    assert not env.out.exists()


# ---------------------------------------------------------------- T6-T8: the geometry guard


def synthetic_surface(synthetic, out: Path, run_id: str = "run_synth") -> Path:
    """A surface artifact for the synthetic dataset, made from its LOCAL_METRIC init points.

    Stands in for a depth-fused surface: Phase 1A is about the artifact boundary, and rendering
    depth from a trained run is Phase 1B.
    """
    ds = synthetic.dataset_dir
    pc = read_ply(ds / "init_points.ply")
    out.mkdir(parents=True, exist_ok=True)
    write_ply(pc, out / SURFACE_POINTS_FILE)
    SurfaceRecord(
        surface_id="surface_test_0001",
        dataset_id=synthetic.manifest.dataset_id,
        dataset_hash=sha256_tree(ds, DATASET_HASH_PATTERNS),
        run_id=run_id,
        method="depth_backprojection",
        point_count=len(pc),
        depth_map_count=len(synthetic.manifest.all_images()),
        parameters={"stride": 1},
        provenance=stamp(None),
    ).save(out / SURFACE_FILE)
    return out


def geometry_argv(pred: Path, synthetic, out: Path | None = None, diagnostic: bool = False):
    argv = [
        "eval",
        "geometry",
        str(pred),
        str(synthetic.dataset_dir),
        "--tls-ply",
        str(synthetic.root / "raw" / "tls_full.ply"),
    ]
    if diagnostic:
        argv.append("--diagnostic")
    if out is not None:
        argv += ["--out", str(out)]
    return argv


def test_t6_claim_bearing_geometry_refuses_a_raw_ply(synthetic):
    """The dataset supports geometry_accuracy; the *input* is what is refused (§16)."""
    r = runner.invoke(app, geometry_argv(synthetic.dataset_dir / "init_points.ply", synthetic))
    assert r.exit_code == 2, r.output
    assert "surface artifact" in r.output and "Gaussian centres are not" in r.output
    assert "surface-depth" in r.output


def test_t7_diagnostic_still_accepts_a_raw_ply(synthetic, tmp_path):
    out = tmp_path / "diag.json"
    r = runner.invoke(
        app,
        geometry_argv(synthetic.dataset_dir / "init_points.ply", synthetic, out, diagnostic=True),
    )
    assert r.exit_code == 0, r.output
    assert "diagnostic" in r.output
    assert json.loads(out.read_text())["claim"] == "geometry_diagnostic"


def test_t8_a_surface_artifact_reaches_the_geometry_evaluator(synthetic, tmp_path):
    surface_dir = synthetic_surface(synthetic, tmp_path / "surface" / "depth_v001")
    claimed, diag = tmp_path / "claim.json", tmp_path / "diag.json"

    r = runner.invoke(app, geometry_argv(surface_dir, synthetic, claimed))
    assert r.exit_code == 0, r.output
    rep = json.loads(claimed.read_text())
    assert rep["claim"] == "geometry_accuracy"

    # the surface path changed the gate, not the geometry: the same points passed raw under
    # --diagnostic produce the same numbers through the same LOCAL_METRIC -> TLS_GLOBAL hop.
    r = runner.invoke(
        app,
        geometry_argv(surface_dir / SURFACE_POINTS_FILE, synthetic, diag, diagnostic=True),
    )
    assert r.exit_code == 0, r.output
    same = json.loads(diag.read_text())
    assert same["claim"] == "geometry_diagnostic"
    assert same["accuracy"] == rep["accuracy"] and same["completeness"] == rep["completeness"]

    # an artifact built from a different dataset is refused even though it is a valid artifact
    alien = tmp_path / "alien"
    synthetic_surface(synthetic, alien)
    rec = SurfaceRecord.load(alien / SURFACE_FILE)
    rec.dataset_id = "another_dataset"
    rec.save(alien / SURFACE_FILE)
    r = runner.invoke(app, geometry_argv(alien, synthetic))
    assert r.exit_code == 2 and "another_dataset" in r.output
