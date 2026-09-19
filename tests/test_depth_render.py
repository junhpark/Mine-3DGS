"""Phase 1B — rendering metric depth from a trained run (§1.7).

Phase 1A closed the surface boundary but could only ever build diagnostic surfaces: handed a
directory of ``.npy`` files, nothing tells it which run they came from. This is the other half.
The question here is not "does the rasteriser look right" — it cannot run without a GPU — but
"can a depth map be turned into evidence, and is every way of faking that evidence refused".

So the renderer adapter is substituted and *nothing else is*: run status, dataset identity,
metric frame, checkpoint identity, per-view resolution, coverage, digests and publication all
run exactly as in production. The one test that touches the real gsplat adapter checks that it
refuses to run without CUDA, which is the only part of it a CPU machine can honestly assert.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from minegs.cli.main import app
from minegs.core.errors import ContractError, MissingDependencyError, NoGpuError
from minegs.core.provenance import sha256_file
from minegs.eval.surface.depth import build_depth_surface
from minegs.eval.surface.models import (
    DEPTH_MANIFEST_FILE,
    SURFACE_FILE,
    DepthManifest,
)
from minegs.eval.surface.render import (
    DepthRenderer,
    GsplatDepthRenderer,
    check_checkpoint_blob,
    render_depths,
)
from minegs.ingest.common.colmap_io import read_model
from minegs.train.profiles import load_profile
from minegs.train.runner.base import RunRecord, RunStatus
from typer.testing import CliRunner

from test_surface import geometry_argv, write_run

runner = CliRunner()

CKPT_REL = "ckpt/ckpt_6999_rank0.pt"


class FakeRenderer(DepthRenderer):
    """Stands in for the rasteriser. Produces arrays; makes no contract decisions."""

    name = "fake-ed"
    backend = "gsplat"

    def __init__(
        self,
        depth_m: float = 4.0,
        skip: tuple[str, ...] = (),
        half_size: tuple[str, ...] = (),
        infinite: tuple[str, ...] = (),
        empty: tuple[str, ...] = (),
        duplicate: str | None = None,
        phantom: str | None = None,
    ) -> None:
        self.depth_m = depth_m
        self.skip, self.half_size = skip, half_size
        self.infinite, self.empty = infinite, empty
        self.duplicate, self.phantom = duplicate, phantom
        self.checkpoint: Path | None = None

    def version(self) -> str:
        return "test"

    def require_available(self) -> None:
        return None

    def settings(self) -> dict:
        return {"render_mode": "ED", "min_alpha": 0.5}

    def render(self, checkpoint, cameras, images):
        self.checkpoint = checkpoint
        for im in images.values():
            if im.name in self.skip:
                continue
            cam = cameras[im.camera_id]
            h, w = cam.height, cam.width
            if im.name in self.half_size:
                h, w = h // 2, w // 2
            arr = np.full((h, w), self.depth_m, np.float32)
            if im.name in self.empty:
                arr[:] = np.nan
            else:
                arr[0, 0] = np.nan  # an honest "this ray hit nothing" pixel in every view
            if im.name in self.infinite:
                arr[0, 1] = np.inf
            yield im, arr
            if self.duplicate == im.name:
                yield im, arr
        if self.phantom is not None:
            im = next(iter(images.values()))
            ghost = SimpleNamespace(
                id=9999, camera_id=im.camera_id, name=self.phantom, world_from_cam=im.world_from_cam
            )
            cam = cameras[im.camera_id]
            yield ghost, np.full((cam.height, cam.width), self.depth_m, np.float32)


def make_run(dataset_dir: Path, run_dir: Path, run_id: str = "run_1b", **over):
    """A succeeded gsplat run with a checkpoint on disk and identity backend->metric frame."""
    over.setdefault("T_local_from_internal", np.eye(4).tolist())
    over.setdefault("final_checkpoint", CKPT_REL)
    over.setdefault("checkpoint_step", 6999)
    rec = write_run(run_dir, dataset_dir, run_id=run_id, **over)
    if rec.final_checkpoint:
        ckpt = run_dir / rec.final_checkpoint
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"weights would live here; the renderer is substituted")
    return rec


@pytest.fixture
def env(dataset_small, tmp_path):
    ds = dataset_small.dataset_dir
    run_dir = tmp_path / "runs" / "run_1b"
    make_run(ds, run_dir)
    return SimpleNamespace(
        dataset_dir=ds,
        model=read_model(ds / "sparse" / "0"),
        run_dir=run_dir,
        out=tmp_path / "depth",
    )


# ---------------------------------------------------------------- the run must be renderable


def test_a_run_from_another_dataset_is_refused(dataset_small, tmp_path):
    ds = dataset_small.dataset_dir
    foreign = tmp_path / "runs" / "foreign"
    make_run(ds, foreign, run_id="foreign", dataset_id="some_other_dataset")
    with pytest.raises(ContractError, match="some_other_dataset"):
        render_depths(foreign, ds, tmp_path / "d1", renderer=FakeRenderer())

    stale = tmp_path / "runs" / "stale"
    make_run(ds, stale, run_id="stale", dataset_hash="0" * 64)
    with pytest.raises(ContractError, match="dataset_hash"):
        render_depths(stale, ds, tmp_path / "d2", renderer=FakeRenderer())
    assert not (tmp_path / "d1").exists() and not (tmp_path / "d2").exists()


def test_an_unfinished_run_is_refused(dataset_small, tmp_path):
    ds = dataset_small.dataset_dir
    failed = tmp_path / "runs" / "failed"
    make_run(ds, failed, run_id="failed", status=RunStatus.FAILED)
    with pytest.raises(ContractError, match="not succeeded"):
        render_depths(failed, ds, tmp_path / "d", renderer=FakeRenderer())


def test_a_run_without_a_checkpoint_has_nothing_to_render(dataset_small, tmp_path):
    ds = dataset_small.dataset_dir
    none = tmp_path / "runs" / "nockpt"
    make_run(ds, none, run_id="nockpt", final_checkpoint=None)
    with pytest.raises(ContractError, match="no final checkpoint"):
        render_depths(none, ds, tmp_path / "d1", renderer=FakeRenderer())

    gone = tmp_path / "runs" / "gone"
    make_run(ds, gone, run_id="gone")
    (gone / CKPT_REL).unlink()
    with pytest.raises(ContractError, match="missing"):
        render_depths(gone, ds, tmp_path / "d2", renderer=FakeRenderer())


def test_depth_from_a_non_metric_run_cannot_be_shown_to_be_metric(dataset_small, tmp_path):
    """A backend frame that is not LOCAL_METRIC one-to-one makes every range a backend unit."""
    ds = dataset_small.dataset_dir
    scaled = np.eye(4)
    scaled[:3, :3] *= 3.0
    run_dir = tmp_path / "runs" / "scaled"
    make_run(ds, run_dir, run_id="scaled", T_local_from_internal=scaled.tolist())
    with pytest.raises(ContractError, match="cannot be shown to be metric"):
        render_depths(run_dir, ds, tmp_path / "d", renderer=FakeRenderer())


def test_the_real_gsplat_renderer_never_runs_without_a_gpu(monkeypatch):
    """There is no CPU path, and nothing about the refusal depends on what CI has installed.

    On a runner without torch/gsplat the missing dependency is named first; on one that has
    them the absent CUDA device is. Either way ``require_available`` raises, which is the
    property that matters: the adapter has no branch that renders anything on a CPU.
    """
    monkeypatch.setattr("minegs.train.runner.base.cuda_available", lambda: False)
    installed = all(_importable(m) for m in ("torch", "gsplat"))
    with pytest.raises(NoGpuError if installed else MissingDependencyError) as e:
        GsplatDepthRenderer().require_available()
    assert "CUDA" in str(e.value) or "pip install" in str(e.value)


def _importable(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - a broken install, not an absent one
        return False


# ---------------------------------------------------------------- what the renderer returns


def test_a_view_the_renderer_skipped_or_invented_is_refused(env, tmp_path):
    dropped = next(iter(env.model.images.values())).name
    with pytest.raises(ContractError, match="missing"):
        render_depths(
            env.run_dir, env.dataset_dir, tmp_path / "d1", renderer=FakeRenderer(skip=(dropped,))
        )
    with pytest.raises(ContractError, match="does not have"):
        render_depths(
            env.run_dir,
            env.dataset_dir,
            tmp_path / "d2",
            renderer=FakeRenderer(phantom="not_in_this_dataset.jpg"),
        )
    with pytest.raises(ContractError, match="twice"):
        render_depths(
            env.run_dir, env.dataset_dir, tmp_path / "d3", renderer=FakeRenderer(duplicate=dropped)
        )
    assert not any((tmp_path / n).exists() for n in ("d1", "d2", "d3"))


def test_depth_at_the_wrong_resolution_is_refused(env, tmp_path):
    bad = next(iter(env.model.images.values())).name
    with pytest.raises(ContractError, match="intrinsics"):
        render_depths(
            env.run_dir, env.dataset_dir, tmp_path / "d", renderer=FakeRenderer(half_size=(bad,))
        )
    assert not (tmp_path / "d").exists()


def test_infinite_or_wholly_empty_depth_is_refused(env, tmp_path):
    bad = next(iter(env.model.images.values())).name
    with pytest.raises(ContractError, match="infinit"):
        render_depths(
            env.run_dir, env.dataset_dir, tmp_path / "d1", renderer=FakeRenderer(infinite=(bad,))
        )
    with pytest.raises(ContractError, match="rendered empty"):
        render_depths(
            env.run_dir, env.dataset_dir, tmp_path / "d2", renderer=FakeRenderer(empty=(bad,))
        )


# ---------------------------------------------------------------- the manifest


def test_the_manifest_records_a_digest_per_depth_map(env):
    manifest, depth_dir = render_depths(
        env.run_dir, env.dataset_dir, env.out, renderer=FakeRenderer()
    )
    assert (depth_dir / DEPTH_MANIFEST_FILE).is_file()
    assert not list(depth_dir.parent.glob(".*minegs-partial"))
    assert len(manifest.depths) == len(env.model.images)
    assert manifest.run_id == "run_1b" and manifest.frame == "LOCAL_METRIC"
    assert manifest.unit == "m"
    assert manifest.checkpoint["sha256"] == sha256_file(env.run_dir / CKPT_REL)
    assert manifest.renderer["name"] == "fake-ed"

    on_disk = DepthManifest.load(depth_dir / DEPTH_MANIFEST_FILE)
    for entry in on_disk.depths:
        f = depth_dir / entry.file
        assert entry.sha256 == sha256_file(f)
        arr = np.load(f)
        assert arr.dtype == np.float32
        cam = env.model.cameras[entry.camera_id]
        assert arr.shape == (cam.height, cam.width) == (entry.height, entry.width)
        # the one NaN per view is the documented "this ray hit nothing"
        assert 0.0 < entry.valid_ratio < 1.0
        assert entry.min_m == pytest.approx(4.0)


def test_a_tampered_depth_map_is_refused(env, tmp_path):
    _, depth_dir = render_depths(env.run_dir, env.dataset_dir, env.out, renderer=FakeRenderer())
    victim = sorted(depth_dir.glob("*.npy"))[0]
    arr = np.load(victim)
    arr[:] = 9.0
    np.save(victim, arr)

    with pytest.raises(ContractError, match="changed after it was rendered"):
        build_depth_surface(depth_dir, env.dataset_dir, env.run_dir, tmp_path / "surface")
    assert not (tmp_path / "surface").exists()


def test_a_manifest_from_another_run_does_not_promote(env, tmp_path):
    _, depth_dir = render_depths(env.run_dir, env.dataset_dir, env.out, renderer=FakeRenderer())
    other = tmp_path / "runs" / "other"
    make_run(env.dataset_dir, other, run_id="other")
    with pytest.raises(ContractError, match="--run-dir names"):
        build_depth_surface(depth_dir, env.dataset_dir, other, tmp_path / "surface")


# ---------------------------------------------------------------- promotion


def test_external_depth_stays_diagnostic_only(env, tmp_path):
    """The Phase 1A path is unchanged: no manifest, no claim."""
    _, depth_dir = render_depths(env.run_dir, env.dataset_dir, env.out, renderer=FakeRenderer())
    (depth_dir / DEPTH_MANIFEST_FILE).unlink()

    rec, _ = build_depth_surface(depth_dir, env.dataset_dir, env.run_dir, tmp_path / "surface")
    assert rec.depth_source == "external_unverified"
    assert rec.supports_accuracy_claim is False


def test_rendered_depth_promotes_the_surface(env, tmp_path):
    manifest, depth_dir = render_depths(
        env.run_dir, env.dataset_dir, env.out, renderer=FakeRenderer()
    )
    rec, _ = build_depth_surface(depth_dir, env.dataset_dir, env.run_dir, tmp_path / "surface")
    assert rec.depth_source == "minegs_render"
    assert rec.supports_accuracy_claim is True
    assert rec.parameters["depth_manifest_id"] == manifest.manifest_id
    assert rec.parameters["renderer"] == "fake-ed"
    assert rec.parameters["checkpoint"] == CKPT_REL


def test_a_rendered_surface_reaches_the_geometry_accuracy_gate(synthetic, tmp_path):
    """The whole chain: run -> rendered depth -> verified manifest -> claim-bearing geometry."""
    ds = synthetic.dataset_dir
    run_dir = tmp_path / "runs" / "run_chain"
    make_run(ds, run_dir, run_id="run_chain")

    _, depth_dir = render_depths(run_dir, ds, renderer=FakeRenderer(depth_m=3.0))
    assert depth_dir == run_dir / "depth"

    rec, surface_dir = build_depth_surface(depth_dir, ds, run_dir, run_dir / "surface" / "v1")
    assert rec.depth_source == "minegs_render"

    out = tmp_path / "geo.json"
    r = runner.invoke(app, geometry_argv(surface_dir, synthetic, out))
    assert r.exit_code == 0, r.output
    assert json.loads(out.read_text())["claim"] == "geometry_accuracy"
    assert (surface_dir / SURFACE_FILE).is_file()


# ---------------------------------------------- render-critical settings this renderer skips


@pytest.mark.parametrize(
    ("witness", "kwargs"),
    [
        ("command", {"command": ["python", "simple_trainer.py", "--pose_opt"]}),
        ("profile", {"profile": {"name": "x", "requests": {"pose_refinement": True}}}),
        ("command", {"command": ["python", "simple_trainer.py", "--antialiased"]}),
        ("profile", {"profile": {"name": "x", "requests": {"antialiasing": True}}}),
    ],
)
def test_a_run_this_renderer_cannot_reproduce_is_refused(dataset_small, tmp_path, witness, kwargs):
    """pose_opt moves the cameras; antialiased changes how opacity composites into depth."""
    ds = dataset_small.dataset_dir
    run_dir = tmp_path / "runs" / witness
    make_run(ds, run_dir, run_id=witness, **kwargs)
    with pytest.raises(ContractError, match="render-critical"):
        render_depths(run_dir, ds, tmp_path / "d", renderer=FakeRenderer())
    assert not (tmp_path / "d").exists()


def test_the_trainers_own_config_is_a_witness_too(env, tmp_path):
    """A record that forgot the flag does not make the run reproducible."""
    cfg = env.run_dir / "backend_out" / "cfg.yml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("max_steps: 7000\npose_opt: True\nantialiased: False\n")
    with pytest.raises(ContractError, match=r"cfg\.yml records pose_opt"):
        render_depths(env.run_dir, env.dataset_dir, tmp_path / "d", renderer=FakeRenderer())


def test_a_checkpoint_carrying_pose_adjust_is_refused():
    """The last witness, read off the weights themselves rather than any record."""
    ckpt = Path("ckpt_6999_rank0.pt")
    assert check_checkpoint_blob({"splats": _SPLATS}, ckpt) is _SPLATS
    with pytest.raises(ContractError, match="pose_adjust"):
        check_checkpoint_blob({"splats": _SPLATS, "pose_adjust": object()}, ckpt)
    with pytest.raises(ContractError, match="not a gsplat checkpoint"):
        check_checkpoint_blob({"step": 6999}, ckpt)
    with pytest.raises(ContractError, match="missing"):
        check_checkpoint_blob({"splats": {"means": 1}}, ckpt)


_SPLATS = {"means": 1, "quats": 1, "scales": 1, "opacities": 1}


def test_a_distorted_camera_model_is_refused(dataset_small, tmp_path):
    """`rasterization` projects pinhole; distortion would be silently dropped from every ray."""
    ds = tmp_path / "ds"
    shutil.copytree(dataset_small.dataset_dir, ds)
    cams = ds / "sparse" / "0" / "cameras.txt"
    rows = []
    for line in cams.read_text().splitlines():
        tok = line.split()
        if line.startswith("#") or not tok:
            rows.append(line)
            continue
        f, cx, cy = tok[4], tok[6], tok[7]
        rows.append(" ".join([tok[0], "SIMPLE_RADIAL", tok[2], tok[3], f, cx, cy, "0.01"]))
    cams.write_text("\n".join(rows) + "\n")

    run_dir = tmp_path / "runs" / "distorted"
    make_run(dataset_small.dataset_dir, run_dir, run_id="distorted")
    with pytest.raises(ContractError, match="distortion"):
        render_depths(run_dir, ds, tmp_path / "d", renderer=FakeRenderer())


# ---------------------------------------------- verified bytes == back-projected bytes


def test_the_manifest_cannot_point_the_verifier_at_a_different_file(env, tmp_path):
    """Hashing one file while back-projecting another would reopen the whole provenance hole."""
    _, depth_dir = render_depths(env.run_dir, env.dataset_dir, env.out, renderer=FakeRenderer())
    manifest = DepthManifest.load(depth_dir / DEPTH_MANIFEST_FILE)
    victim = manifest.depths[0]
    decoy = depth_dir / "decoy.npy"
    shutil.copy(depth_dir / victim.file, decoy)

    # the decoy is a byte-identical copy, so its digest is the recorded one...
    entries = [d.model_dump() for d in manifest.depths]
    entries[0]["file"] = decoy.name
    manifest.depths = entries
    manifest.save(depth_dir / DEPTH_MANIFEST_FILE)
    # ...and the file the fuser will actually read is now free to be anything
    tampered = np.load(depth_dir / victim.file)
    tampered[:] = 9.0
    np.save(depth_dir / victim.file, tampered)

    with pytest.raises(ContractError, match="not the map that would be back-projected"):
        build_depth_surface(depth_dir, env.dataset_dir, env.run_dir, tmp_path / "surface")
    assert not (tmp_path / "surface").exists()


# ---------------------------------------------- checkpoint identity is verified, not just kept


def test_the_manifest_checkpoint_must_be_the_one_the_run_ended_on(env, tmp_path):
    _, depth_dir = render_depths(env.run_dir, env.dataset_dir, env.out, renderer=FakeRenderer())
    where = depth_dir / DEPTH_MANIFEST_FILE

    def repoint(**over):
        m = DepthManifest.load(where)
        m.checkpoint = {**m.checkpoint, **over}
        m.save(where)

    repoint(file="ckpt/ckpt_3000_rank0.pt")
    with pytest.raises(ContractError, match="ended on"):
        build_depth_surface(depth_dir, env.dataset_dir, env.run_dir, tmp_path / "s1")

    repoint(file=CKPT_REL, step=3000)
    with pytest.raises(ContractError, match="ended at"):
        build_depth_surface(depth_dir, env.dataset_dir, env.run_dir, tmp_path / "s2")

    # weights replaced after the render: same path, same step, different model
    repoint(step=6999)
    (env.run_dir / CKPT_REL).write_bytes(b"different weights entirely")
    with pytest.raises(ContractError, match="different weights"):
        build_depth_surface(depth_dir, env.dataset_dir, env.run_dir, tmp_path / "s3")


def test_a_backend_with_no_renderer_is_refused(dataset_small, tmp_path):
    """Phase 1B ships gsplat only; another backend's weights are not renderable here."""
    ds = dataset_small.dataset_dir
    run_dir = tmp_path / "runs" / "other_backend"
    make_run(ds, run_dir, run_id="other_backend", backend={"name": "pgsr", "version": "x"})
    with pytest.raises(ContractError, match="no metric depth renderer"):
        render_depths(run_dir, ds, tmp_path / "d1")
    # and a renderer for the wrong backend cannot be substituted in either
    with pytest.raises(ContractError, match="renders gsplat models"):
        render_depths(run_dir, ds, tmp_path / "d2", renderer=FakeRenderer())


def test_unreasoned_backend_args_are_refused_but_the_light_profile_is_not(dataset_small, tmp_path):
    """backend_args reach the trainer verbatim, so an unknown key is refused, not assumed safe.

    `camera_model`, `with_ut` and `far_plane` all change the projection or the frustum and none
    of them is a boolean flag the other witnesses would notice.
    """
    ds = dataset_small.dataset_dir
    light = load_profile("light").model_dump(mode="json")

    ok_dir = tmp_path / "runs" / "light"
    make_run(ds, ok_dir, run_id="light", profile=light)
    _, depth_dir = render_depths(ok_dir, ds, tmp_path / "ok", renderer=FakeRenderer())
    assert (depth_dir / DEPTH_MANIFEST_FILE).is_file()  # a real profile must still render

    for i, (key, value) in enumerate(
        [("camera_model", "fisheye"), ("with_ut", True), ("far_plane", 50.0)]
    ):
        profile = {**light, "backend_args": {**light["backend_args"], key: value}}
        run_dir = tmp_path / "runs" / f"arg{i}"
        make_run(ds, run_dir, run_id=f"arg{i}", profile=profile)
        with pytest.raises(ContractError, match=key):
            render_depths(run_dir, ds, tmp_path / f"bad{i}", renderer=FakeRenderer())
        assert not (tmp_path / f"bad{i}").exists()


def test_promotion_re_checks_reproducibility_it_does_not_trust_the_manifest(env, tmp_path):
    """Depth written by a build without the render guard must not promote on its manifest."""
    _, depth_dir = render_depths(env.run_dir, env.dataset_dir, env.out, renderer=FakeRenderer())

    record = RunRecord.load(env.run_dir / "run.json")
    record.command = ["python", "simple_trainer.py", "--pose_opt"]
    record.save(env.run_dir / "run.json")

    with pytest.raises(ContractError, match="render-critical"):
        build_depth_surface(depth_dir, env.dataset_dir, env.run_dir, tmp_path / "surface")
    assert not (tmp_path / "surface").exists()


def test_accuracy_is_a_claim_about_the_holdout_only(synthetic, tmp_path):
    """--no-holdout-only measures the chainage the run was initialised on: that is not a claim."""
    ds = synthetic.dataset_dir
    run_dir = tmp_path / "runs" / "holdout"
    make_run(ds, run_dir, run_id="holdout")
    _, depth_dir = render_depths(run_dir, ds, renderer=FakeRenderer(depth_m=3.0))
    rec, surface_dir = build_depth_surface(depth_dir, ds, run_dir, run_dir / "surface" / "v1")
    assert rec.depth_source == "minegs_render"

    held, whole = tmp_path / "held.json", tmp_path / "whole.json"
    r = runner.invoke(app, geometry_argv(surface_dir, synthetic, held))
    assert r.exit_code == 0, r.output
    assert json.loads(held.read_text())["claim"] == "geometry_accuracy"

    r = runner.invoke(app, [*geometry_argv(surface_dir, synthetic, whole), "--no-holdout-only"])
    assert r.exit_code == 0, r.output
    report = json.loads(whole.read_text())
    assert report["claim"] == "geometry_diagnostic"
    assert report["chainage_range_m"] is None
    assert "fit to the data the run saw" in r.output


def test_a_reference_cloud_in_the_wrong_frame_is_refused(synthetic, tmp_path):
    """init_points.ply sits one directory from tls_full.ply and is LOCAL_METRIC, not TLS_GLOBAL."""
    ds = synthetic.dataset_dir
    run_dir = tmp_path / "runs" / "frame"
    make_run(ds, run_dir, run_id="frame")
    _, depth_dir = render_depths(run_dir, ds, renderer=FakeRenderer(depth_m=3.0))
    _, surface_dir = build_depth_surface(depth_dir, ds, run_dir, run_dir / "surface" / "v1")

    argv = [
        "eval",
        "geometry",
        str(surface_dir),
        str(ds),
        "--tls-ply",
        str(ds / "init_points.ply"),  # LOCAL_METRIC, and one `ls` away from the real reference
    ]
    r = runner.invoke(app, argv)
    assert r.exit_code == 2, r.output
    assert "LOCAL_METRIC" in r.output and "TLS_GLOBAL" in r.output

    # --diagnostic still accepts it, with the same sentence as a warning. (--no-holdout-only
    # because init_points.ply has the holdout chainage removed by construction, so restricting
    # to it would leave the reference side empty.)
    out = tmp_path / "diag.json"
    r = runner.invoke(app, [*argv, "--diagnostic", "--no-holdout-only", "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert "not TLS_GLOBAL" in r.output
    assert json.loads(out.read_text())["claim"] == "geometry_diagnostic"


def test_the_manifest_records_what_the_run_was_trained_on(env):
    """A max_images subset run renders every view; two artifacts must not look identical."""
    staged = {"n_images": 12, "n_train_available": 40, "subset": True, "init_source": "tls"}
    record = RunRecord.load(env.run_dir / "run.json")
    record.staged = staged
    record.save(env.run_dir / "run.json")

    manifest, _ = render_depths(env.run_dir, env.dataset_dir, env.out, renderer=FakeRenderer())
    assert manifest.staged["subset"] is True
    assert manifest.staged["n_images"] == 12
    # depth is still rendered for every dataset view; the point is that the gap is visible
    assert len(manifest.depths) == len(env.model.images)
