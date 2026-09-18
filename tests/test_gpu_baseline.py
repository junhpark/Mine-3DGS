"""Phase 0D.2 — the local GPU baseline execution contract, verified without a GPU.

A trainer that writes nothing exits 0 exactly like one that trained for seven thousand steps.
So every test here drives the *real* ``LocalRunner`` -> ``GsplatBackend`` -> ``normalize_outputs``
path and replaces only the executable, with a stand-in that writes the artifact layout upstream
gsplat v1.5.3 actually writes (``ckpts/ckpt_<step>_rank<n>.pt``, ``stats/train_step<step>_rank<n>
.json``, ``ply/point_cloud_<step>.ply``, ``renders/``) — or deliberately fails to.

Nothing here claims a GPU run happened. It claims that when one does, a run is called SUCCEEDED
only if it can be shown to have trained.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from minegs.train.runner import RunConfig, get_runner
from minegs.train.runner import local as runner_local
from minegs.train.runner.base import RunnerConfig, RunStatus, load_record

# gsplat writes its last checkpoint at ``max_steps - 1`` (upstream simple_trainer.py: the save
# branch fires on ``step == max_steps - 1``). A baseline that expected ``max_steps`` would fail
# every real run, so the fake models the real off-by-one rather than a convenient one.
FAKE_TRAINER = """
import json, sys, struct
from pathlib import Path

argv = sys.argv[1:]
def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default

result = Path(opt("--result_dir"))
max_steps = int(opt("--max_steps", "10"))
final = max_steps - 1
cfg = json.loads(Path(__file__).with_suffix(".cfg.json").read_text())

if cfg.get("exit_code"):
    sys.exit(cfg["exit_code"])

if cfg.get("write_stats", True):
    (result / "stats").mkdir(parents=True, exist_ok=True)
    step = cfg.get("stop_at", final)
    (result / "stats" / f"train_step{step:04d}_rank0.json").write_text(
        json.dumps({"mem": 3.25, "ellipse_time": 12.5, "num_GS": cfg.get("n", 64)})
    )

if cfg.get("write_ckpt", True):
    (result / "ckpts").mkdir(parents=True, exist_ok=True)
    step = cfg.get("stop_at", final)
    p = result / "ckpts" / f"ckpt_{step}_rank0.pt"
    p.write_bytes(b"" if cfg.get("empty_ckpt") else b"\\x80\\x02}q\\x00.")

if cfg.get("write_ply", True):
    (result / "ply").mkdir(parents=True, exist_ok=True)
    n = cfg.get("n", 64)
    props = (["x","y","z","nx","ny","nz"] + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(45)] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    hdr = "ply\\nformat binary_little_endian 1.0\\nelement vertex %d\\n" % n
    hdr += "".join("property float %s\\n" % q for q in props) + "end_header\\n"
    import random
    random.seed(0)
    rows = []
    span = cfg.get("span", 10.0)
    rot0 = props.index("rot_0")
    for i in range(n):
        v = [random.uniform(-span / 2, span / 2) for _ in range(3)] + [0.0] * (len(props) - 3)
        v[props.index("opacity")] = 0.5
        for k in range(3):
            v[props.index("scale_%d" % k)] = -2.0   # gsplat stores log-scale
        v[rot0] = 1.0                                # unit quaternion (w, x, y, z)
        rows.append(v)
    if cfg.get("nonfinite"):
        rows[0][0] = float("nan")
    body = b"".join(struct.pack("<%df" % len(props), *r) for r in rows)
    (result / "ply" / f"point_cloud_{cfg.get('stop_at', final)}.ply").write_bytes(
        hdr.encode() + body
    )

if cfg.get("write_cfg", True):
    # upstream dumps vars(cfg) with the default Dumper, so strategy carries a python tag
    (result / "cfg.yml").write_text(
        "data_dir: /staged\\n"
        "max_steps: %d\\n" % max_steps
        + "normalize_world_space: %s\\n" % cfg.get("normalize", "false")
        + "strategy: !!python/object:gsplat.strategy.default.DefaultStrategy\\n"
        + "  absgrad: false\\n"
        + "global_scale: 1.0\\n"
    )

if cfg.get("write_renders", True):
    (result / "renders").mkdir(parents=True, exist_ok=True)
    (result / "renders" / ("val_step%04d.png" % final)).write_bytes(b"\\x89PNG\\r\\n\\x1a\\n")
"""


@pytest.fixture
def gpu(monkeypatch):
    """A CUDA device, as far as the runner can tell. The gate is real; the device is not."""
    monkeypatch.setattr(runner_local, "cuda_available", lambda: True)


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    """Install a fake ``simple_trainer.py`` and return a knob to configure what it writes."""
    script = tmp_path / "simple_trainer.py"
    script.write_text(FAKE_TRAINER)
    monkeypatch.setenv("MINEGS_GSPLAT_TRAINER", str(script))

    def configure(**cfg):
        script.with_suffix(".cfg.json").write_text(json.dumps(cfg))
        return script

    configure()
    return configure


def _run(synthetic, tmp_path, **overrides):
    r = get_runner("local", RunnerConfig(runner="local", native=True))
    run_dir = tmp_path / "runs" / "r1"
    h = r.submit(
        RunConfig(
            dataset_dir=str(synthetic.dataset_dir),
            profile="light",
            run_dir=str(run_dir),
            overrides={"max_steps": 10, "max_images": 4, **overrides},
        )
    )
    return h, run_dir


# ---------------------------------------------------------------- the happy path


def test_a_verified_run_records_what_it_produced(synthetic, tmp_path, gpu, trainer):
    """D2-4 / D2-5 / D2-6 / D2-8 / D2-9: success carries its own evidence."""
    trainer(n=128, span=8.0)
    h, run_dir = _run(synthetic, tmp_path)
    assert h.wait(poll_s=0.01) is RunStatus.SUCCEEDED

    rec = load_record(run_dir)
    assert rec.status is RunStatus.SUCCEEDED and rec.failure_reason is None
    assert rec.max_steps == 10 and rec.observed_final_step == 9  # max_steps - 1, as upstream
    assert rec.final_checkpoint == "ckpt/ckpt_9_rank0.pt"
    assert (run_dir / rec.final_checkpoint).stat().st_size > 0
    assert rec.checkpoint_step == 9
    assert rec.final_model == "point_cloud/point_cloud_9.ply"
    assert (run_dir / rec.final_model).is_file()
    assert rec.gaussian_count == 128
    assert rec.peak_gpu_memory_gb == pytest.approx(3.25)  # from the trainer's own stats json
    assert rec.train_seconds == pytest.approx(12.5)
    assert rec.started_at and rec.completed_at and rec.duration_s is not None
    assert rec.outputs == ["point_cloud/point_cloud_9.ply"]
    assert rec.frame_of_outputs == "LOCAL_METRIC"  # D2-12
    assert rec.extents["output_span_m"] > 0 and "init_span_m" in rec.extents
    # D2-10: a render to look at, named in the record rather than left for the reader to find
    assert rec.renders == ["renders/val_step0009.png"]
    assert (run_dir / rec.renders[0]).is_file()


def test_the_executed_command_is_the_recorded_command(
    synthetic, tmp_path, gpu, trainer, monkeypatch
):
    """D2-7 (tests list): run.json must describe the process that actually ran."""
    seen: list[list[str]] = []
    real_popen = runner_local.subprocess.Popen

    def spy(argv, **kw):
        seen.append(list(argv))
        return real_popen(argv, **kw)

    monkeypatch.setattr(runner_local.subprocess, "Popen", spy)
    h, run_dir = _run(synthetic, tmp_path)
    h.wait(poll_s=0.01)
    # subprocess.run goes through Popen too, so the spy also catches provenance's git and
    # tool-version probes; the trainer is the one handed a --result_dir.
    launched = [a for a in seen if "--result_dir" in a]
    assert len(launched) == 1, seen
    assert load_record(run_dir).command == launched[0]


def test_a_fresh_run_carries_no_checkpoint_argument(synthetic, tmp_path, gpu, trainer):
    """D2-9 (tests list). The 0D.1 guard, asserted on a real assembled run."""
    h, run_dir = _run(synthetic, tmp_path)
    h.wait(poll_s=0.01)
    argv = load_record(run_dir).command
    assert not any(a.startswith("--ckpt") for a in argv)
    assert "--no-normalize_world_space" in argv  # D2-11: no hidden normalisation
    assert not any(a.lstrip("-") == "normalize_world_space" for a in argv)


# ---------------------------------------------------------------- fail closed


def test_no_cuda_refuses_rather_than_training_on_the_cpu(synthetic, tmp_path, trainer, monkeypatch):
    """D2-1 (tests list). A CPU baseline is not a slower GPU baseline, it is another experiment."""
    from minegs.core.errors import NoGpuError

    monkeypatch.setattr(runner_local, "cuda_available", lambda: False)
    with pytest.raises(NoGpuError, match="No CUDA device"):
        _run(synthetic, tmp_path)


def test_a_trainer_that_fails_makes_a_failed_run(synthetic, tmp_path, gpu, trainer):
    """D2-2 (tests list)."""
    trainer(exit_code=3)
    h, run_dir = _run(synthetic, tmp_path)
    assert h.wait(poll_s=0.01) is RunStatus.FAILED
    rec = load_record(run_dir)
    assert rec.status is RunStatus.FAILED and "non-zero" in (rec.failure_reason or "")


@pytest.mark.parametrize(
    ("cfg", "needle"),
    [
        ({"write_ckpt": False}, "wrote no checkpoint"),
        ({"empty_ckpt": True}, "is empty"),
        ({"write_ply": False}, "produced no PLY"),
        ({"nonfinite": True}, "not finite"),
        ({"stop_at": 3}, "stopped at step 3"),
        ({"span": 0.05}, "beyond the 20.0x"),
    ],
)
def test_exit_zero_without_the_artifacts_is_not_success(
    synthetic, tmp_path, gpu, trainer, cfg, needle
):
    """D2-3 / D2-4 / D2-5 (tests list) plus D2-4 progression and D2-7 frame.

    Every case here is a trainer that exits 0. That is the whole point: the exit code is the
    weakest signal in the run, so none of these may reach SUCCEEDED.
    """
    trainer(n=64, **cfg)
    h, run_dir = _run(synthetic, tmp_path)
    assert h.wait(poll_s=0.01) is RunStatus.FAILED
    rec = load_record(run_dir)
    assert rec.status is RunStatus.FAILED
    assert needle in (rec.failure_reason or ""), rec.failure_reason
    assert rec.outputs == []  # a failed run advertises no artifacts


def test_a_failed_verification_stays_failed_on_every_later_call(synthetic, tmp_path, gpu, trainer):
    """D2-13 (tests list): the terminal status is resolved once, not recomputed from the exit code.

    A latch that only remembers *that* it finalised lets the second call fall through to the
    process's own exit status — which for these runs is 0 — and report SUCCEEDED for a run that
    was just refused.
    """
    trainer(write_ckpt=False)
    h, run_dir = _run(synthetic, tmp_path)
    assert h.wait(poll_s=0.01) is RunStatus.FAILED
    assert h.status() is RunStatus.FAILED
    assert h.status() is RunStatus.FAILED
    assert load_record(run_dir).status is RunStatus.FAILED


def test_an_unfinished_run_is_not_mistakable_for_a_finished_one(synthetic, tmp_path, gpu, trainer):
    """D2-13 (tests list) and §12: a run nobody finalised reads as RUNNING, never SUCCEEDED."""
    h, run_dir = _run(synthetic, tmp_path)
    # the record as it stands the moment the trainer was launched, before anyone asked
    rec = load_record(run_dir)
    assert rec.status is RunStatus.RUNNING
    assert rec.outputs == [] and rec.final_model is None and rec.completed_at is None
    h.wait(poll_s=0.01)


# ---------------------------------------------------------------- provenance


def test_identity_and_runtime_are_recorded(synthetic, tmp_path, gpu, trainer):
    """D2-8 (tests list): git SHA, dataset identity, image, runtime description."""
    h, run_dir = _run(synthetic, tmp_path)
    h.wait(poll_s=0.01)
    rec = load_record(run_dir)
    assert rec.dataset_id and len(rec.dataset_hash) == 64
    assert rec.provenance.git_commit
    assert rec.runner == "local" and rec.backend["name"] == "gsplat"
    assert "torch" in rec.runtime  # best-effort description, collected before the trainer ran
    assert rec.staged["init_source"] == "init_points.ply"


def test_the_docker_path_pins_the_image_by_digest(synthetic, tmp_path, gpu, trainer, monkeypatch):
    """D2-1: a tag is not a pin. The digest is what makes a run repeatable."""
    from minegs.core.errors import ContractError

    monkeypatch.setattr(runner_local, "docker_available", lambda: True)
    r = get_runner("local", RunnerConfig(runner="local", native=False, image="minegs:gpu"))
    with pytest.raises(ContractError, match="pinned by digest"):
        r.submit(
            RunConfig(
                dataset_dir=str(synthetic.dataset_dir),
                profile="light",
                run_dir=str(tmp_path / "runs" / "d1"),
            )
        )


def test_evidence_is_read_back_from_the_artifacts_not_the_profile(tmp_path):
    """D2-4/D2-5/D2-9: ``collect_evidence`` discovers, it does not assume.

    The step it reports is the one on disk. A profile asking for 7000 steps whose output stops at
    3 must report 3, or the check that catches a short run has nothing to catch it with.
    """
    from minegs.train.backends.gsplat import GsplatBackend
    from minegs.train.profiles import load_profile

    work = tmp_path / "backend_out"
    (work / "ckpts").mkdir(parents=True)
    (work / "stats").mkdir(parents=True)
    (work / "ckpts" / "ckpt_3_rank0.pt").write_bytes(b"x")
    (work / "stats" / "train_step0003_rank0.json").write_text(
        json.dumps({"mem": 1.5, "ellipse_time": 2.0, "num_GS": 7})
    )
    prof = load_profile("light")
    prof.max_steps = 7000
    ev = GsplatBackend().collect_evidence(work, prof)
    assert ev.observed_final_step == 3 and ev.configured_max_steps == 7000
    assert ev.checkpoint_step == 3 and ev.final_checkpoint.name == "ckpt_3_rank0.pt"
    assert ev.peak_gpu_memory_gb == 1.5 and ev.gaussian_count == 7
    assert ev.final_model is None  # no ply/ was written, and it does not invent one


def test_the_final_model_is_the_latest_step_not_the_last_name(tmp_path):
    """``point_cloud_10.ply`` sorts before ``point_cloud_9.ply`` as a string."""
    from minegs.train.backends.gsplat import GsplatBackend
    from minegs.train.profiles import load_profile

    work = tmp_path / "backend_out"
    (work / "ply").mkdir(parents=True)
    for step in (9, 10, 100):
        (work / "ply" / f"point_cloud_{step}.ply").write_bytes(b"ply\n")
    ev = GsplatBackend().collect_evidence(work, load_profile("light"))
    assert ev.final_model.name == "point_cloud_100.ply"


def test_gaussian_extent_is_measured_against_the_initialisation(synthetic, tmp_path, gpu, trainer):
    """D2-7: the recorded spans are the evidence behind the frame claim, not a bare verdict."""
    trainer(n=256, span=6.0)
    h, run_dir = _run(synthetic, tmp_path)
    assert h.wait(poll_s=0.01) is RunStatus.SUCCEEDED
    ext = load_record(run_dir).extents
    assert ext["unit"] == "m"
    assert ext["output_span_m"] == pytest.approx(6.0, abs=1.5)
    assert ext["init_span_m"] > 0 and ext["camera_span_m"] > 0
    assert 1 / 20.0 < ext["output_over_init"] < 20.0


def test_the_trainers_own_config_is_what_settles_the_frame_question(
    synthetic, tmp_path, gpu, trainer
):
    """D2-7 / D2-11: gsplat defaults normalize_world_space to True, so its cfg.yml is the record.

    The extent ratio is corroboration — blind to rotation and translation, and thin-margined
    against a real normalisation. What the trainer wrote down about its own configuration is
    not. A run whose cfg.yml says it normalised is refused even when the output still measures
    plausibly, which is precisely the case a ratio would wave through.
    """
    trainer(n=64, span=8.0, normalize="true")
    h, run_dir = _run(synthetic, tmp_path)
    assert h.wait(poll_s=0.01) is RunStatus.FAILED
    rec = load_record(run_dir)
    assert "normalize_world_space" in (rec.failure_reason or "")
    assert "arbitrary units" in (rec.failure_reason or "")
    assert rec.extents["trainer_normalize_world_space"] == "true"


def test_a_normalised_scene_is_refused_even_though_the_trainer_exited_zero(
    synthetic, tmp_path, gpu, trainer
):
    """D2-7 / failure semantics: a scene rescaled to unit size is what normalisation looks like."""
    trainer(n=64, span=0.02)
    h, run_dir = _run(synthetic, tmp_path)
    assert h.wait(poll_s=0.01) is RunStatus.FAILED
    rec = load_record(run_dir)
    assert "normalised the scene" in (rec.failure_reason or "")
    assert np.isfinite(rec.extents["output_over_init"])


def test_the_default_cli_invocation_finalises_the_run(synthetic, tmp_path, gpu, trainer):
    """A verification that only runs when someone asks is not a contract.

    Finalisation happens when the handle reaches a terminal state, so a CLI that returned
    immediately left ``run.json`` at 'running' with ``point_cloud/`` never written, however the
    trainer ended — and the default invocation could therefore never satisfy D2-8. Waiting is
    the default for that reason; ``--no-wait`` still exists and says what it costs.
    """
    from minegs.cli.main import app
    from typer.testing import CliRunner

    r = CliRunner().invoke(
        app, ["train", "run", str(synthetic.dataset_dir), "--profile", "light", "--native"]
    )
    assert r.exit_code == 0, r.output
    out = " ".join(r.output.split())
    assert "status: succeeded" in out

    run_dir = Path(out.split(" -> ", 1)[1].split(" status:", 1)[0].strip())
    rec = load_record(run_dir)
    assert rec.status is RunStatus.SUCCEEDED
    assert rec.outputs and (run_dir / rec.outputs[0]).is_file()
    assert rec.completed_at and rec.final_checkpoint


def test_no_wait_says_the_run_is_unverified_rather_than_implying_success(
    synthetic, tmp_path, gpu, trainer
):
    """The old default, kept — but it no longer looks like a finished run."""
    from minegs.cli.main import app
    from typer.testing import CliRunner

    r = CliRunner().invoke(
        app,
        ["train", "run", str(synthetic.dataset_dir), "--profile", "light", "--native", "--no-wait"],
    )
    assert r.exit_code == 0, r.output
    out = " ".join(r.output.split())
    assert "will not be verified" in out and "status: succeeded" not in out
