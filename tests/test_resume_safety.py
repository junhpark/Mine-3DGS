"""Phase 0D.1 — a requested resume must fail closed, and a fresh run must stay fresh.

gsplat v1.5.3 cannot continue training. Verified against upstream ``examples/simple_trainer.py``
(sha256 79319e1cd7404e4d1ba0c425634235c39e6054f0643b904a01feea6179462c05):

* ``Config.ckpt`` — *"Path to the .pt files. If provide, it will skip training and run
  evaluation only."*
* ``main()`` — ``if cfg.ckpt is not None:`` runs eval/render_traj and returns, ``else:`` runs
  ``train()``. The two are mutually exclusive.
* ``train()`` sets ``init_step = 0`` unconditionally and loads nothing.
* the saved ``.pt`` holds only ``step`` and ``splats`` (plus pose/appearance modules) — no
  optimizer moments, no densification-strategy state.

Two failures follow, and both are silent, which is why they are tested rather than reasoned
about. Requesting a resume and getting iteration 0 back produces a plausible run under a run id
that claims to continue another. Passing ``--ckpt`` to a training run produces an *evaluation*
pass on the parent's weights and reports success. Neither raises anything on its own.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from minegs.core.errors import ContractError, NotYetImplementedError
from minegs.train.backends.base import BackendCapabilities
from minegs.train.backends.gsplat import (
    RESUME_REFUSAL,
    GsplatBackend,
    _assert_no_refused_flags,
    canonical_option,
)
from minegs.train.profiles import load_profile
from minegs.train.runner import RunConfig, get_runner
from minegs.train.runner import local as runner_local
from minegs.train.runner.base import RunnerConfig, refuse_resume


class RecordingPopen:
    """Stands in for ``subprocess.Popen`` and counts every attempt to start the trainer."""

    calls: list[list[str]] = []

    def __init__(self, argv, **kwargs):
        RecordingPopen.calls.append(list(argv))

    def poll(self):
        return 0


@pytest.fixture
def no_exec(monkeypatch):
    """Replace ``subprocess`` *inside the runner module only*.

    Patching ``subprocess.Popen`` globally would also break ``subprocess.run``, which provenance
    stamping uses — and a test that cannot tell "the trainer started" from "git rev-parse ran"
    cannot assert that nothing was executed.
    """
    RecordingPopen.calls = []
    monkeypatch.setattr(runner_local, "cuda_available", lambda: True)
    monkeypatch.setattr(
        runner_local, "subprocess", SimpleNamespace(Popen=RecordingPopen, STDOUT=subprocess.STDOUT)
    )
    return RecordingPopen


def test_gsplat_does_not_claim_training_resume():
    be = GsplatBackend()
    assert be.capabilities().resume is False
    note = be.capability_notes["resume"]
    assert "evaluation only" in note and "init_step = 0" in note


def test_resume_request_fails_before_the_trainer_and_leaves_no_run_directory(
    synthetic, tmp_path, no_exec
):
    run_dir = tmp_path / "runs" / "child"
    r = get_runner("local", RunnerConfig(runner="local", native=True))
    with pytest.raises(ContractError, match="does not support resuming training") as e:
        r.submit(
            RunConfig(
                dataset_dir=str(synthetic.dataset_dir),
                profile="light",
                run_dir=str(run_dir),
                resume_from=str(tmp_path / "runs" / "parent"),
            )
        )
    assert "evaluation only" in str(e.value)  # the refusal quotes the upstream contract
    assert no_exec.calls == []
    assert not run_dir.exists()


def test_a_backend_declaring_resume_still_has_nowhere_to_resume_from():
    """Deliberately unimplemented rather than partially implemented.

    Continuing a run needs a checkpoint that restores the whole training state — optimizer,
    schedulers, strategy state, step, RNG. Mine-3DGS does not own that contract yet, and a
    resume that silently drops half of it is a different experiment, not a slower one.
    """
    claims_resume = SimpleNamespace(
        name="future",
        capabilities=lambda: BackendCapabilities(resume=True),
        capability_notes={},
    )
    with pytest.raises(NotYetImplementedError, match=r"Phase 0D\.3"):
        refuse_resume(claims_resume)


# Every way a checkpoint could be spelled into backend_args. The canonical "ckpt" was refused
# from the start; the rest reached the assembled command as --__ckpt / --ckpt<space> / --CKPT and
# were forwarded verbatim, which is the eval-only path the refusal exists to close.
CHECKPOINT_SPELLINGS = ("ckpt", "--ckpt", "-ckpt", "ckpt ", " ckpt", "CKPT", "ckpt\t", "--CKPT")


@pytest.mark.parametrize("key", CHECKPOINT_SPELLINGS)
def test_a_checkpoint_cannot_arrive_through_backend_args(synthetic, tmp_path, key):
    prof = load_profile("light")
    prof.backend_args[key] = "/somewhere/ckpt_1000.pt"
    with pytest.raises(ContractError, match="evaluation only"):
        GsplatBackend().build_command(
            synthetic.dataset_dir, tmp_path / "out", prof, check_trainer=False
        )


def test_a_key_that_is_not_an_option_name_is_refused(synthetic, tmp_path):
    """Inner whitespace cannot be a flag, and printed as one it reads as two arguments."""
    prof = load_profile("light")
    prof.backend_args["ck pt"] = "/somewhere/ckpt_1000.pt"
    with pytest.raises(ContractError, match="not an option name"):
        GsplatBackend().build_command(
            synthetic.dataset_dir, tmp_path / "out", prof, check_trainer=False
        )


def test_no_spelling_of_the_checkpoint_flag_survives_command_assembly():
    """The guard scans the assembled argv, so it does not matter which route produced the flag."""
    for argv in (
        ["--ckpt", "/x.pt"],
        ["--ckpt=/x.pt"],
        ["python", "t.py", "--ckpt", "/x.pt"],
        ["--CKPT", "/x.pt"],
        ["-ckpt", "/x.pt"],
        ["--__ckpt", "/x.pt"],  # what a "--ckpt" backend_args key used to render as
        ["-- ckpt", "/x.pt"],
    ):
        with pytest.raises(ContractError, match="evaluation only"):
            _assert_no_refused_flags(argv)
    # Unrelated flags, and the safe direction of a refused one, must still pass.
    for argv in (
        ["python", "t.py", "--max_steps", "7000"],
        ["--strategy.absgrad"],
        ["--no-normalize_world_space"],
        ["--sh_degree", "3"],
    ):
        _assert_no_refused_flags(argv)
    assert "skip training and run evaluation only" in RESUME_REFUSAL


def test_one_spelling_per_option(synthetic, tmp_path):
    """Canonicalisation must fold spellings together, not let a second one shadow the first."""
    assert canonical_option("--depth-loss") == canonical_option("depth_loss") == "depth_loss"
    assert canonical_option("sh_degree") == "sh_degree"  # ordinary keys pass through untouched
    prof = load_profile("light")
    prof.backend_args["--sh-degree"] = 2  # light.yaml already sets sh_degree: 3
    with pytest.raises(ContractError, match="two spellings of the same option"):
        GsplatBackend().build_command(
            synthetic.dataset_dir, tmp_path / "out", prof, check_trainer=False
        )


def test_fresh_gsplat_command_carries_no_checkpoint_argument(synthetic, tmp_path):
    """A checkpoint sitting in out_dir is not discovered: the adapter never looks."""
    out = tmp_path / "out"
    (out / "ckpts").mkdir(parents=True)
    (out / "ckpts" / "ckpt_6999.pt").write_bytes(b"x")
    cmd = GsplatBackend().build_command(
        synthetic.dataset_dir, out, load_profile("light"), check_trainer=False
    )
    assert not any("ckpt" in a.lower() for a in cmd.argv if a.startswith("-"))


def test_a_fresh_run_never_picks_up_a_checkpoint_lying_around(
    synthetic, tmp_path, no_exec, monkeypatch
):
    """Resume is requested, never inferred: checkpoints in the run directory change nothing."""
    trainer = tmp_path / "simple_trainer.py"  # the real one lives in the GPU image
    trainer.write_text("")
    monkeypatch.setenv("MINEGS_GSPLAT_TRAINER", str(trainer))
    run_dir = tmp_path / "runs" / "fresh"
    (run_dir / "backend_out" / "ckpts").mkdir(parents=True)
    (run_dir / "backend_out" / "ckpts" / "ckpt_900.pt").write_bytes(b"x")
    r = get_runner("local", RunnerConfig(runner="local", native=True))
    r.submit(
        RunConfig(dataset_dir=str(synthetic.dataset_dir), profile="light", run_dir=str(run_dir))
    )
    argv = no_exec.calls[-1]
    assert not any(a.startswith("--ckpt") for a in argv)
    assert Path(run_dir / "run.json").exists()
