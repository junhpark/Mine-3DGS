"""Phase 0D.1 — resume target, checkpoint resolution and host↔container namespaces.

The failure these tests exist to prevent is silent: a requested resume that starts training
from iteration 0 produces a plausible run, a plausible PLY and a run.json claiming a parent.
Nothing raises, nothing looks wrong, and the experiment is not the one that was asked for.
So most assertions here are about *refusals* and about what is absent from an argv.

``gsplat`` v1.5.3 cannot continue training at all (see ``GsplatBackend`` and
``docs/ROADMAP.md`` §Phase 0D), so the runner-level namespace tests drive a minimal
resume-capable backend. That is the layer the entry blocker lived in: the translation, the
mount and the guard are the runner's, and they are exercised for real here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from minegs.core.errors import ContractError
from minegs.core.frames import Sim3
from minegs.train.backends.base import BackendCapabilities, TrainBackend, TrainCommand
from minegs.train.backends.gsplat import RESUME_REFUSAL, GsplatBackend
from minegs.train.profiles import load_profile
from minegs.train.runner import RunConfig, get_runner
from minegs.train.runner import base as runner_base
from minegs.train.runner import local as runner_local
from minegs.train.runner.base import RunnerConfig, load_record
from minegs.train.runner.resume import (
    CONTAINER_RESUME_DIR,
    check_compatibility,
    checkpoint_dir,
    container_checkpoint,
    guard_resume_argv,
    resolve_resume_checkpoint,
    validate_checkpoint,
)

IMAGE = "ghcr.io/example/minegs@sha256:" + "0" * 64


class ResumableBackend(TrainBackend):
    """A backend that *can* continue training, so the runner's resume path is reachable.

    Deliberately not gsplat: the pinned trainer has no resume path, and pretending otherwise
    in a test would assert a contract the shipped system does not have.
    """

    name = "resumable"

    def version(self) -> str:
        return "test"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(resume=True)

    def build_command(self, dataset_dir, out_dir, profile, resume_checkpoint=None, **kwargs):
        argv = [
            "python",
            "trainer.py",
            "--data_dir",
            str(dataset_dir),
            "--result_dir",
            str(out_dir),
        ]
        if resume_checkpoint is not None:
            argv += ["--ckpt", str(resume_checkpoint)]
        return TrainCommand(argv=argv, T_local_from_internal=Sim3.identity())

    def normalize_outputs(self, out_dir, run_dir, T_local_from_internal):
        return []


class RecordingPopen:
    """Stands in for ``subprocess.Popen`` and counts every attempt to start the trainer."""

    calls: list[list[str]] = []

    def __init__(self, argv, **kwargs):
        RecordingPopen.calls.append(list(argv))
        self.argv = list(argv)

    def poll(self):
        return 0


def no_exec(monkeypatch):
    """Replace ``subprocess`` *inside the runner module only*.

    Patching ``subprocess.Popen`` globally would also break ``subprocess.run``, which
    provenance stamping uses — and a test that cannot tell "the trainer started" from "git
    rev-parse ran" cannot assert that nothing was executed.
    """
    shim = SimpleNamespace(Popen=RecordingPopen, STDOUT=subprocess.STDOUT)
    monkeypatch.setattr(runner_local, "subprocess", shim)


@pytest.fixture
def runner_env(monkeypatch):
    """A LocalRunner whose backend can resume, whose host has a GPU, and that never execs."""
    backend = ResumableBackend()
    monkeypatch.setattr(runner_base, "get_backend", lambda name: backend)
    monkeypatch.setattr(runner_local, "get_backend", lambda name: backend)
    monkeypatch.setattr(runner_local, "cuda_available", lambda: True)
    monkeypatch.setattr(runner_local, "docker_available", lambda: True)
    RecordingPopen.calls = []
    no_exec(monkeypatch)
    return backend


def submit(dataset_dir: Path, run_dir: Path, *, native: bool = True, **cfg):
    r = get_runner("local", RunnerConfig(runner="local", native=native, image=IMAGE))
    return r.submit(
        RunConfig(dataset_dir=str(dataset_dir), profile="light", run_dir=str(run_dir), **cfg)
    )


def make_parent(dataset_dir: Path, run_dir: Path, iterations=(200,), *, native: bool = True):
    """A real parent run (real run.json, real staged hash) plus checkpoints on disk."""
    submit(dataset_dir, run_dir, native=native)
    ckpts = run_dir / "backend_out" / "ckpts"
    ckpts.mkdir(parents=True)
    for it in iterations:
        (ckpts / f"ckpt_{it}.pt").write_bytes(b"weights-%d" % it)
    RecordingPopen.calls = []
    return run_dir


# --------------------------------------------------------------- §36 checkpoint resolver


def test_latest_checkpoint_is_chosen_by_iteration(tmp_path):
    d = tmp_path / "ckpts"
    d.mkdir()
    for name in ("ckpt_100.pt", "ckpt_200.pt"):
        (d / name).write_bytes(b"x")
    assert resolve_resume_checkpoint(d) == (d / "ckpt_200.pt", 200)


def test_checkpoint_order_is_numeric_not_lexical(tmp_path):
    """``sorted()`` puts ckpt_9.pt after ckpt_10.pt — that is a wrong resume, not a crash."""
    d = tmp_path / "ckpts"
    d.mkdir()
    for name in ("ckpt_9.pt", "ckpt_10.pt"):
        (d / name).write_bytes(b"x")
    assert sorted(p.name for p in d.iterdir())[-1] == "ckpt_9.pt"  # the trap this avoids
    assert resolve_resume_checkpoint(d) == (d / "ckpt_10.pt", 10)


def test_gsplat_rank_suffixed_names_are_understood(tmp_path):
    """v1.5.3 writes ckpt_{step}_rank{n}.pt; a single-rank run must still resolve."""
    d = tmp_path / "ckpts"
    d.mkdir()
    (d / "ckpt_6999_rank0.pt").write_bytes(b"x")
    assert resolve_resume_checkpoint(d) == (d / "ckpt_6999_rank0.pt", 6999)


def test_missing_checkpoint_directory_is_refused(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    with pytest.raises(ContractError, match="no checkpoint directory"):
        checkpoint_dir(parent)
    with pytest.raises(ContractError, match="no checkpoint directory"):
        resolve_resume_checkpoint(tmp_path / "nope")


def test_empty_checkpoint_directory_is_refused(tmp_path):
    d = tmp_path / "ckpts"
    d.mkdir()
    (d / "events.tfevents").write_bytes(b"x")  # present, but not a checkpoint
    with pytest.raises(ContractError, match="no checkpoint file"):
        resolve_resume_checkpoint(d)


def test_unparseable_checkpoint_names_fail_closed(tmp_path):
    """Skipping an unrecognised name could resume an older iteration and report success."""
    d = tmp_path / "ckpts"
    d.mkdir()
    (d / "ckpt_final.pt").write_bytes(b"x")
    with pytest.raises(ContractError, match="refusing to guess"):
        resolve_resume_checkpoint(d)
    (d / "ckpt_100.pt").write_bytes(b"x")  # a parseable one does not excuse the other
    with pytest.raises(ContractError, match=r"ckpt_final\.pt"):
        resolve_resume_checkpoint(d)


def test_multi_rank_checkpoint_is_refused(tmp_path):
    d = tmp_path / "ckpts"
    d.mkdir()
    for rank in (0, 1):
        (d / f"ckpt_500_rank{rank}.pt").write_bytes(b"x")
    with pytest.raises(ContractError, match="distributed"):
        resolve_resume_checkpoint(d)


def test_checkpoint_directory_falls_back_to_run_layout(tmp_path):
    """A finished run keeps its checkpoints in ckpt/ as well; an interrupted one only in
    backend_out/ckpts. Search order is fixed, so the source is never ambiguous."""
    parent = tmp_path / "parent"
    (parent / "ckpt").mkdir(parents=True)
    assert checkpoint_dir(parent) == parent / "ckpt"
    (parent / "backend_out" / "ckpts").mkdir(parents=True)
    assert checkpoint_dir(parent) == parent / "backend_out" / "ckpts"


def test_checkpoint_leaving_the_parent_run_is_refused(tmp_path):
    parent = tmp_path / "parent"
    ckpts = parent / "backend_out" / "ckpts"
    ckpts.mkdir(parents=True)
    outside = tmp_path / "elsewhere.pt"
    outside.write_bytes(b"x")
    link = ckpts / "ckpt_100.pt"
    link.symlink_to(outside)
    with pytest.raises(ContractError, match="outside the parent run"):
        validate_checkpoint(link, parent)
    real = ckpts / "ckpt_200.pt"
    real.write_bytes(b"xy")
    assert validate_checkpoint(real, parent) == 2
    dangling = ckpts / "ckpt_300.pt"
    dangling.symlink_to(ckpts / "gone.pt")
    with pytest.raises(ContractError, match="not a regular file"):
        validate_checkpoint(dangling, parent)
    # ...and a dangling link is not a candidate in the first place, so it cannot be selected
    assert resolve_resume_checkpoint(ckpts) == (real, 200)


# ------------------------------------------------------------------- §16 argv invariant


def test_guard_refuses_a_command_that_lost_its_checkpoint():
    ckpt = Path("/data/resume/ckpt_200.pt")
    guard_resume_argv(["python", "t.py", "--ckpt", str(ckpt)], ckpt)  # ok
    guard_resume_argv(["python", "t.py"], None)  # fresh run: nothing to check
    for argv in (
        ["python", "t.py"],
        ["python", "t.py", "--ckpt", "/data/resume/ckpt_100.pt"],
        ["python", "t.py", "--ckpt"],
    ):
        with pytest.raises(ContractError, match="resume was requested"):
            guard_resume_argv(argv, ckpt)


# ------------------------------------------------------- §37/§38 namespace translation


def test_docker_resume_mounts_the_parent_read_only_and_uses_a_container_path(
    synthetic, tmp_path, runner_env
):
    parent = make_parent(synthetic.dataset_dir, tmp_path / "runs" / "parent", native=False)
    submit(
        synthetic.dataset_dir,
        tmp_path / "runs" / "child",
        native=False,
        resume_from=str(parent),
    )
    argv = RecordingPopen.calls[-1]
    host_ckpt = parent / "backend_out" / "ckpts" / "ckpt_200.pt"
    assert argv[:2] == ["docker", "run"]
    assert f"{host_ckpt.parent}:{CONTAINER_RESUME_DIR}:ro" in argv
    assert argv[argv.index("--ckpt") + 1] == "/data/resume/ckpt_200.pt"
    # The host path may only appear in the mount; a trainer argument carrying it would be a
    # path that does not exist inside the container.
    trainer_args = argv[argv.index(IMAGE) + 1 :]
    assert not any(str(host_ckpt) in a for a in trainer_args)
    assert str(host_ckpt) not in " ".join(a for a in argv if not a.endswith(":ro"))


def test_native_resume_uses_the_host_path_unchanged(synthetic, tmp_path, runner_env):
    parent = make_parent(synthetic.dataset_dir, tmp_path / "runs" / "parent")
    submit(synthetic.dataset_dir, tmp_path / "runs" / "child", resume_from=str(parent))
    argv = RecordingPopen.calls[-1]
    host_ckpt = parent / "backend_out" / "ckpts" / "ckpt_200.pt"
    assert argv[argv.index("--ckpt") + 1] == str(host_ckpt)
    assert str(CONTAINER_RESUME_DIR) not in " ".join(argv)
    assert "docker" not in argv


def test_a_colon_in_the_host_path_is_refused_rather_than_mounted_wrong(tmp_path):
    """``docker -v`` splits on colons, so such a source would mount something else entirely."""
    from minegs.train.runner.resume import docker_resume_mount

    odd = tmp_path / "run:1" / "backend_out" / "ckpts"
    odd.mkdir(parents=True)
    with pytest.raises(ContractError, match="cannot be expressed as a docker volume"):
        docker_resume_mount(odd / "ckpt_1.pt")
    assert docker_resume_mount(tmp_path / "ckpts" / "ckpt_1.pt")[0] == "-v"


def test_container_path_is_derived_from_the_mount_point_only():
    assert container_checkpoint(Path("/anywhere/runs/p/backend_out/ckpts/ckpt_7.pt")) == (
        CONTAINER_RESUME_DIR / "ckpt_7.pt"
    )


# ----------------------------------------------------------- §39 fail before execution


def test_resume_without_a_checkpoint_fails_before_any_process_starts(
    synthetic, tmp_path, runner_env
):
    parent = tmp_path / "runs" / "parent"
    submit(synthetic.dataset_dir, parent)  # a real run, but it never wrote a checkpoint
    RecordingPopen.calls = []
    child = tmp_path / "runs" / "child"
    with pytest.raises(ContractError, match="no checkpoint directory"):
        submit(synthetic.dataset_dir, child, resume_from=str(parent))
    assert RecordingPopen.calls == []
    assert not child.exists()  # §28: a refused resume leaves nothing behind


def test_unknown_parent_run_is_refused(synthetic, tmp_path, runner_env):
    with pytest.raises(ContractError, match="no such run directory"):
        submit(synthetic.dataset_dir, tmp_path / "c1", resume_from=str(tmp_path / "ghost"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ContractError, match=r"no run\.json"):
        submit(synthetic.dataset_dir, tmp_path / "c2", resume_from=str(empty))
    assert RecordingPopen.calls == []


# --------------------------------------------------------------- §40 fresh regression


def test_fresh_run_never_picks_up_a_checkpoint_lying_around(synthetic, tmp_path, runner_env):
    """Resume is requested, never inferred: checkpoints in the run directory change nothing."""
    run_dir = tmp_path / "runs" / "fresh"
    (run_dir / "backend_out" / "ckpts").mkdir(parents=True)
    (run_dir / "backend_out" / "ckpts" / "ckpt_900.pt").write_bytes(b"x")
    submit(synthetic.dataset_dir, run_dir)
    argv = RecordingPopen.calls[-1]
    assert "--ckpt" not in argv
    assert load_record(run_dir).resume is None


def test_fresh_gsplat_command_has_no_checkpoint_flag(synthetic, tmp_path):
    cmd = GsplatBackend().build_command(
        synthetic.dataset_dir, tmp_path / "out", load_profile("light"), check_trainer=False
    )
    assert "--ckpt" not in cmd.argv


# ------------------------------------------------------------------ §41 compatibility


def _parent_with(dataset_dir: Path, run_dir: Path, **edits):
    make_parent(dataset_dir, run_dir)
    rec = load_record(run_dir)
    for key, value in edits.items():
        head, _, tail = key.partition(".")
        if tail:
            getattr(rec, head)[tail] = value
        else:
            setattr(rec, head, value)
    rec.save(run_dir / "run.json")
    return run_dir


@pytest.mark.parametrize(
    ("edits", "message"),
    [
        ({"dataset_hash": "sha256:different"}, "dataset_hash"),
        ({"backend.name": "somethingelse"}, "backend:"),
        ({"chunk_id": "C01"}, "chunk_id"),
        ({"frame_of_outputs": "TLS_GLOBAL"}, "frame_of_outputs"),
        ({"profile.data_factor": 2}, "profile.data_factor"),
        ({"profile.max_images": 7}, "profile.max_images"),
        ({"profile.max_steps": 999999}, "profile.max_steps"),
    ],
)
def test_incompatible_parent_is_refused_and_says_what_differs(
    synthetic, tmp_path, runner_env, edits, message
):
    parent = _parent_with(synthetic.dataset_dir, tmp_path / "runs" / "parent", **edits)
    with pytest.raises(ContractError, match=message):
        submit(synthetic.dataset_dir, tmp_path / "runs" / "child", resume_from=str(parent))
    assert RecordingPopen.calls == []


def test_a_longer_schedule_is_allowed_but_a_shorter_one_is_not(synthetic, tmp_path, runner_env):
    parent = _parent_with(
        synthetic.dataset_dir, tmp_path / "runs" / "parent", **{"profile.max_steps": 100}
    )
    submit(synthetic.dataset_dir, tmp_path / "runs" / "longer", resume_from=str(parent))
    assert "--ckpt" in RecordingPopen.calls[-1]


def test_staged_dataset_mismatch_is_refused(synthetic, tmp_path, runner_env):
    """A matching dataset_hash is not enough: staging policy decides what the trainer sees."""
    parent = _parent_with(
        synthetic.dataset_dir, tmp_path / "runs" / "parent", **{"staged.sha256": "sha256:other"}
    )
    with pytest.raises(ContractError, match=r"staged\.sha256"):
        submit(synthetic.dataset_dir, tmp_path / "runs" / "child", resume_from=str(parent))
    assert RecordingPopen.calls == []
    parent2 = _parent_with(
        synthetic.dataset_dir, tmp_path / "runs" / "p2", **{"staged.init_source": "points3D.txt"}
    )
    with pytest.raises(ContractError, match=r"staged\.init_source"):
        submit(synthetic.dataset_dir, tmp_path / "runs" / "c2", resume_from=str(parent2))


def test_compatibility_reports_every_mismatch_at_once(tmp_path):
    from minegs.core.provenance import stamp
    from minegs.train.runner.base import RunRecord

    parent = RunRecord(
        run_id="p",
        dataset_id="d",
        dataset_hash="sha256:a",
        backend={"name": "resumable", "version": "test"},
        profile={"max_steps": 100, "data_factor": 4, "backend_args": {"strategy": "default"}},
        runner="local",
        provenance=stamp(),
    )
    with pytest.raises(ContractError) as e:
        check_compatibility(
            parent,
            backend_name="other",
            dataset_hash="sha256:b",
            chunk_id="C01",
            profile_dump={"max_steps": 100, "data_factor": 2, "backend_args": {"strategy": "mcmc"}},
        )
    for expected in ("backend:", "dataset_hash", "chunk_id", "data_factor", "strategy"):
        assert expected in str(e.value)


# --------------------------------------------------------------------- §42 provenance


def test_resumed_run_records_its_lineage(synthetic, tmp_path, runner_env):
    parent = make_parent(synthetic.dataset_dir, tmp_path / "runs" / "parent", iterations=(100, 700))
    child = tmp_path / "runs" / "child"
    submit(synthetic.dataset_dir, child, resume_from=str(parent))
    rec = load_record(child)
    host_ckpt = parent / "backend_out" / "ckpts" / "ckpt_700.pt"
    assert rec.resume is not None and rec.resume.requested
    assert rec.resume.parent_run_id == load_record(parent).run_id
    assert rec.resume.parent_run_dir == str(parent)
    assert rec.resume.checkpoint == str(host_ckpt)
    assert rec.resume.checkpoint_exec_path == str(host_ckpt)  # native run
    assert rec.resume.checkpoint_iteration == 700
    import hashlib

    assert rec.resume.checkpoint_sha256 == hashlib.sha256(host_ckpt.read_bytes()).hexdigest()
    # The checkpoint changes the result, so it is a source asset, and the parent is a lineage
    # parent — resume must be readable from provenance, not only from the resume block (§9).
    assert rec.resume.parent_run_id in rec.provenance.parent_ids
    assets = {a.path: a.sha256 for a in rec.provenance.source_assets}
    assert assets[str(host_ckpt)] == rec.resume.checkpoint_sha256


def test_docker_run_records_both_namespaces(synthetic, tmp_path, runner_env):
    parent = make_parent(synthetic.dataset_dir, tmp_path / "runs" / "parent", native=False)
    child = tmp_path / "runs" / "child"
    submit(synthetic.dataset_dir, child, native=False, resume_from=str(parent))
    resume = load_record(child).resume
    assert resume.checkpoint == str(parent / "backend_out" / "ckpts" / "ckpt_200.pt")
    assert resume.checkpoint_exec_path == "/data/resume/ckpt_200.pt"


# -------------------------------------------------------- gsplat cannot resume (v1.5.3)


def test_gsplat_does_not_claim_resume():
    be = GsplatBackend()
    assert be.capabilities().resume is False
    assert "evaluation only" in be.capability_notes["resume"]


def test_gsplat_refuses_a_resume_checkpoint(synthetic, tmp_path):
    with pytest.raises(ContractError, match="skip training and run evaluation only"):
        GsplatBackend().build_command(
            synthetic.dataset_dir,
            tmp_path / "out",
            load_profile("light"),
            resume_checkpoint=Path("/data/resume/ckpt_1.pt"),
            check_trainer=False,
        )


def test_gsplat_refuses_a_checkpoint_smuggled_through_backend_args(synthetic, tmp_path):
    prof = load_profile("light")
    prof.backend_args["ckpt"] = "/data/resume/ckpt_1.pt"
    with pytest.raises(ContractError, match="evaluation only"):
        GsplatBackend().build_command(
            synthetic.dataset_dir, tmp_path / "out", prof, check_trainer=False
        )


def test_resume_from_against_gsplat_fails_with_the_upstream_reason(
    synthetic, tmp_path, monkeypatch
):
    monkeypatch.setattr(runner_local, "cuda_available", lambda: True)
    RecordingPopen.calls = []
    no_exec(monkeypatch)
    parent = tmp_path / "runs" / "parent"
    (parent / "backend_out" / "ckpts").mkdir(parents=True)
    with pytest.raises(ContractError, match="does not support resuming training"):
        submit(synthetic.dataset_dir, tmp_path / "runs" / "child", resume_from=str(parent))
    assert RecordingPopen.calls == []
    assert RESUME_REFUSAL  # the message the refusal quotes is the upstream contract
