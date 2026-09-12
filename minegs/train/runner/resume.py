"""Resume target resolution (Phase 0D.1).

Two rules shape this module.

**Checkpoint discovery is a host-filesystem operation and belongs to the Runner layer.**
The Phase 0D entry blocker was a backend that globbed ``out_dir/"ckpts"`` while ``out_dir``
was already the *container* path ``/data/run/backend_out``: the host process looked for a
directory that only exists inside the container, found nothing, and dropped ``--ckpt``.
Only the runner knows the host run directory, the mounts and the execution namespace, so
only the runner resolves checkpoints (§21). A backend receives an already-decided path in
the namespace the command will run in and does one thing with it: emit ``--ckpt`` (§20).

**A requested resume that cannot be honoured is a hard failure.** Never a fresh run, never
an eval-only run. A restart from iteration 0 is not a slow resume, it is a different
experiment, and it would be recorded under a run id that claims to continue another one.
Every branch below that cannot produce exactly one checkpoint raises ``ContractError``.

Checkpoint naming follows the pinned backend, not a guess. gsplat v1.5.3 writes
``ckpt_{step}_rank{world_rank}.pt`` (``examples/simple_trainer.py``); the rank-less
``ckpt_{step}.pt`` form is accepted too. Anything else in the checkpoint directory that
looks like a checkpoint but does not parse is refused rather than skipped — silently
ignoring ``ckpt_final.pt`` could hand back an older iteration while reporting success.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, ClassVar

from minegs.core.config import VersionedModel
from minegs.core.errors import ContractError
from minegs.core.provenance import SourceAsset, sha256_file

# Where the parent run's checkpoints live, in search order. A run that finished has both
# (``normalize_outputs`` copies ``backend_out/ckpts`` to the run-layout ``ckpt/``); a run that
# was interrupted has only the first. First hit wins, so the source is never ambiguous.
CKPT_SUBDIRS = ("backend_out/ckpts", "ckpt")

# Container mount point for the parent's checkpoint directory (read-only). The parent run is
# never mounted writable: a failed child must not be able to damage the run it continues.
CONTAINER_RESUME_DIR = PurePosixPath("/data/resume")

# gsplat v1.5.3: ``f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"``.
CKPT_NAME = re.compile(r"^ckpt_(?P<iteration>\d+)(?:_rank(?P<rank>\d+))?\.pt$")

# Anything the checkpoint directory offers that we might mistake for a checkpoint.
CKPT_GLOB = "ckpt_*"


class ResumeInfo(VersionedModel):
    """``run.json``'s ``resume`` block: the lineage of a resumed run (§9, §11).

    Present only on runs that were asked to resume. Absent (``None``) means a fresh run —
    resume must never be inferred from the command string.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0"
    requested: bool = True
    parent_run_id: str | None = None
    parent_run_dir: str | None = None
    checkpoint: str | None = None  # HOST path, the one that was validated
    # The path actually handed to the trainer: a container path under docker, the host path
    # under --native. Recorded separately so Phase 0D.2 can check what the trainer was given
    # against what was validated, instead of re-deriving the translation from the command.
    checkpoint_exec_path: str | None = None
    checkpoint_sha256: str | None = None
    checkpoint_iteration: int | None = None


@dataclass(frozen=True)
class ResumeTarget:
    """A resolved, validated resume: everything the runner needs, nothing it must re-derive."""

    parent_run_dir: Path
    parent_run_id: str
    checkpoint: Path  # host path
    iteration: int
    sha256: str
    size_bytes: int

    def source_asset(self) -> SourceAsset:
        """The checkpoint is a result-changing input, so it belongs in ``source_assets`` (§9)."""
        return SourceAsset(
            path=str(self.checkpoint), sha256=self.sha256, size_bytes=self.size_bytes
        )

    def info(self, exec_path: PurePath | None = None) -> ResumeInfo:
        return ResumeInfo(
            requested=True,
            parent_run_id=self.parent_run_id,
            parent_run_dir=str(self.parent_run_dir),
            checkpoint=str(self.checkpoint),
            checkpoint_exec_path=str(exec_path) if exec_path is not None else None,
            checkpoint_sha256=self.sha256,
            checkpoint_iteration=self.iteration,
        )


# --------------------------------------------------------------- host <-> container paths
#
# The Phase 0D entry blocker lived exactly here. Keep the two namespaces in separate functions
# so no caller can pass one where the other is meant: a host path is what gets *validated*, a
# container path is what gets *executed* (§17).


def container_checkpoint(host_checkpoint: Path) -> PurePosixPath:
    """Where the mount below makes ``host_checkpoint`` visible inside the container."""
    return CONTAINER_RESUME_DIR / host_checkpoint.name


def docker_resume_mount(host_checkpoint: Path) -> list[str]:
    """``-v <parent ckpt dir>:/data/resume:ro`` — the checkpoint directory only, read-only.

    Only the checkpoint directory, so a failed child cannot write into the parent run; and
    read-only, so it cannot write into it even by accident (§18).
    """
    source = host_checkpoint.parent
    if ":" in str(source):
        # ``docker -v`` splits on colons, so such a path would silently mount something else.
        raise ContractError(
            f"{source}: a checkpoint directory whose path contains ':' cannot be expressed as a "
            "docker volume argument. Move the run directory, or use --native."
        )
    return ["-v", f"{source}:{CONTAINER_RESUME_DIR}:ro"]


# ------------------------------------------------------------------ checkpoint resolution


def checkpoint_dir(parent_run_dir: Path) -> Path:
    """First existing directory of ``CKPT_SUBDIRS``; raises naming all of them (§15)."""
    for sub in CKPT_SUBDIRS:
        cand = parent_run_dir / sub
        if cand.is_dir():
            return cand
    tried = ", ".join(str(parent_run_dir / s) for s in CKPT_SUBDIRS)
    raise ContractError(
        f"resume requested from {parent_run_dir}, but it has no checkpoint directory (looked for "
        f"{tried}). A run that never wrote a checkpoint cannot be resumed; start a fresh run "
        "instead of continuing one that does not exist."
    )


def parse_iteration(name: str) -> int | None:
    m = CKPT_NAME.match(name)
    return int(m.group("iteration")) if m else None


def resolve_resume_checkpoint(ckpt_dir: Path) -> tuple[Path, int]:
    """Latest checkpoint in ``ckpt_dir`` by *parsed iteration*, never by lexical order.

    ``sorted()`` puts ``ckpt_9.pt`` after ``ckpt_10.pt``, which would silently resume the wrong
    iteration — the kind of failure that only shows up in the loss curve weeks later.
    """
    if not ckpt_dir.is_dir():
        raise ContractError(f"{ckpt_dir}: no checkpoint directory")
    candidates = sorted(p for p in ckpt_dir.glob(CKPT_GLOB) if p.is_file())
    if not candidates:
        raise ContractError(
            f"{ckpt_dir}: no checkpoint file matching {CKPT_GLOB!r}. Resume was requested, so this "
            "is a hard failure rather than a fresh run — the two are different experiments."
        )
    parsed: dict[int, list[Path]] = {}
    unparsed = []
    for p in candidates:
        it = parse_iteration(p.name)
        if it is None:
            unparsed.append(p.name)
        else:
            parsed.setdefault(it, []).append(p)
    if unparsed:
        # Refuse rather than skip: an unrecognised name may well be the newest checkpoint, and
        # resuming from an older one while reporting success is worse than not resuming at all.
        raise ContractError(
            f"{ckpt_dir}: cannot read an iteration from {sorted(unparsed)}. Expected "
            "ckpt_<iteration>.pt or ckpt_<iteration>_rank<n>.pt (gsplat v1.5.3 naming); refusing "
            "to guess which checkpoint is the latest."
        )
    iteration = max(parsed)
    at_latest = sorted(parsed[iteration])
    if len(at_latest) > 1:
        # One .pt per rank is how a distributed run saves; resuming those together is Phase 6+
        # (docs/ROADMAP.md). Picking one rank would resume a fraction of the model.
        raise ContractError(
            f"{ckpt_dir}: iteration {iteration} has {len(at_latest)} checkpoint files "
            f"({[p.name for p in at_latest]}), which is a multi-rank (distributed) checkpoint. "
            "Resuming distributed runs is not implemented; resume a single-GPU run."
        )
    return at_latest[0], iteration


def validate_checkpoint(path: Path, parent_run_dir: Path) -> int:
    """Readable regular file that stays inside the parent run; returns its size (§15)."""
    real = path.resolve()
    if not real.is_file():
        raise ContractError(f"{path}: checkpoint is not a regular file")
    root = parent_run_dir.resolve()
    if not real.is_relative_to(root):
        raise ContractError(
            f"{path}: checkpoint resolves to {real}, outside the parent run {root}. A checkpoint "
            "reached through a link out of the run directory has no recorded provenance."
        )
    try:
        with open(real, "rb") as f:
            f.read(1)
    except OSError as e:
        raise ContractError(f"{path}: checkpoint is unreadable ({e})") from e
    return real.stat().st_size


def load_parent_record(resume_from: str | Path) -> tuple[Path, Any]:
    """``(parent_run_dir, RunRecord)`` for ``--resume-from``; both absences are hard failures."""
    from minegs.train.runner.base import load_record

    # Absolute from here on: a docker -v source must be absolute, and a relative parent in
    # run.json would only be resolvable from whatever directory the run was submitted in.
    parent_run_dir = Path(resume_from).resolve()
    if not parent_run_dir.is_dir():
        raise ContractError(
            f"--resume-from {resume_from}: no such run directory. Point it at an existing "
            "runs/<run_id> directory."
        )
    try:
        record = load_record(parent_run_dir)
    except ContractError as e:
        raise ContractError(f"--resume-from {resume_from}: {e}") from e
    return parent_run_dir, record


# ---------------------------------------------------------------------- compatibility


def _profile_keys(profile_dump: dict[str, Any]) -> dict[str, Any]:
    """Training-critical profile settings a resume may not change (§24).

    ``max_steps`` is deliberately absent: extending a run is the normal reason to resume, and
    it is checked separately because only one direction is allowed.
    """
    args = profile_dump.get("backend_args") or {}
    return {
        "backend_args.strategy": args.get("strategy", "default"),
        "backend_args.normalize_world_space": bool(args.get("normalize_world_space", False)),
        "backend_args.init_type": args.get("init_type"),
        "data_factor": profile_dump.get("data_factor"),
        "max_images": profile_dump.get("max_images"),
        "requests": profile_dump.get("requests") or {},
    }


def check_compatibility(
    parent: Any,
    *,
    backend_name: str,
    dataset_hash: str,
    chunk_id: str | None,
    profile_dump: dict[str, Any],
) -> None:
    """Refuse a resume whose child would not continue the same experiment (§23, §24).

    Every mismatch is named, with both values, because "incompatible" alone sends the user
    back to diffing two run.json files by hand.
    """
    problems: list[str] = []
    parent_backend = (parent.backend or {}).get("name")
    if parent_backend != backend_name:
        problems.append(f"backend: parent {parent_backend!r} != requested {backend_name!r}")
    if parent.dataset_hash != dataset_hash:
        problems.append(f"dataset_hash: parent {parent.dataset_hash} != current {dataset_hash}")
    if parent.chunk_id != chunk_id:
        problems.append(f"chunk_id: parent {parent.chunk_id!r} != requested {chunk_id!r}")
    if parent.frame_of_outputs != "LOCAL_METRIC":
        problems.append(
            f"frame_of_outputs: parent {parent.frame_of_outputs!r} != 'LOCAL_METRIC' "
            "(the baseline output frame contract, §3)"
        )
    parent_keys, child_keys = _profile_keys(parent.profile or {}), _profile_keys(profile_dump)
    for key in sorted(set(parent_keys) | set(child_keys)):
        if parent_keys.get(key) != child_keys.get(key):
            problems.append(
                f"profile.{key}: parent {parent_keys.get(key)!r} != {child_keys.get(key)!r}"
            )
    parent_steps = (parent.profile or {}).get("max_steps")
    child_steps = profile_dump.get("max_steps")
    if (
        isinstance(parent_steps, int)
        and isinstance(child_steps, int)
        and child_steps < parent_steps
    ):
        problems.append(
            f"profile.max_steps: {child_steps} < parent {parent_steps}; resuming into a shorter "
            "schedule has no defined meaning (raise max_steps, or start a fresh run)"
        )
    if problems:
        raise ContractError(
            "resume refused — the child run would not continue the parent's experiment: "
            + "; ".join(problems)
        )


def check_staged_compatibility(parent: Any, staged: dict[str, Any]) -> None:
    """The trainer's actual input must be byte-identical to the parent's (§25).

    A matching ``dataset_hash`` is not enough: ``max_images``, the chunk and the staging policy
    all change what the trainer sees while the dataset on disk stays the same.
    """
    problems = []
    for key in ("sha256", "init_source"):
        parent_value = (parent.staged or {}).get(key)
        child_value = staged.get(key)
        if parent_value != child_value:
            problems.append(f"staged.{key}: parent {parent_value!r} != current {child_value!r}")
    if problems:
        raise ContractError(
            "resume refused — the staged dataset the trainer would see differs from the parent's: "
            + "; ".join(problems)
        )


# ------------------------------------------------------------------------------- guards


def guard_resume_argv(argv: list[str], checkpoint: PurePath | None) -> None:
    """Last line of defence for §16: resume requested ⇒ the executed argv carries that checkpoint.

    Placed after command assembly and before ``Popen`` so no route — a backend that forgets to
    emit the flag, a future refactor, a passthrough that drops it — can turn a requested resume
    into a silent fresh run.
    """
    if checkpoint is None:
        return
    want = str(checkpoint)
    if "--ckpt" not in argv or argv[argv.index("--ckpt") + 1 :][:1] != [want]:
        raise ContractError(
            f"resume was requested with checkpoint {want}, but the assembled command does not pass "
            f"it (--ckpt missing or pointing elsewhere): {argv}. Refusing to start what would be a "
            "fresh run under a resumed run's id."
        )


# ------------------------------------------------------------------------- orchestration


def resolve_resume(
    resume_from: str | Path,
    *,
    backend: Any,
    dataset_hash: str,
    chunk_id: str | None,
    profile_dump: dict[str, Any],
) -> ResumeTarget:
    """Everything a resume must satisfy *before* the child run leaves a trace on disk (§28).

    Order matters: the backend's own ability to resume is checked first, so a backend that
    cannot continue training says so instead of the user discovering it from a compatibility
    message about a run that could never have been resumed anyway.
    """
    caps = backend.capabilities()
    if not caps.has("resume"):
        note = backend.capability_notes.get("resume", "")
        raise ContractError(
            " ".join(
                [
                    f"backend {backend.name} does not support resuming training, so "
                    f"--resume-from cannot be honoured.",
                    note,
                ]
            ).strip()
        )
    parent_run_dir, parent = load_parent_record(resume_from)
    check_compatibility(
        parent,
        backend_name=backend.name,
        dataset_hash=dataset_hash,
        chunk_id=chunk_id,
        profile_dump=profile_dump,
    )
    ckpt, iteration = resolve_resume_checkpoint(checkpoint_dir(parent_run_dir))
    size = validate_checkpoint(ckpt, parent_run_dir)
    # Hashed once here; both run.json's resume block and provenance.source_assets reuse it,
    # because a checkpoint runs to gigabytes (§26).
    return ResumeTarget(
        parent_run_dir=parent_run_dir,
        parent_run_id=parent.run_id,
        checkpoint=ckpt,
        iteration=iteration,
        sha256=sha256_file(ckpt),
        size_bytes=size,
    )
