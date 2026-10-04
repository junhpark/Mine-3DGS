"""Train every chunk of a plan, one after another, through the ordinary run path (Phase 5 AD-5).

Deliberately thin: for each chunk in plan order, the same ``Runner.submit`` / ``wait`` that
``minegs train run`` uses, with the plan and the chunk id. No scheduler, queue, parallelism or
retry. The first chunk that does not succeed stops the loop, and nothing downstream can call the
set complete (``ChunkRunSet`` refuses a missing or failed chunk unless told to report an
incomplete set, which then says so).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from minegs.chunks.plan import verify_chunk_plan


def train_chunks(
    dataset_dir: str | Path,
    chunk_plan: str | Path,
    profile: str,
    runner,
    *,
    runs_dir: str | Path | None = None,
    backend: str = "gsplat",
    depth_supervision: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    poll_s: float = 2.0,
) -> list[dict[str, Any]]:
    """Run each chunk in order; stop at the first run that does not succeed.

    Returns one entry per chunk attempted: ``{chunk_id, run_id, run_dir, status, failure}``.
    """
    from minegs.train.runner import RunConfig
    from minegs.train.runner.base import RunStatus, load_record

    ds = Path(dataset_dir)
    plan = verify_chunk_plan(ds, chunk_plan)
    root = Path(runs_dir) if runs_dir is not None else ds.parent / "runs" / plan.plan_id
    done: list[dict[str, Any]] = []
    for chunk in plan.chunks:
        h = runner.submit(
            RunConfig(
                dataset_dir=str(ds),
                profile=profile,
                backend=backend,
                runner=runner.name,
                run_dir=str(root / chunk.chunk_id),
                chunk_id=chunk.chunk_id,
                chunk_plan=str(chunk_plan),
                depth_supervision=None if depth_supervision is None else str(depth_supervision),
                overrides=dict(overrides or {}),
            )
        )
        status = h.wait(poll_s=poll_s)
        rec = load_record(h.run_dir)
        done.append(
            {
                "chunk_id": chunk.chunk_id,
                "run_id": h.run_id,
                "run_dir": str(h.run_dir),
                "status": status.value,
                "failure": rec.failure_reason,
            }
        )
        if status is not RunStatus.SUCCEEDED:
            break
    return done


__all__ = ["train_chunks"]
