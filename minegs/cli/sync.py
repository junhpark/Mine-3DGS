from __future__ import annotations

from pathlib import Path

import typer

from minegs.cli._common import console, run_guarded

app = typer.Typer(no_args_is_help=True)


@app.command()
def push(
    dataset_dir: Path = typer.Argument(...),
    remote: str = typer.Argument(..., help="rclone remote:path"),
    dry_run: bool = typer.Option(False),
) -> None:
    """Push exactly the files the dataset hash covers (never raw/) to the pod volume.

    RunPod runs do this themselves, to a content-addressed path, and the pod re-hashes what
    arrived (docs/PHASE6_CONTRACT.md §5). This command is the bare transport.
    """
    from minegs.train.runner import sync

    run_guarded(lambda: console.print(" ".join(sync.push(dataset_dir, remote, dry_run=dry_run))))


@app.command()
def pull(
    remote: str = typer.Argument(...),
    local_dir: Path = typer.Argument(...),
    dry_run: bool = typer.Option(False),
) -> None:
    """Pull a run directory (all but backend_out/) from the pod volume — transport only.

    A RunPod run's outputs are fetched through `minegs train fetch`, which checks every file
    against the run's output manifest before publishing it (docs/PHASE6_CONTRACT.md §9).
    """
    from minegs.train.runner import sync

    run_guarded(lambda: console.print(" ".join(sync.pull(remote, local_dir, dry_run=dry_run))))
