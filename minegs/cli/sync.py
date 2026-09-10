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
    """Push dataset/ (never raw/) to the pod volume (§1.4, §8.2)."""
    from minegs.train.runner import sync

    run_guarded(lambda: console.print(" ".join(sync.push(dataset_dir, remote, dry_run=dry_run))))


@app.command()
def pull(
    remote: str = typer.Argument(...),
    local_dir: Path = typer.Argument(...),
    dry_run: bool = typer.Option(False),
) -> None:
    """Pull runs/<id>/ (ply, log, run.json) from the pod volume."""
    from minegs.train.runner import sync

    run_guarded(lambda: console.print(" ".join(sync.pull(remote, local_dir, dry_run=dry_run))))
