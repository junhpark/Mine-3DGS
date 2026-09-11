from __future__ import annotations

import typer

import minegs
from minegs.cli import dataset, eval_cmd, ingest, sync, train, viz

app = typer.Typer(
    name="minegs",
    help="Metric Gaussian Splatting for underground tunnels — ingest / dataset / train / eval / viz / sync.",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)
app.add_typer(ingest.app, name="ingest", help="raw (E57 / video / 360) -> intermediate products")
app.add_typer(
    dataset.app,
    name="dataset",
    help="dataset contract: synthetic / validate / info / migrate / chunks",
)
app.add_typer(train.app, name="train", help="submit and inspect training runs (local / RunPod)")
app.add_typer(
    eval_cmd.app,
    name="eval",
    help="protocol / register / geometry / sections / volume / change / render",
)
app.add_typer(viz.app, name="viz", help="viewer / overlay / export")
app.add_typer(sync.app, name="sync", help="rclone push (dataset only) / pull (runs)")


def _version(value: bool) -> None:
    if value:
        typer.echo(f"minegs {minegs.__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version, is_eager=True, help="Print version and exit."
    ),
) -> None:
    """minegs command line."""


if __name__ == "__main__":  # pragma: no cover
    app()
