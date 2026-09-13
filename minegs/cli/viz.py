from __future__ import annotations

from pathlib import Path

import typer

from minegs.cli._common import console, run_guarded

app = typer.Typer(no_args_is_help=True)


@app.command()
def view(
    dataset_dir: Path = typer.Argument(...),
    run_ply: Path | None = typer.Option(None),
    tls: Path | None = typer.Option(None, help="TLS PLY (LOCAL_METRIC or TLS_GLOBAL)"),
    golden_gate: Path | None = typer.Option(
        None, "--golden-gate", help="a golden-gate report dir; uses its LOCAL_METRIC TLS sample"
    ),
    port: int = typer.Option(8080),
) -> None:
    """Viser viewer: frustums, init points, splats, TLS, centerline scrubber (§12, §26)."""
    from minegs.core.errors import ContractError
    from minegs.viz.viewer import launch

    def go() -> None:
        tls_path = tls
        if golden_gate is not None:
            from minegs.dataset.golden_gate import TLS_SAMPLE_FILE

            tls_path = golden_gate / TLS_SAMPLE_FILE
            if not tls_path.is_file():
                raise ContractError(
                    f"{golden_gate}: no {TLS_SAMPLE_FILE}; run `minegs dataset golden-gate` first"
                )
        launch(dataset_dir, run_ply, tls_path, port)

    run_guarded(go)


@app.command()
def overlay(
    pano: Path = typer.Argument(...),
    scan_ply: Path = typer.Argument(...),
    out: Path = typer.Argument(...),
    az_sign: int = typer.Option(1),
    el_flip: bool = typer.Option(False),
    az_offset: float = typer.Option(0.0),
) -> None:
    """Reproject scanner-frame points onto a panorama with a given convention (golden gate)."""
    import numpy as np
    from PIL import Image

    from minegs.core.pointcloud import read_ply
    from minegs.ingest.common.geometry import PanoConvention
    from minegs.viz.overlay import render_overlay, save_overlay

    def go() -> None:
        img = np.asarray(Image.open(pano).convert("RGB"))
        p = save_overlay(
            out,
            render_overlay(
                img, read_ply(scan_ply).xyz, PanoConvention(az_sign, el_flip, az_offset)
            ),
        )
        console.print(f"wrote {p}")

    run_guarded(go)


@app.command()
def export(
    ply: Path = typer.Argument(...),
    out_dir: Path = typer.Argument(...),
    fmt: str = typer.Option("spz", help="spz | splat"),
) -> None:
    """Research .ply -> distribution .spz (or legacy .splat)."""
    from minegs.viz.export import export_run

    run_guarded(lambda: console.print(f"wrote {export_run(ply, out_dir, fmt)}"))
