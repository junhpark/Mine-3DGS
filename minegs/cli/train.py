from __future__ import annotations

import shlex
from pathlib import Path

import typer

from minegs.cli._common import console, dump_json, run_guarded

app = typer.Typer(no_args_is_help=True)


@app.command()
def profiles() -> None:
    """List builtin profiles and their capability requests."""
    from minegs.train.profiles import BUILTIN, load_profile

    for n in BUILTIN:
        p = load_profile(n)
        console.print(
            f"[bold]{p.name}[/]  backend={p.backend} runner={p.default_runner} images={p.max_images} factor={p.data_factor} steps={p.max_steps}"
        )
        console.print(
            f"   requires={p.required_capabilities()} optional={p.optional_capabilities()}"
        )


@app.command()
def command(
    dataset_dir: Path = typer.Argument(...),
    profile: str = typer.Option("light"),
    backend: str = typer.Option("gsplat"),
    out_dir: Path = typer.Option(Path("/data/run/backend_out")),
) -> None:
    """Print the backend command a run would execute (dry run, no GPU / trainer needed).

    The real run points the trainer at a *staged* copy (runs/<id>/staged) with the profile's
    image subset and init_points.ply as points3D — see minegs.train.staging.
    """
    from minegs.train.backends import get_backend
    from minegs.train.profiles import load_profile

    def go() -> None:
        be = get_backend(backend)
        prof = load_profile(profile)
        cmd = be.build_command(dataset_dir, out_dir, prof, check_trainer=False)
        # shlex.join, not " ".join: a value containing whitespace would otherwise print as two
        # arguments, so the line a reader copies would not be the command that runs.
        #
        # ...and then printed verbatim, which rich does not do by default. Wrapped to the console
        # width it emits real newlines, so pasting the output runs the first line as a command of
        # its own — a *runnable* training command that has quietly lost --no-normalize_world_space
        # and --max_steps. A long enough dataset path is folded mid-token. And a value containing
        # brackets is parsed as rich markup and partly deleted, so the printed --tag is not the
        # --tag that runs. soft_wrap keeps it one line, markup=False keeps the value, and
        # highlight=False keeps rich from colouring what is meant to be copied.
        console.print(shlex.join(cmd.argv), soft_wrap=True, markup=False, highlight=False)
        console.print(
            f"[dim]data_dir above is the staged dataset at run time; "
            f"max_images={prof.max_images} applied by staging[/]"
        )
        console.print(
            f"T_local_from_internal identity={cmd.T_local_from_internal.is_identity()}  "
            f"capabilities={be.resolve_requests(prof)}"
        )

    run_guarded(go)


@app.command()
def run(
    dataset_dir: Path = typer.Argument(...),
    profile: str = typer.Option("light"),
    runner: str | None = typer.Option(
        None, help="local | runpod (default: profile.default_runner)"
    ),
    backend: str = typer.Option("gsplat"),
    config: Path | None = typer.Option(None, help="configs/runner/*.yaml"),
    native: bool = typer.Option(False, help="local: run in this python env instead of docker"),
    resume_from: Path | None = typer.Option(
        None,
        "--resume-from",
        help="continue an existing run: path to runs/<run_id>. NOT IMPLEMENTED — neither the "
        "runner (no checkpoint discovery, no host/container path translation) nor any shipped "
        "backend (gsplat v1.5.3 cannot continue training) can honour it, so it always fails "
        "closed rather than silently restarting from iteration 0 (Phase 0D.3, docs/ROADMAP.md).",
    ),
    chunk: str | None = typer.Option(None),
    wait: bool = typer.Option(False),
) -> None:
    """Submit a training run. Output: <dataset>/../runs/<run_id>/ with LOCAL_METRIC .ply (§8).

    --resume-from names a parent run explicitly, and always fails closed today: neither the
    runner nor any shipped backend implements resuming, and a restart from iteration 0 is a
    different experiment, not a slower resume (docs/ROADMAP.md §Phase 0D).
    """
    from minegs.train.backends import get_backend
    from minegs.train.profiles import load_profile
    from minegs.train.runner import RunConfig, get_runner
    from minegs.train.runner.base import RunnerConfig

    def go() -> None:
        prof = load_profile(profile)
        # Check the (profile, backend) capability contract BEFORE dispatching to a runner, so a
        # profile that cannot run says why (e.g. heavy -> depth_loss) instead of surfacing
        # whatever its default runner happens to complain about first.
        get_backend(backend or prof.backend).resolve_requests(prof)
        rname = runner or prof.default_runner
        rcfg = RunnerConfig.load(config) if config else RunnerConfig(runner=rname, native=native)
        if native:
            rcfg.native = True
        r = get_runner(rname, rcfg)
        h = r.submit(
            RunConfig(
                dataset_dir=str(dataset_dir),
                profile=profile,
                backend=backend,
                runner=rname,
                resume_from=str(resume_from) if resume_from else None,
                chunk_id=chunk,
            )
        )
        console.print(f"submitted [bold]{h.run_id}[/] -> {h.run_dir}")
        if wait:
            st = h.wait()
            console.print(f"status: {st.value}  artifacts: {[str(p) for p in h.fetch_artifacts()]}")

    run_guarded(go)


@app.command()
def status(run_dir: Path = typer.Argument(...)) -> None:
    from minegs.train.runner.base import load_record

    run_guarded(lambda: dump_json(load_record(run_dir), None))


@app.command()
def logs(run_dir: Path = typer.Argument(...), tail: int = typer.Option(50)) -> None:
    p = run_dir / "log" / "train.log"
    if not p.exists():
        console.print("[yellow]no log yet[/]")
        return
    lines = p.read_text(errors="replace").splitlines()
    console.print("\n".join(lines[-tail:]))


@app.command()
def fetch(run_dir: Path = typer.Argument(...), config: Path | None = typer.Option(None)) -> None:
    """Pull artifacts of a RunPod run into run_dir (local runs are already in place)."""
    from minegs.train.runner.base import load_record

    def go() -> None:
        rec = load_record(run_dir)
        if rec.runner == "local":
            console.print(f"local run; outputs: {rec.outputs}")
            return
        from minegs.core.errors import NotYetImplementedError

        raise NotYetImplementedError("fetching RunPod run artifacts", "6")

    run_guarded(go)
