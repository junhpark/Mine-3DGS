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
    depth_supervision: Path | None = typer.Option(
        None, help="a verified DepthSupervisionRecord directory (profiles requesting depth_loss)"
    ),
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
        sup_dir = sup_sha = None
        if depth_supervision is not None:
            from minegs.train.supervision.depth import verify_depth_supervision

            v = verify_depth_supervision(dataset_dir, depth_supervision)
            sup_dir, sup_sha = v.path, v.artifact_sha256
        cmd = be.build_command(
            dataset_dir,
            out_dir,
            prof,
            check_trainer=False,
            depth_supervision_dir=sup_dir,
            depth_supervision_sha256=sup_sha,
        )
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
    chunk: str | None = typer.Option(None, help="a chunk id of --chunk-plan (Phase 5)"),
    chunk_plan: Path | None = typer.Option(
        None, "--chunk-plan", help="a chunks/<plan_id> directory (`minegs dataset chunk-plan`)"
    ),
    depth_supervision: Path | None = typer.Option(
        None,
        "--depth-supervision",
        help="a DepthSupervisionRecord directory (`minegs dataset depth-supervision`); required "
        "by profiles that request depth_loss (heavy, heavy-depth), refused by the others",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="runpod: print what would be uploaded and the pod request, with no upload, no pod "
        "and no cost (`train command` is the local equivalent)",
    ),
) -> None:
    """Run training to completion. Output: <dataset>/../runs/<run_id>/ with LOCAL_METRIC .ply (§8).

    Blocks until the run reaches a terminal state, and there is deliberately no flag to detach.
    A run is only verified when it finishes — outputs normalised, checkpoint and PLY checked,
    frame invariant confirmed — and nothing can pick that up afterwards: there is no reattach
    path, so a detached run would leave run.json at 'running' for ever however the trainer
    ended. Detaching needs a finalize-on-inspection path that does not exist yet; until it does,
    use your shell's job control if you need the terminal back.

    --resume-from names a parent run explicitly, and always fails closed today: neither the
    runner nor any shipped backend implements resuming, and a restart from iteration 0 is a
    different experiment, not a slower resume (docs/ROADMAP.md §Phase 0D).
    """
    from minegs.core.errors import ContractError
    from minegs.train.backends import get_backend
    from minegs.train.profiles import load_profile
    from minegs.train.runner import RunConfig, get_runner
    from minegs.train.runner.base import RunnerConfig, RunStatus, load_record

    def go() -> None:
        prof = load_profile(profile)
        # Check the (profile, backend) capability contract BEFORE dispatching to a runner, so a
        # profile that cannot run says why (e.g. heavy -> depth_loss) instead of surfacing
        # whatever its default runner happens to complain about first.
        get_backend(backend or prof.backend).resolve_requests(prof)
        rname = runner or prof.default_runner
        rcfg = RunnerConfig.load(config) if config else RunnerConfig(runner=rname, native=native)
        if rcfg.runner != rname:
            raise ContractError(
                f"--config names runner {rcfg.runner!r}, but this run would go to {rname!r}; "
                f"pass --runner {rcfg.runner} to use that config, or a config for {rname}"
            )
        if native:
            rcfg.native = True
        r = get_runner(rname, rcfg)
        cfg = RunConfig(
            dataset_dir=str(dataset_dir),
            profile=profile,
            backend=backend,
            runner=rname,
            resume_from=str(resume_from) if resume_from else None,
            chunk_id=chunk,
            chunk_plan=str(chunk_plan) if chunk_plan else None,
            depth_supervision=str(depth_supervision) if depth_supervision else None,
        )
        if dry_run:
            if rname != "runpod":
                raise ContractError(
                    "--dry-run plans a RunPod run; for a local run see `train command`"
                )
            dump_json(r.plan(cfg), None)
            return
        h = r.submit(cfg)
        console.print(f"submitted [bold]{h.run_id}[/] -> {h.run_dir}")
        # a remote run is asked about at the config's pace, not every two seconds
        st = h.wait(poll_s=float(rcfg.poll_interval_s) if rname == "runpod" else 2.0)
        console.print(f"status: {st.value}  artifacts: {[str(p) for p in h.fetch_artifacts()]}")
        if st is not RunStatus.SUCCEEDED:
            rec = load_record(h.run_dir)
            raise ContractError(rec.failure_reason or f"run finished {st.value}")

    run_guarded(go)


@app.command("chunks")
def chunks(
    dataset_dir: Path = typer.Argument(...),
    chunk_plan: Path = typer.Option(..., "--chunk-plan", help="a chunks/<plan_id> directory"),
    profile: str = typer.Option("light"),
    runner: str | None = typer.Option(
        None, help="local | runpod (default: profile.default_runner)"
    ),
    backend: str = typer.Option("gsplat"),
    config: Path | None = typer.Option(None, help="configs/runner/*.yaml"),
    native: bool = typer.Option(False, help="local: run in this python env instead of docker"),
    runs_dir: Path | None = typer.Option(None, help="default <dataset>/../runs/<plan_id>"),
    depth_supervision: Path | None = typer.Option(None, "--depth-supervision"),
) -> None:
    """Train every chunk of a plan in order with one profile; stop at the first failure.

    Each chunk is an ordinary `train run` with the plan and its chunk id: the same staging,
    trainer, evidence and refusals. No scheduler, parallelism or retry (Phase 5 AD-5).
    """
    from minegs.chunks.run import train_chunks
    from minegs.core.errors import ContractError
    from minegs.train.backends import get_backend
    from minegs.train.profiles import load_profile
    from minegs.train.runner import get_runner
    from minegs.train.runner.base import RunnerConfig

    def go() -> None:
        prof = load_profile(profile)
        get_backend(backend or prof.backend).resolve_requests(prof)
        rname = runner or prof.default_runner
        rcfg = RunnerConfig.load(config) if config else RunnerConfig(runner=rname, native=native)
        if rcfg.runner != rname:
            raise ContractError(
                f"--config names runner {rcfg.runner!r}, but these runs would go to {rname!r}"
            )
        if native:
            rcfg.native = True
        done = train_chunks(
            dataset_dir,
            chunk_plan,
            profile,
            get_runner(rname, rcfg),
            runs_dir=runs_dir,
            backend=backend,
            depth_supervision=depth_supervision,
            poll_s=float(rcfg.poll_interval_s) if rname == "runpod" else 2.0,
        )
        for d in done:
            console.print(f"  {d['chunk_id']}: {d['status']}  {d['run_dir']}")
        failed = [d for d in done if d["status"] != "succeeded"]
        if failed:
            raise ContractError(
                f"chunk {failed[0]['chunk_id']} did not succeed ({failed[0]['failure']}); the "
                "remaining chunks were not started and the chunk set is not complete"
            )

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
def fetch(
    run_dir: Path = typer.Argument(...),
    config: Path | None = typer.Option(None, help="the configs/runner/runpod.yaml it ran with"),
) -> None:
    """Check a RunPod run and, once it has finished, pull and verify its outputs into run_dir.

    Success is the worker's durable status on the volume plus every output file matching the
    run's output manifest — never the pod's lifecycle (docs/PHASE6_CONTRACT.md §6, §9).
    """
    from minegs.core.errors import ContractError
    from minegs.train.runner.base import RunnerConfig, load_record

    def go() -> None:
        rec = load_record(run_dir)
        if rec.runner == "local":
            console.print(f"local run; outputs: {rec.outputs}")
            return
        if config is None:
            raise ContractError(
                "a RunPod run is fetched with the runner config it ran with (--config)"
            )
        from minegs.train.runner.runpod import RunPodRunner

        h = RunPodRunner(RunnerConfig.load(config)).attach(run_dir)
        st = h.status()
        console.print(f"{h.run_id}: {st.value}  artifacts: {[str(p) for p in h.fetch_artifacts()]}")

    run_guarded(go)


@app.command()
def cancel(
    run_dir: Path = typer.Argument(...),
    config: Path = typer.Option(..., help="the configs/runner/runpod.yaml it ran with"),
) -> None:
    """Terminate a RunPod run's pod. A run that already finished keeps its result."""
    from minegs.train.runner.base import RunnerConfig
    from minegs.train.runner.runpod import RunPodRunner

    def go() -> None:
        h = RunPodRunner(RunnerConfig.load(config)).attach(run_dir)
        h.cancel()
        console.print(f"{h.run_id}: {h.status().value}")

    run_guarded(go)


@app.command("remote-worker", hidden=True)
def remote_worker(inputs: Path = typer.Argument(..., help="jobs/<run_id>/inputs.json")) -> None:
    """Pod side of a RunPod run: verify the inputs, train, write status and outputs (Phase 6)."""
    from minegs.train.remote.worker import run_worker

    raise typer.Exit(run_worker(inputs))
