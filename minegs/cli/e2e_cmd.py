"""``minegs e2e`` — one entry point for the whole chain (§Phase 2 §4).

Three commands, and the split between the first and the third is the point of the module:

* ``run`` executes stages. It is the expensive one — an E57 extraction, a training run.
* ``status`` answers "where is this, and what would be refused if I ran it now" without
  touching anything.
* ``report`` rebuilds the document from artifacts that already exist. Rewriting a heading must
  never cost a training run, so this command runs no stage at all.

What this module deliberately does *not* offer is a way to substitute the trainer or the
renderer. Those seams exist so the structural gate can run on a machine with no GPU, and they
are reached from Python by a test that then records the substitution. A ``--fake-renderer``
flag would be a way to mint a Phase 2 report from the command line without a GPU ever running,
which is precisely the claim this project spends its contracts refusing.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from minegs.cli._common import console, dump_json, run_guarded

app = typer.Typer(no_args_is_help=True)

_STAGES = "ingest, dataset, train, depth, surface, geometry, sections_volume, report"


@app.command("run")
def run(
    config: Path = typer.Argument(
        ..., help="workflow config (YAML/JSON); see docs/PHASE2_E57_G2.md"
    ),
    work_dir: Path = typer.Option(
        ..., "--work-dir", "-w", help="where the ledger and this workflow's artifacts live"
    ),
    through: str = typer.Option("report", "--through", help=f"stop after this stage ({_STAGES})"),
    rebuild_from: str | None = typer.Option(
        None,
        "--rebuild-from",
        help="clear this stage and every stage after it, then run them again. The explicit "
        "answer to a stale stage.",
    ),
) -> None:
    """Carry one survey through ingest → dataset → train → depth → surface → evaluation → report.

    Stages that already succeeded against exactly these inputs are reused. A stage whose inputs
    have moved is *not* reused and not silently re-run either: the workflow says what changed
    and stops, because its recorded outputs describe a survey that is no longer on disk.

    This runs the real thing. Training needs a GPU and a working backend; there is no flag here
    that substitutes one.
    """
    from minegs.e2e.runner import Workflow, load_e2e_config, stage_from_name
    from minegs.e2e.stages import default_specs

    def go() -> None:
        cfg = load_e2e_config(config)
        # Always passed, never conditionally: reopening a workflow with changed settings is a
        # different workflow, and the constructor is what says so.
        wf = Workflow(work_dir, cfg)
        stop = stage_from_name(through)
        start = stage_from_name(rebuild_from) if rebuild_from else None
        console.print(f"workflow [bold]{wf.state.workflow_id}[/] in {work_dir}")

        def say(event: str, rec) -> None:
            if event == "start":
                console.print(f"  [dim]{rec.stage.value}[/] …")
            else:
                mark = "reused" if rec.status.value == "reused" else rec.status.value
                secs = "" if rec.elapsed_seconds is None else f" {rec.elapsed_seconds:.1f}s"
                console.print(f"  {rec.stage.value}: [bold]{mark}[/]{secs}")

        state = wf.execute(default_specs(), through=stop, rebuild_from=start, on_stage=say)
        _summarise(state)

    run_guarded(go)


@app.command("status")
def status(
    work_dir: Path = typer.Option(..., "--work-dir", "-w", help="an existing workflow directory"),
    out: Path | None = typer.Option(
        None, "--json", help="write the ledger to this file as JSON instead of printing a table"
    ),
) -> None:
    """What has run, what it cost, and what would be refused if the workflow ran now.

    Read-only. The stale column is the same question ``run`` asks before reusing a stage, asked
    without doing anything about it — which is what an operator wants before deciding what to
    rebuild.
    """
    from minegs.e2e.models import STAGE_ORDER, WORKFLOW_STATE_FILE, WorkflowState
    from minegs.e2e.runner import Workflow
    from minegs.e2e.stages import default_specs

    def go() -> None:
        from minegs.core.errors import ContractError

        path = work_dir / WORKFLOW_STATE_FILE
        if not path.is_file():
            raise ContractError(f"{path}: no workflow there")
        if out is not None:
            dump_json(WorkflowState.load(path).model_dump(mode="json"), out)
            return
        wf = Workflow(work_dir)
        stale = set(wf.stale_stages(default_specs()))
        table = Table(title=f"workflow {wf.state.workflow_id}")
        for column in ("stage", "status", "seconds", "fingerprint", "stale"):
            table.add_column(column)
        for stage in STAGE_ORDER:
            rec = wf.state.stages[stage]
            table.add_row(
                stage.value,
                rec.status.value,
                "—" if rec.elapsed_seconds is None else f"{rec.elapsed_seconds:.1f}",
                (rec.input_fingerprint or "—")[:12],
                "yes" if stage in stale else "",
            )
        console.print(table)
        done = wf.state.done_through()
        console.print(f"complete through: [bold]{done[-1].value if done else 'nothing'}[/]")
        for stage in STAGE_ORDER:
            rec = wf.state.stages[stage]
            if rec.failure_reason:
                console.print(f"[red]{stage.value}:[/] {rec.failure_reason}")
        if stale:
            console.print(
                "[yellow]stale:[/] "
                + ", ".join(s.value for s in STAGE_ORDER if s in stale)
                + " — their inputs moved since they ran; rebuild explicitly "
                f"(`--rebuild-from {min(stale, key=STAGE_ORDER.index).value}`)"
            )

    run_guarded(go)


@app.command("report")
def report(
    work_dir: Path = typer.Option(..., "--work-dir", "-w", help="an existing workflow directory"),
    out: Path | None = typer.Option(
        None, "--out", help="where to write the report (default: <work_dir>/report)"
    ),
) -> None:
    """Rebuild ``phase2_report.json`` and ``phase2_report.md`` from existing artifacts.

    Runs no stage. A workflow that stopped part-way is reported rather than refused — with the
    stages that did not complete named in the report, because a partial report an operator can
    read beats a refusal they cannot.
    """
    from minegs.e2e.models import Phase2Report
    from minegs.e2e.report import report_from_workflow

    def go() -> None:
        path = report_from_workflow(work_dir, out)
        rep = Phase2Report.load(path)
        console.print(f"report [bold]{rep.report_id}[/]")
        console.print(f"  {path}")
        console.print(f"  {path.with_name('phase2_report.md')}")
        _maturity(rep)

    run_guarded(go)


def _summarise(state) -> None:
    from minegs.e2e.models import Phase2Report, Stage

    rec = state.stages[Stage.REPORT]
    if not rec.usable or not rec.outputs.get("report_json_path"):
        console.print("[yellow]no report written yet[/] (the run stopped before that stage)")
        return
    console.print(f"report: {rec.outputs['report_json_path']}")
    console.print(f"        {rec.outputs['report_md_path']}")
    _maturity(Phase2Report.load(rec.outputs["report_json_path"]))


def _maturity(rep) -> None:
    """The three statements a reader needs before believing any number above them."""
    console.print(f"  structural: [bold]{rep.maturity.structural_status}[/]")
    console.print(f"  real-data validation: [bold]{rep.maturity.real_data_validation_status}[/]")
    console.print(f"  human visual review: [bold]{rep.maturity.human_visual_review_status}[/]")
    console.print(f"  real GPU training executed: {rep.training.real_gpu_execution}")
    console.print(f"  real depth renderer executed: {rep.reconstruction.real_renderer_execution}")
    console.print(f"[yellow]{rep.maturity.statement}[/]")
