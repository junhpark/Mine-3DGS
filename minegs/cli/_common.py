from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape

from minegs.core.errors import MinegsError

console = Console()
err_console = Console(stderr=True)

EXIT_CONTRACT = 2
EXIT_PROTOCOL = 3
EXIT_MISSING_DEP = 4


def _report(label: str, e: BaseException) -> None:
    """Print ``label`` as markup and the message verbatim.

    Interpolating the message into the markup string let rich parse it too, and an error whose
    remedy is ``pip install 'minegs[e57]'`` reads ``[e57]`` as a style tag and deletes it — so
    the one line telling the user how to fix their install told them to run a command that does
    nothing. Every remedy this CLI prints names an extra in brackets.
    """
    err_console.print(label, escape(str(e)))


def run_guarded(fn, *args, **kwargs):
    """Map MinegsError subclasses to exit codes; everything else propagates with a traceback."""
    from minegs.core.errors import (
        ContractError,
        MissingDependencyError,
        NotYetImplementedError,
        ProtocolViolation,
    )

    try:
        return fn(*args, **kwargs)
    except ProtocolViolation as e:
        _report("[bold red]protocol violation:[/]", e)
        raise typer.Exit(EXIT_PROTOCOL) from None
    except MissingDependencyError as e:
        _report("[bold yellow]missing dependency:[/]", e)
        raise typer.Exit(EXIT_MISSING_DEP) from None
    except NotYetImplementedError as e:
        _report("[bold yellow]not yet:[/]", e)
        raise typer.Exit(EXIT_MISSING_DEP) from None
    except (ContractError, MinegsError) as e:
        _report("[bold red]error:[/]", e)
        raise typer.Exit(EXIT_CONTRACT) from None


def dump_json(data: Any, out: Path | None) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False, default=_default)
    if out is None:
        sys.stdout.write(text + "\n")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        console.print(f"wrote {out}")


def _default(o: Any) -> Any:
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
