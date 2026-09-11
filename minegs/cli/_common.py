from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from minegs.core.errors import MinegsError

console = Console()
err_console = Console(stderr=True)

EXIT_CONTRACT = 2
EXIT_PROTOCOL = 3
EXIT_MISSING_DEP = 4


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
        err_console.print(f"[bold red]protocol violation:[/] {e}")
        raise typer.Exit(EXIT_PROTOCOL) from None
    except MissingDependencyError as e:
        err_console.print(f"[bold yellow]missing dependency:[/] {e}")
        raise typer.Exit(EXIT_MISSING_DEP) from None
    except NotYetImplementedError as e:
        err_console.print(f"[bold yellow]not yet:[/] {e}")
        raise typer.Exit(EXIT_MISSING_DEP) from None
    except (ContractError, MinegsError) as e:
        err_console.print(f"[bold red]error:[/] {e}")
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
