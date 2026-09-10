"""GLUEMAP — low-texture / low-overlap specialist. Heavy dependency stack; deferred (§14)."""

from __future__ import annotations

from pathlib import Path

from minegs.core.errors import NotYetImplementedError
from minegs.ingest.video.sfm.base import SfMBackend, SfMOptions


class GLUEMAP(SfMBackend):
    name = "gluemap"

    def commands(self, images_dir: Path, work_dir: Path, opts: SfMOptions) -> list[list[str]]:
        raise NotYetImplementedError("GLUEMAP experimental SfM backend", "2+")
