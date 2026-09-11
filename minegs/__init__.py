"""minegs — metric Gaussian Splatting research framework for underground tunnels.

See docs/ARCHITECTURE.md. Principles (§1): one dataset contract, one codebase with
purpose-specific runtimes, CLI as source of truth, raw data stays local, three
coordinate frames, leak-proof evaluation protocols, surfaces not centers, provenance
on every artifact.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
