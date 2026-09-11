from __future__ import annotations

import numpy as np
import pytest
from minegs.core.synthetic import SyntheticResult, SyntheticSpec, generate

from e57_fakes import FakeHeader, FakeNode, install_fake_pye57


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> SyntheticResult:
    """One small synthetic tunnel shared by the session (Phase 0A gate fixture)."""
    root = tmp_path_factory.mktemp("syn")
    return generate(
        root, SyntheticSpec(length_m=90.0, station_spacing_m=15.0, image_size=48, points_per_m=1500)
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def fake_e57(tmp_path, monkeypatch):
    """Install a fake pye57 whose E57 class is built from the headers a test supplies.

    Returns an installer: ``path, opened = fake_e57(headers, root=..., name=...)``. Because
    every reader opens files through ``_nodes.open_e57``, one install covers the inventory,
    the mapper and the extractor alike.
    """

    def install(headers: list[FakeHeader], root: FakeNode | None = None, name: str = "f.e57"):
        return install_fake_pye57(monkeypatch, tmp_path, headers, root, name)

    return install
