from __future__ import annotations

import numpy as np
import pytest
from minegs.core.synthetic import SyntheticResult, SyntheticSpec, generate


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
