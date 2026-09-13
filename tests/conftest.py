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


# ---------------------------------------------------------------- Phase 0C fixtures


@pytest.fixture(scope="session")
def staging_small(tmp_path_factory: pytest.TempPathFactory):
    """A small synthetic Phase 0B.3 staging tree (pinhole cube faces, Matterport-like axes)."""
    from minegs.core.synthetic_staging import StagingSpec, generate_staging

    root = tmp_path_factory.mktemp("stg")
    return generate_staging(
        root,
        StagingSpec(length_m=60.0, station_spacing_m=12.0, points_per_m=3000, image_size=64),
    )


@pytest.fixture(scope="session")
def calibration_small(staging_small):
    from minegs.dataset.calibrate import calibrate_camera_convention
    from minegs.dataset.staging_input import load_staging

    cal = calibrate_camera_convention(load_staging(staging_small.staging_dir))
    path = staging_small.root / "camera_convention.json"
    cal.save(path)
    return cal, path


@pytest.fixture(scope="session")
def build_config_small(staging_small, calibration_small):
    """Identity SOURCE→TLS, design centerline, every-3rd test station, one holdout range."""
    from minegs.dataset.build_config import DatasetBuildConfig

    _, conv_path = calibration_small
    cl_path = staging_small.root / "centerline_source.csv"
    staging_small.centerline_source.to_csv(cl_path)
    return DatasetBuildConfig.model_validate(
        {
            "dataset_id": "syn_0c_small",
            "source_frame": {"mode": "explicit_identity", "note": "synthetic survey frame"},
            "camera": {"mode": "e57_pinhole", "convention_file": str(conv_path)},
            "split": {"test_every": 3},
            "centerline": {"file": str(cl_path), "frame": "SOURCE"},
            "geometry_holdout": {"ranges_m": [[20.0, 26.0]]},
            "initialization": {"voxel_m": 0.05, "max_points": 120_000, "sparse_max_points": 20_000},
            "capture_epoch": {"id": "ep1"},
        }
    )


@pytest.fixture(scope="session")
def dataset_small(staging_small, build_config_small, tmp_path_factory):
    from minegs.dataset.materialize import build_dataset

    out = tmp_path_factory.mktemp("ds") / "dataset"
    return build_dataset(staging_small.staging_dir, out, build_config_small)


@pytest.fixture(scope="session")
def golden_gate_small(dataset_small, staging_small, tmp_path_factory):
    from minegs.dataset.golden_gate import run_golden_gate

    out = tmp_path_factory.mktemp("gg")
    return run_golden_gate(dataset_small.dataset_dir, staging_small.staging_dir, out), out
