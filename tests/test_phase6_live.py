"""Phase 6 live RunPod smoke — never runs unless explicitly asked for (docs/PHASE6_ACCEPTANCE.md).

Billable. It needs ALL of: ``-m runpod_live``, ``MINEGS_RUNPOD_LIVE=1``, ``RUNPOD_API_KEY``,
``MINEGS_RUNPOD_CONFIG`` (a runner config with a digest-pinned image, network volume and storage
remote) and ``MINEGS_RUNPOD_DATASET`` (a small materialised dataset). A credential alone never
starts it; CI never sets the opt-in.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.runpod_live

_REASON = "live RunPod is opt-in: MINEGS_RUNPOD_LIVE=1, RUNPOD_API_KEY, MINEGS_RUNPOD_CONFIG, MINEGS_RUNPOD_DATASET"


@pytest.mark.skipif(
    os.environ.get("MINEGS_RUNPOD_LIVE") != "1"
    or not all(
        os.environ.get(k)
        for k in ("RUNPOD_API_KEY", "MINEGS_RUNPOD_CONFIG", "MINEGS_RUNPOD_DATASET")
    ),
    reason=_REASON,
)
def test_one_light_run_on_a_real_pod(tmp_path):  # pragma: no cover - billable, opt-in only
    from minegs.train.runner import RunConfig
    from minegs.train.runner.base import RunnerConfig, RunStatus, load_record
    from minegs.train.runner.runpod import RunPodRunner

    runner = RunPodRunner(RunnerConfig.load(os.environ["MINEGS_RUNPOD_CONFIG"]))
    h = runner.submit(
        RunConfig(
            dataset_dir=os.environ["MINEGS_RUNPOD_DATASET"],
            profile="light",
            run_dir=str(tmp_path / "run"),
            overrides={"max_steps": 200},
        )
    )
    try:
        st = h.wait(poll_s=30, timeout_s=3 * 3600)
    finally:
        if h.status() not in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED):
            runner.terminate(h)
    assert st is RunStatus.SUCCEEDED, load_record(h.run_dir).failure_reason
