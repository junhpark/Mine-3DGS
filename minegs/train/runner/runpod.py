"""RunPodRunner (§8.2) — **Phase 6, not runnable yet**.

``submit`` raises ``NotYetImplementedError`` before touching any external service, so the
CLI/README cannot present this path as working. What Phase 6 has to deliver, in order:

1. ``sync.push`` dataset/ (never raw/) to an rclone remote that is *mounted as the pod's
   network volume* (or a pod-side ``rclone copy`` from that remote into ``/data``). The
   current sketch pushed to a remote without any step that lands it inside the pod.
2. Create the pod from the same GPU image digest as LocalRunner, with the staged dataset
   built pod-side (``minegs.train.staging``), checkpoints on the volume, ``--resume-from``
   resolved pod-side (``minegs.train.runner.resume``: host discovery, one namespace per path).
3. Poll with the *container exit code*, not the pod lifecycle state: EXITED/TERMINATED is
   not success.
4. Pod-side ``rclone copy`` of ``/data/runs/<run_id>/`` back to the remote, then local
   ``sync.pull`` — again a step the sketch lacked.
5. run.json records docker_digest, pod GPU type, and both sync hashes (§9).
"""

from __future__ import annotations

from pathlib import Path

from minegs.core.errors import NotYetImplementedError
from minegs.train.runner.base import RunConfig, RunHandle, Runner


class RunPodRunner(Runner):
    name = "runpod"

    def submit(self, run: RunConfig) -> RunHandle:
        raise NotYetImplementedError(
            "RunPodRunner (pod creation, volume sync in/out, exit-code polling)", "6"
        )

    def terminate(self, handle: RunHandle) -> None:
        raise NotYetImplementedError("RunPodRunner.terminate", "6")


def _cli_note() -> str:
    return (
        "RunPod execution is Phase 6 and not implemented; run locally "
        "(--runner local) or see minegs/train/runner/runpod.py for the plan."
    )


__all__ = ["Path", "RunPodRunner", "_cli_note"]
