"""Phase 2: carrying one E57 survey through the whole evidence chain (§Phase 2).

A thin orchestration layer. It owns ordering, state, artifact identity and timing, and it owns
nothing else: every stage calls the same domain function the CLI calls, and every artifact is
checked by the same validator that checks it on the hand-run path.
"""

from minegs.e2e.models import (
    STAGE_ORDER,
    Phase2Report,
    Stage,
    StageRecord,
    StageStatus,
    WorkflowState,
)

__all__ = [
    "STAGE_ORDER",
    "Phase2Report",
    "Stage",
    "StageRecord",
    "StageStatus",
    "WorkflowState",
]
