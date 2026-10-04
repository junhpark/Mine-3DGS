"""Training supervision artifacts that are not initialisation (Phase 4)."""

from minegs.train.supervision.depth import (
    DepthSupervisionRecord,
    VerifiedDepthSupervision,
    verify_depth_supervision,
)

__all__ = ["DepthSupervisionRecord", "VerifiedDepthSupervision", "verify_depth_supervision"]
