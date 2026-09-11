"""Registration (§7): initial_alignment -> Sim(3) -> SE(3) ICP -> diagnostics -> manifest."""

from minegs.eval.register.diagnostics import RegistrationDiagnostics, diagnose
from minegs.eval.register.initial_alignment import align_correspondences
from minegs.eval.register.rigid_icp import icp_point_to_point
from minegs.eval.register.sim3 import umeyama

__all__ = [
    "RegistrationDiagnostics",
    "align_correspondences",
    "diagnose",
    "icp_point_to_point",
    "umeyama",
]
