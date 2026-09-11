"""Closed-form Sim(3)/SE(3) from correspondences (Umeyama 1991)."""

from __future__ import annotations

import numpy as np

from minegs.core.errors import ContractError
from minegs.core.frames import Sim3


def umeyama(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = True, weights: np.ndarray | None = None
) -> Sim3:
    """Return T with ``dst ≈ T.apply(src)``. ``with_scale=False`` gives s = 1 (SE3)."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    n = len(src)
    if n < 3 or len(dst) != n:
        raise ContractError("umeyama needs >= 3 matched correspondences")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    mu_s = (w[:, None] * src).sum(0)
    mu_d = (w[:, None] * dst).sum(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = (w[:, None, None] * (xd[:, :, None] @ xs[:, None, :])).sum(0)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var_s = (w * (xs**2).sum(1)).sum()
        s = float(np.trace(np.diag(D) @ S) / var_s)
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return Sim3(s, R, t)
