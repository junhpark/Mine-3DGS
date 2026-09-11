"""Point-to-point SE(3) ICP with trimmed correspondences (robust refine step of §7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from minegs.core.frames import SE3
from minegs.eval.register.sim3 import umeyama


@dataclass
class ICPResult:
    T: SE3  # target_from_source
    rmse_m: float
    inlier_ratio: float
    iterations: int
    converged: bool
    history: list[float]


def icp_point_to_point(
    source: np.ndarray,
    target: np.ndarray,
    T_init: SE3 | None = None,
    max_dist_m: float = 0.5,
    max_iters: int = 50,
    tol: float = 1e-6,
    trim_ratio: float = 1.0,
    max_source_points: int = 200_000,
    seed: int = 0,
) -> ICPResult:
    src = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if len(src) > max_source_points:
        src = src[np.random.default_rng(seed).choice(len(src), max_source_points, replace=False)]
    tree = cKDTree(tgt)
    T = T_init or SE3.identity()
    prev = np.inf
    history: list[float] = []
    converged = False
    it = 0
    for it in range(1, max_iters + 1):  # noqa: B007 — reported in the result
        cur = T.apply(src)
        d, j = tree.query(cur, distance_upper_bound=max_dist_m)
        m = np.isfinite(d)
        if m.sum() < 3:
            break
        # trimmed: keep the closest trim_ratio fraction of matches
        keep = np.flatnonzero(m)
        if trim_ratio < 1.0:
            k = max(3, int(len(keep) * trim_ratio))
            keep = keep[np.argsort(d[keep])[:k]]
        rmse = float(np.sqrt(np.mean(d[keep] ** 2)))
        history.append(rmse)
        step = umeyama(cur[keep], tgt[j[keep]], with_scale=False).se3()
        T = step @ T
        if abs(prev - rmse) < tol:
            converged = True
            break
        prev = rmse
    cur = T.apply(src)
    d, _ = tree.query(cur, distance_upper_bound=max_dist_m)
    m = np.isfinite(d)
    rmse = float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else float("inf")
    return ICPResult(T, rmse, float(m.mean()), it, converged, history)
