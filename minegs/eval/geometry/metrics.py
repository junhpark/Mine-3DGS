"""Bidirectional geometry metrics (§11): accuracy (pred -> ref), completeness (ref -> pred),
symmetric Chamfer, median / P90 / P95 / RMSE. One direction alone hides holes."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial import cKDTree


class DistanceStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int
    mean_m: float
    rmse_m: float
    median_m: float
    p90_m: float
    p95_m: float
    max_m: float
    ratio_within_tau: dict[str, float] = Field(default_factory=dict)  # e.g. {"0.02": 0.91}
    clipped_ratio: float = 0.0  # fraction beyond max_dist (outliers / holes)


class GeometryReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame: str = "TLS_GLOBAL"
    max_dist_m: float
    accuracy: DistanceStats  # pred -> ref
    completeness: DistanceStats  # ref -> pred
    chamfer_m: float  # mean(acc.mean, comp.mean)
    f_score: dict[str, float] = Field(default_factory=dict)
    chainage_range_m: tuple[float, float] | None = None
    claim: str = "geometry_diagnostic"


def nn_distances(
    query: np.ndarray, reference: np.ndarray, max_dist: float | None = None
) -> np.ndarray:
    d, _ = cKDTree(np.asarray(reference, dtype=np.float64)).query(
        np.asarray(query, dtype=np.float64), distance_upper_bound=max_dist if max_dist else np.inf
    )
    return d


def stats(
    d: np.ndarray, max_dist: float, taus: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1)
) -> DistanceStats:
    d = np.asarray(d, dtype=np.float64)
    inf = ~np.isfinite(d)
    clipped = float(inf.mean()) if len(d) else 0.0
    dc = np.where(inf, max_dist, d)
    if len(dc) == 0:
        return DistanceStats(
            n=0,
            mean_m=float("nan"),
            rmse_m=float("nan"),
            median_m=float("nan"),
            p90_m=float("nan"),
            p95_m=float("nan"),
            max_m=float("nan"),
            clipped_ratio=0.0,
        )
    return DistanceStats(
        n=len(d),
        mean_m=float(dc.mean()),
        rmse_m=float(np.sqrt((dc**2).mean())),
        median_m=float(np.median(dc)),
        p90_m=float(np.quantile(dc, 0.9)),
        p95_m=float(np.quantile(dc, 0.95)),
        max_m=float(dc.max()),
        ratio_within_tau={f"{t:g}": float((dc <= t).mean()) for t in taus},
        clipped_ratio=clipped,
    )


def compare_clouds(
    pred: np.ndarray,
    ref: np.ndarray,
    max_dist_m: float = 1.0,
    taus: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1),
    max_points: int = 2_000_000,
    seed: int = 0,
    frame: str = "TLS_GLOBAL",
) -> GeometryReport:
    rng = np.random.default_rng(seed)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1, 3)
    if len(pred) > max_points:
        pred = pred[rng.choice(len(pred), max_points, replace=False)]
    if len(ref) > max_points:
        ref = ref[rng.choice(len(ref), max_points, replace=False)]
    acc = stats(nn_distances(pred, ref, max_dist_m), max_dist_m, taus)
    comp = stats(nn_distances(ref, pred, max_dist_m), max_dist_m, taus)
    f = {}
    for t in taus:
        k = f"{t:g}"
        p, r = acc.ratio_within_tau[k], comp.ratio_within_tau[k]
        f[k] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return GeometryReport(
        frame=frame,
        max_dist_m=max_dist_m,
        accuracy=acc,
        completeness=comp,
        chamfer_m=0.5 * (acc.mean_m + comp.mean_m),
        f_score=f,
    )
