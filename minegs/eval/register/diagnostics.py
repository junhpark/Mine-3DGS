"""Registration quality -> ``manifest.registration`` (scale, rmse_m, inlier_ratio, transform).
A registration without these numbers cannot be used for evaluation (§7)."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.spatial import cKDTree

from minegs.core.frames import SE3, Sim3
from minegs.core.manifest import Registration


class RegistrationDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str
    scale: float
    rmse_m: float
    median_m: float
    p90_m: float
    inlier_ratio: float
    inlier_threshold_m: float
    n_source: int
    transform: list[list[float]]

    def to_manifest(self) -> Registration:
        return Registration(
            method=self.method,
            scale=self.scale,
            rmse_m=self.rmse_m,
            inlier_ratio=self.inlier_ratio,
            transform=self.transform,
            n_correspondences=self.n_source,
            inlier_threshold_m=self.inlier_threshold_m,
        )


def diagnose(
    T: Sim3 | SE3,
    source: np.ndarray,
    target: np.ndarray,
    inlier_m: float = 0.1,
    method: str = "sim3+icp",
    max_points: int = 500_000,
) -> RegistrationDiagnostics:
    src = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    if len(src) > max_points:
        src = src[np.random.default_rng(0).choice(len(src), max_points, replace=False)]
    d, _ = cKDTree(np.asarray(target, dtype=np.float64)).query(T.apply(src))
    inl = d < inlier_m
    scale = T.s if isinstance(T, Sim3) else 1.0
    return RegistrationDiagnostics(
        method=method,
        scale=float(scale),
        rmse_m=float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else float("inf"),
        median_m=float(np.median(d)),
        p90_m=float(np.quantile(d, 0.9)),
        inlier_ratio=float(inl.mean()),
        inlier_threshold_m=inlier_m,
        n_source=len(src),
        transform=T.to_list(),
    )
