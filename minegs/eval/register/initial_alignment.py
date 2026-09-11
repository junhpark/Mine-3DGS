"""Initial Sim(3) from known targets / station correspondences (§7), with RANSAC."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from minegs.core.errors import ContractError
from minegs.core.frames import Sim3
from minegs.eval.register.sim3 import umeyama


def load_targets_csv(path: str | Path) -> dict[str, np.ndarray]:
    """``id,x,y,z`` rows -> {id: xyz}."""
    out: dict[str, np.ndarray] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["id"].strip()] = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
    if not out:
        raise ContractError(f"{path}: no targets")
    return out


def match_by_id(
    src: dict[str, np.ndarray], dst: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ids = sorted(set(src) & set(dst))
    if len(ids) < 3:
        raise ContractError(f"need >= 3 common target ids, got {ids}")
    return np.array([src[i] for i in ids]), np.array([dst[i] for i in ids]), ids


def align_correspondences(
    src: np.ndarray,
    dst: np.ndarray,
    with_scale: bool = True,
    ransac_iters: int = 200,
    inlier_m: float = 0.1,
    seed: int = 0,
) -> tuple[Sim3, np.ndarray]:
    """Robust Umeyama: returns (T dst_from_src, inlier mask). Small n -> plain fit."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    n = len(src)
    if n < 3:
        raise ContractError("need >= 3 correspondences")
    if n <= 4:
        return umeyama(src, dst, with_scale), np.ones(n, dtype=bool)
    rng = np.random.default_rng(seed)
    best_mask = np.zeros(n, dtype=bool)
    for _ in range(ransac_iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            T = umeyama(src[idx], dst[idx], with_scale)
        except Exception:
            continue
        err = np.linalg.norm(T.apply(src) - dst, axis=1)
        mask = err < inlier_m
        if mask.sum() > best_mask.sum():
            best_mask = mask
    if best_mask.sum() < 3:
        best_mask = np.ones(n, dtype=bool)
    return umeyama(src[best_mask], dst[best_mask], with_scale), best_mask
