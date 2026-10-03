"""Where a depth sample lives along the drift, and whether it may train (Phase 4 AD-3).

A depth sample is a statement about the tunnel along one camera ray: the surface at this pixel
is this far away. Two things can make it evidence about the evaluation holdout. The surface
point can lie inside a held-out chainage range, which is the obvious case. Or the ray can pass
through a held-out range on its way to a surface beyond it. Then the sample says "nothing
stands between the camera and that wall", which is a claim about the holdout geometry. If the
holdout geometry was removed before projecting, it is also wrong. Both are refused here.

The ray rule has two parts, and a pair is refused if either fires. The interval between the
camera's chainage and the point's, widened by a margin, must not touch a holdout range. On a
straight centerline chainage is monotone along any segment, so that interval is exactly the set
of chainages the segment visits. On a bent one it is not: inside a bend, chainage jumps across
the bisector, and a segment can pass through held-out chainage while both of its ends are
metres outside it. So every segment that comes within reach of a holdout is also located at
points ``RAY_STEP_M`` apart, and refused if any of them is unlocated or within the margin of a
holdout. What that cannot see is a passage shorter than ``RAY_STEP_M`` through held-out space,
which needs a chainage discontinuity, i.e. the bisector of a bend (docs/PHASE4_CONTRACT.md AD-3).

"Where" has to be known before anything else is decided. A point beyond the ends of the
centerline, or further from it than the tunnel could be, has no chainage. It is clamped onto
the end segment, which looks like an answer and is not one. Such a point is *unlocated*. With a
holdout declared, an unlocated sample is never emitted, and an artifact that carries one is
refused. That is fail-closed rather than a warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from minegs.core.centerline import Centerline
from minegs.core.errors import ContractError
from minegs.core.manifest import Manifest

#: Default lateral reach of the axis, in metres. A mine drift is a few metres across; a point
#: ten metres off the centerline belongs to another heading, or to nothing.
DEFAULT_MAX_RADIAL_M = 10.0
#: Default widening of the camera-to-point chainage interval (AD-3, ray rule), in metres.
DEFAULT_RAY_MARGIN_M = 0.5
#: Spacing of the points at which a camera-to-point segment is located (AD-3, ray rule).
RAY_STEP_M = 0.1
#: Points located per batch by the sampled ray rule (bounds memory on long surveys).
_RAY_CHUNK = 1_000_000
#: Spacing of the axis samples the sampled rule's pre-filter measures distance to.
_AXIS_STEP_M = 0.25

#: Exclusion reasons that are leakage decisions, kept apart from quality ones (confidence 0).
HOLDOUT_POINT = "holdout_point"
HOLDOUT_RAY = "holdout_ray"
UNLOCATED = "unlocated_support"


def dataset_centerline(dataset_dir: str | Path, manifest: Manifest) -> Centerline | None:
    """The dataset's reference axis in LOCAL_METRIC, or None when it declares none."""
    ref = manifest.centerline
    if ref is None:
        return None
    path = Path(dataset_dir) / ref.file
    if not path.is_file():
        raise ContractError(f"{dataset_dir}: manifest names centerline {ref.file}, not present")
    cl = Centerline.from_csv(path, ref.frame, ref.source)
    if ref.frame == "TLS_GLOBAL":
        cl = cl.transformed(manifest.T_local_from_tls, "LOCAL_METRIC")
    return cl


def declared_holdout(manifest: Manifest) -> list[tuple[float, float]]:
    ho = manifest.split.geometry_holdout
    if ho is None:
        return []
    return [(float(lo), float(hi)) for lo, hi in ho.chainage_ranges_m]


@dataclass(frozen=True)
class Support:
    """The axis, the holdout and the two parameters that decide what is located."""

    centerline: Centerline | None
    holdout: list[tuple[float, float]] = field(default_factory=list)
    max_radial_m: float = DEFAULT_MAX_RADIAL_M
    ray_margin_m: float = DEFAULT_RAY_MARGIN_M

    @classmethod
    def of_dataset(
        cls,
        dataset_dir: str | Path,
        manifest: Manifest,
        *,
        max_radial_m: float = DEFAULT_MAX_RADIAL_M,
        ray_margin_m: float = DEFAULT_RAY_MARGIN_M,
    ) -> Support:
        holdout = declared_holdout(manifest)
        cl = dataset_centerline(dataset_dir, manifest)
        if holdout and cl is None:
            raise ContractError(
                f"{dataset_dir}: a geometry holdout {holdout} is declared in chainage, but the "
                "dataset has no centerline to measure chainage along. Which depth samples lie in "
                "the holdout cannot be decided, so none can be admitted (fail closed)."
            )
        if not (max_radial_m > 0 and np.isfinite(max_radial_m)):
            raise ContractError(f"max_radial_m must be a positive distance, got {max_radial_m}")
        if not (ray_margin_m >= 0 and np.isfinite(ray_margin_m)):
            raise ContractError(f"ray_margin_m must be non-negative, got {ray_margin_m}")
        if cl is not None:
            # Past the ends of the axis nothing has a chainage. A holdout that reaches there
            # holds space whose samples could only be judged by clamping onto the axis, which
            # at a bent end lands inside the drift (docs/PHASE4_CONTRACT.md AD-3).
            outside = [
                (lo, hi) for lo, hi in holdout if lo < cl.s_start - 1e-9 or hi > cl.s_end + 1e-9
            ]
            if outside:
                raise ContractError(
                    f"{dataset_dir}: geometry holdout {outside} reaches past the centerline, "
                    f"which spans [{cl.s_start}, {cl.s_end}] m. Which depth samples lie in the "
                    "part beyond the axis cannot be decided (fail closed)."
                )
        return cls(cl, holdout, float(max_radial_m), float(ray_margin_m))

    @property
    def required(self) -> bool:
        """Whether every sample must be located: only when there is a holdout to leak into."""
        return bool(self.holdout)

    def locate(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Chainage of each point and whether it is genuinely located (not clamped, not far)."""
        p = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        if self.centerline is None:
            return np.full(len(p), np.nan), np.zeros(len(p), dtype=bool)
        if not len(p):
            return np.zeros(0), np.zeros(0, dtype=bool)
        cl = self.centerline
        s, r = cl.project(p)
        v = cl.vertices
        t0 = (v[1] - v[0]) / np.linalg.norm(v[1] - v[0])
        t1 = (v[-1] - v[-2]) / np.linalg.norm(v[-1] - v[-2])
        before = (p - v[0]) @ t0 < 0.0
        after = (p - v[-1]) @ t1 > 0.0
        ok = np.isfinite(s) & (r <= self.max_radial_m) & ~before & ~after
        return s, ok

    def in_holdout(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        m = np.zeros(s.shape, dtype=bool)
        for lo, hi in self.holdout:
            m |= (s >= lo) & (s <= hi)
        return m

    def ray_crosses_holdout(self, s_cam: np.ndarray, s_pt: np.ndarray) -> np.ndarray:
        """The interval rule: [min - margin, max + margin] touches a holdout range."""
        a = np.minimum(s_cam, s_pt) - self.ray_margin_m
        b = np.maximum(s_cam, s_pt) + self.ray_margin_m
        m = np.zeros(np.shape(a), dtype=bool)
        for lo, hi in self.holdout:
            m |= (a <= hi) & (b >= lo)
        return m

    def _near_holdout(self, s: np.ndarray) -> np.ndarray:
        m = np.zeros(np.shape(s), dtype=bool)
        for lo, hi in self.holdout:
            m |= (s >= lo - self.ray_margin_m) & (s <= hi + self.ray_margin_m)
        return m

    def _may_reach_holdout(self, cam: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Segments that come close enough to a holdout for the sampled rule to matter.

        A located point within the margin of a holdout lies within ``max_radial_m`` of the axis
        at a held-out chainage. A segment whose midpoint is further than that, plus half its
        length, plus the axis sample spacing and a metre of slack, from every such axis point
        cannot contain one, so it is not sampled.
        """
        from scipy.spatial import cKDTree

        cl = self.centerline
        assert cl is not None
        parts = []
        for lo, hi in self.holdout:
            a = max(lo - self.ray_margin_m, cl.s_start)
            b = min(hi + self.ray_margin_m, cl.s_end)
            n = max(2, int(np.ceil((b - a) / _AXIS_STEP_M)) + 1)
            parts.append(np.linspace(a, b, n))
        axis = cl.point_at(np.concatenate(parts))
        mid = 0.5 * (cam + pts)
        half = 0.5 * np.linalg.norm(pts - cam, axis=1)
        d, _ = cKDTree(axis).query(mid)
        return d - half - _AXIS_STEP_M <= self.max_radial_m + 1.0

    def segment_touches_holdout(
        self, camera_xyz: np.ndarray, point_xyz: np.ndarray, step_m: float = RAY_STEP_M
    ) -> np.ndarray:
        """The sampled rule: a point of the segment, located every ``step_m``, is unlocated or
        within the margin of a holdout. Exact up to passages shorter than ``step_m``."""
        cam = np.asarray(camera_xyz, dtype=np.float64).reshape(-1, 3)
        pts = np.asarray(point_xyz, dtype=np.float64).reshape(-1, 3)
        out = np.zeros(len(pts), dtype=bool)
        if not self.holdout or not len(pts):
            return out
        idx = np.flatnonzero(self._may_reach_holdout(cam, pts))
        if not len(idx):
            return out
        length = np.linalg.norm(pts[idx] - cam[idx], axis=1)
        per = np.maximum(np.ceil(length / step_m).astype(np.int64) + 1, 2)  # ends included
        start = 0
        while start < len(idx):
            stop = start + max(1, int(np.searchsorted(np.cumsum(per[start:]), _RAY_CHUNK, "right")))
            sel, m = idx[start:stop], per[start:stop]
            first = np.cumsum(m) - m
            seg = np.repeat(np.arange(len(sel)), m)
            t = (np.arange(int(m.sum())) - np.repeat(first, m)) / np.repeat(m - 1, m)
            q = cam[sel][seg] + t[:, None] * (pts[sel] - cam[sel])[seg]
            s, ok = self.locate(q)
            out[sel] = np.logical_or.reduceat(~ok | self._near_holdout(s), first)
            start = stop
        return out

    def classify(
        self, camera_xyz: np.ndarray, point_xyz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """For N (camera, point) pairs: which may train, each point's chainage, why not.

        Returns ``(keep, s_point, reasons)``. ``reasons`` maps each leakage reason to a mask,
        assigned in order of precedence (unlocated, then holdout point, then holdout ray), so
        every refused pair has exactly one reason.
        """
        cam = np.asarray(camera_xyz, dtype=np.float64).reshape(-1, 3)
        pts = np.asarray(point_xyz, dtype=np.float64).reshape(-1, 3)
        n = len(pts)
        s_pt, ok_pt = self.locate(pts)
        if not self.required:
            # No holdout: nothing to leak into. Chainage is still reported where it is known.
            empty = np.zeros(n, dtype=bool)
            return (
                np.ones(n, dtype=bool),
                s_pt,
                {UNLOCATED: empty, HOLDOUT_POINT: empty, HOLDOUT_RAY: empty},
            )
        s_cam, ok_cam = self.locate(cam)
        unlocated = ~(ok_pt & ok_cam)
        point = ~unlocated & self.in_holdout(s_pt)
        ray = ~unlocated & ~point & self.ray_crosses_holdout(s_cam, s_pt)
        rest = ~(unlocated | point | ray)
        if rest.any():
            ray[rest] = self.segment_touches_holdout(cam[rest], pts[rest])
        keep = ~(unlocated | point | ray)
        return keep, s_pt, {UNLOCATED: unlocated, HOLDOUT_POINT: point, HOLDOUT_RAY: ray}


def support_ranges(s: np.ndarray, bin_m: float = 1.0) -> list[tuple[float, float]]:
    """Merged chainage intervals covered by a set of samples, on a fixed 1 m grid.

    Deterministic in the samples: the builder and the verifier compute it from the same
    re-derived points, so the recorded value can be compared rather than trusted.
    """
    s = np.asarray(s, dtype=np.float64)
    s = s[np.isfinite(s)]
    if not len(s):
        return []
    bins = np.unique(np.floor(s / bin_m).astype(np.int64))
    out: list[tuple[float, float]] = []
    start = prev = int(bins[0])
    for b in bins[1:]:
        b = int(b)
        if b != prev + 1:
            out.append((start * bin_m, (prev + 1) * bin_m))
            start = b
        prev = b
    out.append((start * bin_m, (prev + 1) * bin_m))
    return out


__all__ = [
    "DEFAULT_MAX_RADIAL_M",
    "DEFAULT_RAY_MARGIN_M",
    "HOLDOUT_POINT",
    "HOLDOUT_RAY",
    "RAY_STEP_M",
    "UNLOCATED",
    "Support",
    "dataset_centerline",
    "declared_holdout",
    "support_ranges",
]
