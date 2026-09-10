"""Centerline (§10): first-class artifact that chunking, sections, volume and change all
reference. Chainage ``s`` is arc length along the polyline (m).

Import from design (CSV: ``x,y,z`` or ``s,x,y,z``), or extract from a TLS cloud
(``extract_from_points``: PCA axis -> binned centroids -> smoothed polyline; good enough for
straight-ish drifts, override with a design line for curved headings).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from minegs.core.errors import ContractError
from minegs.core.frames import SE3


@dataclass
class Centerline:
    vertices: np.ndarray  # (N,3)
    frame: str = "TLS_GLOBAL"
    source: str = "design"  # design | extracted
    chainage_offset_m: float = 0.0  # s of the first vertex

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        if len(v) < 2:
            raise ContractError("centerline needs >= 2 vertices")
        seg = np.linalg.norm(np.diff(v, axis=0), axis=1)
        keep = np.concatenate([[True], seg > 1e-9])
        v = v[keep]
        if len(v) < 2:
            raise ContractError("centerline is degenerate")
        self.vertices = v
        self._seg = np.linalg.norm(np.diff(v, axis=0), axis=1)
        self._s = np.concatenate([[0.0], np.cumsum(self._seg)]) + self.chainage_offset_m

    # ---------------------------------------------------------------- basics
    @property
    def chainage(self) -> np.ndarray:
        return self._s

    @property
    def s_start(self) -> float:
        return float(self._s[0])

    @property
    def s_end(self) -> float:
        return float(self._s[-1])

    @property
    def length(self) -> float:
        return float(self._s[-1] - self._s[0])

    def _locate(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.clip(np.asarray(s, dtype=np.float64), self.s_start, self.s_end)
        i = np.clip(np.searchsorted(self._s, s, side="right") - 1, 0, len(self._seg) - 1)
        u = (s - self._s[i]) / np.maximum(self._seg[i], 1e-12)
        return i, np.clip(u, 0.0, 1.0)

    def point_at(self, s: float | np.ndarray) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        i, u = self._locate(arr)
        p = self.vertices[i] + (self.vertices[i + 1] - self.vertices[i]) * u[:, None]
        return p[0] if np.isscalar(s) or np.ndim(s) == 0 else p

    def tangent_at(self, s: float | np.ndarray) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        i, _ = self._locate(arr)
        d = self.vertices[i + 1] - self.vertices[i]
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        return d[0] if np.isscalar(s) or np.ndim(s) == 0 else d

    def frame_at(self, s: float, up: np.ndarray | None = None) -> SE3:
        """Section frame at chainage s: x = tangent, z ~ up, y = z x x. ``T_line_from_section``."""
        t = self.tangent_at(float(s))
        up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=np.float64)
        if abs(float(np.dot(t, up))) > 0.99:  # vertical shaft: pick another up
            up = np.array([0.0, 1.0, 0.0])
        y = np.cross(up, t)
        y /= np.linalg.norm(y)
        z = np.cross(t, y)
        R = np.column_stack([t, y, z])
        return SE3(R, self.point_at(float(s)))

    def frames_at(
        self, s: np.ndarray, up: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised ``frame_at``: returns (R (N,3,3), origin (N,3)) with columns [t, y, z]."""
        arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        t = self.tangent_at(arr).reshape(-1, 3)
        up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=np.float64)
        ups = np.broadcast_to(up, t.shape).copy()
        vertical = np.abs(t @ up) > 0.99
        ups[vertical] = np.array([0.0, 1.0, 0.0])
        y = np.cross(ups, t)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        z = np.cross(t, y)
        R = np.stack([t, y, z], axis=-1)
        return R, self.point_at(arr).reshape(-1, 3)

    def project(
        self, points: np.ndarray, fine_step_m: float = 0.25
    ) -> tuple[np.ndarray, np.ndarray]:
        """For each point: chainage s of the closest polyline point and radial distance (m).

        Nearest vertex of a finely resampled polyline via KD-tree, then exact projection onto
        the two segments adjacent to it. Memory is O(N), not O(N * segments).
        """
        from scipy.spatial import cKDTree

        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(p) == 0:
            return np.zeros(0), np.zeros(0)
        n_fine = max(2, int(np.ceil(self.length / fine_step_m)) + 1)
        s_fine = np.linspace(self.s_start, self.s_end, n_fine)
        v_fine = self.point_at(s_fine)
        _, k = cKDTree(v_fine).query(p)
        best_s = s_fine[k].copy()
        best_r = np.linalg.norm(p - v_fine[k], axis=1)
        for k0, k1 in ((np.maximum(k - 1, 0), k), (k, np.minimum(k + 1, n_fine - 1))):
            a, b = v_fine[k0], v_fine[k1]
            d = b - a
            seg2 = np.maximum(np.einsum("ij,ij->i", d, d), 1e-12)
            u = np.clip(np.einsum("ij,ij->i", p - a, d) / seg2, 0.0, 1.0)
            closest = a + u[:, None] * d
            r = np.linalg.norm(p - closest, axis=1)
            better = r < best_r
            best_r = np.where(better, r, best_r)
            best_s = np.where(better, s_fine[k0] + u * (s_fine[k1] - s_fine[k0]), best_s)
        return best_s, best_r

    def resample(self, interval_m: float) -> Centerline:
        n = max(2, int(np.floor(self.length / interval_m)) + 1)
        s = np.linspace(self.s_start, self.s_end, n)
        return Centerline(self.point_at(s), self.frame, self.source, self.s_start)

    def transformed(self, T: SE3, frame: str) -> Centerline:
        return Centerline(T.apply(self.vertices), frame, self.source, self.chainage_offset_m)

    def stations(
        self, interval_m: float, start: float | None = None, end: float | None = None
    ) -> np.ndarray:
        s0 = self.s_start if start is None else max(start, self.s_start)
        s1 = self.s_end if end is None else min(end, self.s_end)
        if s1 <= s0:
            return np.zeros(0)
        return np.arange(s0, s1 + 1e-9, interval_m)

    # ---------------------------------------------------------------- io
    @classmethod
    def from_csv(
        cls, path: str | Path, frame: str = "TLS_GLOBAL", source: str = "design"
    ) -> Centerline:
        path = Path(path)
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        rows = [r for r in rows if r and not r[0].startswith("#")]
        if not rows:
            raise ContractError(f"{path}: empty centerline")
        header = [h.strip().lower() for h in rows[0]]
        has_header = any(h in ("x", "y", "z", "s", "chainage", "chainage_m") for h in header)
        body = rows[1:] if has_header else rows
        data = np.array(
            [[float(v) for v in r[: len(header) if has_header else len(r)]] for r in body]
        )
        offset = 0.0
        if has_header:
            ix, iy, iz = header.index("x"), header.index("y"), header.index("z")
            xyz = data[:, [ix, iy, iz]]
            for key in ("s", "chainage", "chainage_m"):
                if key in header:
                    offset = float(data[0, header.index(key)])
                    break
        elif data.shape[1] == 4:
            offset = float(data[0, 0])
            xyz = data[:, 1:4]
        elif data.shape[1] == 3:
            xyz = data
        else:
            raise ContractError(f"{path}: expected 3 or 4 columns")
        return cls(xyz, frame, source, offset)

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["chainage_m", "x", "y", "z"])
            for s, p in zip(self._s, self.vertices, strict=True):
                w.writerow([f"{s:.4f}", f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])
        return path

    # ---------------------------------------------------------------- extraction
    @classmethod
    def extract_from_points(
        cls,
        points: np.ndarray,
        bin_m: float = 2.0,
        smooth_bins: int = 3,
        min_points_per_bin: int = 20,
        frame: str = "TLS_GLOBAL",
    ) -> Centerline:
        """PCA principal axis -> bins along it -> centroid per bin -> moving-average smoothing."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(p) < 2 * min_points_per_bin:
            raise ContractError("too few points to extract a centerline")
        c = p.mean(axis=0)
        _, _, vt = np.linalg.svd(p - c, full_matrices=False)
        axis = vt[0]
        if axis[np.argmax(np.abs(axis))] < 0:  # deterministic direction
            axis = -axis
        t = (p - c) @ axis
        bins = np.floor((t - t.min()) / bin_m).astype(int)
        n_bins = int(bins.max()) + 1
        centroids = []
        for b in range(n_bins):
            m = bins == b
            if m.sum() >= min_points_per_bin:
                centroids.append(p[m].mean(axis=0))
        if len(centroids) < 2:
            raise ContractError("centerline extraction failed: too few populated bins")
        v = np.array(centroids)
        if smooth_bins > 1 and len(v) > smooth_bins:
            k = np.ones(smooth_bins) / smooth_bins
            sm = np.column_stack([np.convolve(v[:, i], k, mode="valid") for i in range(3)])
            v = np.vstack(
                [v[: smooth_bins // 2], sm, v[len(v) - (smooth_bins - 1 - smooth_bins // 2) :]]
            )
        return cls(v, frame, "extracted")
