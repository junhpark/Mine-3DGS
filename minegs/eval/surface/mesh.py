"""Triangle-mesh helpers used by volume (§11) — pure numpy; no open3d needed."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from minegs.core.errors import ContractError


def read_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    v, f = [], []
    for line in Path(path).read_text().splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "v":
            v.append([float(x) for x in tok[1:4]])
        elif tok[0] == "f":
            idx = [int(t.split("/")[0]) - 1 for t in tok[1:]]
            for k in range(1, len(idx) - 1):  # fan-triangulate
                f.append([idx[0], idx[k], idx[k + 1]])
    if not v or not f:
        raise ContractError(f"{path}: empty mesh")
    return np.array(v), np.array(f, dtype=np.int64)


def write_obj(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for p in vertices:
            fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for t in faces:
            fh.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
    return path


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume via divergence theorem. Needs a closed, consistently oriented mesh."""
    v = np.asarray(vertices, dtype=np.float64)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def sample_mesh_surface(
    vertices: np.ndarray, faces: np.ndarray, n: int, seed: int = 0
) -> np.ndarray:
    """Uniform area-weighted surface samples (for GS-mesh vs TLS comparison)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(vertices, dtype=np.float64)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if area.sum() <= 0:
        raise ContractError("degenerate mesh")
    fi = rng.choice(len(faces), size=n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    return (1 - r1)[:, None] * a[fi] + (r1 * (1 - r2))[:, None] * b[fi] + (r1 * r2)[:, None] * c[fi]
