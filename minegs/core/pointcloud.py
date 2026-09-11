"""Point cloud container + PLY I/O (no external deps).

Handles the two shapes we care about:
* plain xyz(+rgb, +normals) clouds — init_points.ply, TLS tiles, surface samples;
* raw 3DGS parameter clouds (``f_dc_*``, ``opacity``, ``scale_*``, ``rot_*``) are kept as
  ``extra`` columns so a run's ``point_cloud/*.ply`` can be re-expressed in LOCAL_METRIC
  (§3, §8.1) without understanding every attribute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from minegs.core.errors import ContractError
from minegs.core.frames import SE3, Sim3, check_float32_safe

_PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}
_NP_TO_PLY = {
    "i1": "char",
    "u1": "uchar",
    "i2": "short",
    "u2": "ushort",
    "i4": "int",
    "u4": "uint",
    "f4": "float",
    "f8": "double",
}


@dataclass
class PointCloud:
    xyz: np.ndarray  # (N,3) float64
    rgb: np.ndarray | None = None  # (N,3) uint8
    normals: np.ndarray | None = None  # (N,3) float32/64
    extra: dict[str, np.ndarray] = field(default_factory=dict)  # per-vertex scalar columns
    frame: str = "UNKNOWN"

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=np.float64).reshape(-1, 3)
        n = len(self.xyz)
        if self.rgb is not None:
            self.rgb = np.asarray(self.rgb).reshape(-1, 3)
            if self.rgb.dtype != np.uint8:
                self.rgb = np.clip(np.rint(self.rgb), 0, 255).astype(np.uint8)
            if len(self.rgb) != n:
                raise ContractError("rgb length mismatch")
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float64).reshape(-1, 3)
            if len(self.normals) != n:
                raise ContractError("normals length mismatch")
        for k, v in self.extra.items():
            if len(v) != n:
                raise ContractError(f"extra column {k!r} length mismatch")

    def __len__(self) -> int:
        return len(self.xyz)

    def transformed(self, T: SE3 | Sim3, frame: str | None = None) -> PointCloud:
        xyz = T.apply(self.xyz)
        normals = None
        if self.normals is not None:
            normals = self.normals @ T.R.T
        extra = dict(self.extra)
        # 3DGS scale attributes are log-scale: under Sim3 they shift by log(s)
        if isinstance(T, Sim3) and abs(T.s - 1) > 1e-12:
            for k in list(extra):
                if k.startswith("scale_"):
                    extra[k] = extra[k] + np.log(T.s)
        # 3DGS rotations (rot_0..3 = w,x,y,z) rotate by R
        if {"rot_0", "rot_1", "rot_2", "rot_3"} <= set(extra):
            from minegs.core.frames import quat_to_rotmat, rotmat_to_quat

            q = np.stack([extra[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
            out = np.empty_like(q)
            for i in range(len(q)):
                out[i] = rotmat_to_quat(T.R @ quat_to_rotmat(q[i]))
            for i in range(4):
                extra[f"rot_{i}"] = out[:, i].astype(extra[f"rot_{i}"].dtype)
        return PointCloud(xyz, self.rgb, normals, extra, frame or self.frame)

    def subsample(self, n: int, seed: int = 0) -> PointCloud:
        if n >= len(self):
            return self
        idx = np.random.default_rng(seed).choice(len(self), size=n, replace=False)
        return self.select(np.sort(idx))

    def select(self, idx: np.ndarray) -> PointCloud:
        return PointCloud(
            self.xyz[idx],
            None if self.rgb is None else self.rgb[idx],
            None if self.normals is None else self.normals[idx],
            {k: v[idx] for k, v in self.extra.items()},
            self.frame,
        )

    def concat(self, other: PointCloud) -> PointCloud:
        rgb = None
        if self.rgb is not None and other.rgb is not None:
            rgb = np.concatenate([self.rgb, other.rgb])
        normals = None
        if self.normals is not None and other.normals is not None:
            normals = np.concatenate([self.normals, other.normals])
        extra = {
            k: np.concatenate([v, other.extra[k]])
            for k, v in self.extra.items()
            if k in other.extra
        }
        return PointCloud(np.concatenate([self.xyz, other.xyz]), rgb, normals, extra, self.frame)

    def centroid(self) -> np.ndarray:
        return self.xyz.mean(axis=0) if len(self) else np.zeros(3)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.xyz.min(axis=0), self.xyz.max(axis=0)

    def is_gaussian_cloud(self) -> bool:
        return "opacity" in self.extra and "scale_0" in self.extra

    def check_float32_safe(self) -> float:
        return check_float32_safe(self.xyz, f"point cloud in frame {self.frame}")


def voxel_downsample(xyz: np.ndarray, voxel_m: float) -> np.ndarray:
    """Indices of one representative point per voxel (first occurrence)."""
    if voxel_m <= 0:
        return np.arange(len(xyz))
    keys = np.floor(xyz / voxel_m).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return np.sort(idx)


# ---------------------------------------------------------------- PLY read / write


def write_ply(pc: PointCloud, path: str | Path, binary: bool = True, xyz_dtype: str = "f4") -> Path:
    """Write vertices. Default float32 xyz — caller must be in a small frame (LOCAL_METRIC)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if xyz_dtype == "f4":
        pc.check_float32_safe()
    cols: list[tuple[str, np.ndarray]] = [
        ("x", pc.xyz[:, 0].astype(xyz_dtype)),
        ("y", pc.xyz[:, 1].astype(xyz_dtype)),
        ("z", pc.xyz[:, 2].astype(xyz_dtype)),
    ]
    if pc.normals is not None:
        cols += [
            ("nx", pc.normals[:, 0].astype("f4")),
            ("ny", pc.normals[:, 1].astype("f4")),
            ("nz", pc.normals[:, 2].astype("f4")),
        ]
    if pc.rgb is not None:
        cols += [("red", pc.rgb[:, 0]), ("green", pc.rgb[:, 1]), ("blue", pc.rgb[:, 2])]
    for k, v in pc.extra.items():
        v = np.asarray(v)
        if v.dtype.kind == "f" and v.dtype.itemsize > 4:
            v = v.astype("f4")
        cols.append((k, v))
    dtype = np.dtype([(k, v.dtype.str) for k, v in cols])
    rec = np.empty(len(pc), dtype=dtype)
    for k, v in cols:
        rec[k] = v
    header = ["ply", "format binary_little_endian 1.0" if binary else "format ascii 1.0"]
    header.append(f"comment minegs frame={pc.frame}")
    header.append(f"element vertex {len(pc)}")
    for k, v in cols:
        header.append(f"property {_NP_TO_PLY[v.dtype.str.lstrip('<>|=')]} {k}")
    header.append("end_header\n")
    with open(path, "wb") as f:
        f.write("\n".join(header).encode("ascii"))
        if binary:
            f.write(rec.astype(dtype.newbyteorder("<")).tobytes())
        else:
            np.savetxt(f, np.column_stack([rec[k] for k, _ in cols]), fmt="%.8g")
    return path


def read_ply(path: str | Path) -> PointCloud:
    path = Path(path)
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ContractError(f"{path}: not a PLY file")
        fmt = None
        n_vertex = 0
        props: list[tuple[str, str]] = []
        frame = "UNKNOWN"
        in_vertex = False
        other_elements: list[tuple[str, int, list[tuple[str, str]]]] = []
        while True:
            line = f.readline()
            if not line:
                raise ContractError(f"{path}: truncated header")
            tok = line.decode("ascii", "replace").strip().split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "comment":
                if len(tok) >= 3 and tok[2].startswith("frame="):
                    frame = tok[2].split("=", 1)[1]
            elif tok[0] == "element":
                in_vertex = tok[1] == "vertex"
                if in_vertex:
                    n_vertex = int(tok[2])
                else:
                    other_elements.append((tok[1], int(tok[2]), []))
            elif tok[0] == "property":
                if tok[1] == "list":
                    if in_vertex:
                        raise ContractError(f"{path}: list properties on vertex not supported")
                    other_elements[-1][2].append(("list", tok[-1]))
                elif in_vertex:
                    props.append((tok[2], _PLY_TYPES[tok[1]]))
                else:
                    other_elements[-1][2].append((tok[2], _PLY_TYPES[tok[1]]))
            elif tok[0] == "end_header":
                break
        if fmt is None:
            raise ContractError(f"{path}: missing format line")
        if fmt == "ascii":
            data = (
                np.loadtxt(f, max_rows=n_vertex, ndmin=2) if n_vertex else np.zeros((0, len(props)))
            )
            rec = {name: data[:, i].astype(t) for i, (name, t) in enumerate(props)}
        else:
            order = "<" if fmt == "binary_little_endian" else ">"
            dtype = np.dtype([(name, order + t) for name, t in props])
            buf = f.read(dtype.itemsize * n_vertex)
            arr = np.frombuffer(buf, dtype=dtype, count=n_vertex)
            rec = {name: arr[name] for name, _ in props}
    for k in ("x", "y", "z"):
        if k not in rec:
            raise ContractError(f"{path}: missing vertex property {k}")
    xyz = np.column_stack([rec.pop("x"), rec.pop("y"), rec.pop("z")]).astype(np.float64)
    rgb = None
    if {"red", "green", "blue"} <= set(rec):
        rgb = np.column_stack([rec.pop("red"), rec.pop("green"), rec.pop("blue")])
    normals = None
    if {"nx", "ny", "nz"} <= set(rec):
        normals = np.column_stack([rec.pop("nx"), rec.pop("ny"), rec.pop("nz")]).astype(np.float64)
    rec.pop("alpha", None)
    return PointCloud(xyz, rgb, normals, {k: np.asarray(v) for k, v in rec.items()}, frame)
