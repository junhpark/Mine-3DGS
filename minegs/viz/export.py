"""Export (§12): research ``.ply`` (3DGS attributes) -> ``.spz`` (SuperSplat, primary) or
legacy ``.splat``. Both are *distribution* formats; evaluation never reads them."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from minegs.core.errors import ContractError, MissingDependencyError
from minegs.core.pointcloud import PointCloud, read_ply

SH_C0 = 0.28209479177387814


def gaussian_attributes(pc: PointCloud) -> dict[str, np.ndarray]:
    if not pc.is_gaussian_cloud():
        raise ContractError("PLY has no 3DGS attributes (opacity/scale_*/rot_*)")
    e = pc.extra
    return {
        "xyz": pc.xyz.astype(np.float32),
        "scale": np.exp(np.stack([e["scale_0"], e["scale_1"], e["scale_2"]], 1)).astype(np.float32),
        "rot": np.stack([e["rot_0"], e["rot_1"], e["rot_2"], e["rot_3"]], 1).astype(np.float32),
        "opacity": (1 / (1 + np.exp(-e["opacity"].astype(np.float64)))).astype(np.float32),
        "rgb": np.clip(
            (0.5 + SH_C0 * np.stack([e["f_dc_0"], e["f_dc_1"], e["f_dc_2"]], 1)) * 255, 0, 255
        ).astype(np.uint8),
    }


def write_splat(pc: PointCloud, path: str | Path) -> Path:
    """Legacy antimatter15 ``.splat``: 32 bytes/splat = pos f32x3, scale f32x3, rgba u8x4, rot u8x4."""
    a = gaussian_attributes(pc)
    rot = a["rot"] / np.linalg.norm(a["rot"], axis=1, keepdims=True)
    order = np.argsort(-(a["opacity"] * np.prod(a["scale"], axis=1)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for i in order:
            f.write(struct.pack("<3f", *a["xyz"][i]))
            f.write(struct.pack("<3f", *a["scale"][i]))
            f.write(struct.pack("<4B", *a["rgb"][i], int(np.clip(a["opacity"][i] * 255, 0, 255))))
            f.write(struct.pack("<4B", *np.clip(rot[i] * 128 + 128, 0, 255).astype(np.uint8)))
    return path


def write_spz(pc: PointCloud, path: str | Path) -> Path:
    """``.spz`` via the ``spz`` python package (Niantic). Optional dependency."""
    try:
        import spz  # type: ignore[import-not-found]
    except ImportError as e:
        raise MissingDependencyError("spz", "viz", ".spz export (pip install spz)") from e
    a = gaussian_attributes(pc)
    cloud = spz.GaussianCloud(
        positions=a["xyz"],
        scales=np.log(a["scale"]),
        rotations=a["rot"][:, [1, 2, 3, 0]],
        alphas=a["opacity"],
        colors=(a["rgb"].astype(np.float32) / 255 - 0.5) / SH_C0,
        sh=np.zeros((len(a["xyz"]), 0), dtype=np.float32),
    )
    spz.save_spz(cloud, str(path))
    return Path(path)


def export_run(ply: str | Path, out_dir: str | Path, fmt: str = "spz") -> Path:
    pc = read_ply(ply)
    out_dir = Path(out_dir)
    if fmt == "spz":
        return write_spz(pc, out_dir / (Path(ply).stem + ".spz"))
    if fmt == "splat":
        return write_splat(pc, out_dir / (Path(ply).stem + ".splat"))
    raise ContractError(f"unknown export format {fmt!r}")
