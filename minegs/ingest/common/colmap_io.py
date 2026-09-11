"""COLMAP text model I/O (``cameras.txt``, ``images.txt``, ``points3D.txt``) plus the
COLMAP >= 3.12 / 4.0 rig files (``rigs.txt``, ``frames.txt``) used for 360 ring crops (§6.2).

Conventions: COLMAP stores ``cam_from_world`` as (qw,qx,qy,qz,tx,ty,tz).
All positions in a *dataset* are LOCAL_METRIC (§3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from minegs.core.errors import ContractError
from minegs.core.frames import SE3, quat_to_rotmat, rotmat_to_quat

CAMERA_MODELS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: list[float]

    def K(self) -> np.ndarray:
        if self.model == "SIMPLE_PINHOLE" or self.model == "SIMPLE_RADIAL":
            f, cx, cy = self.params[:3]
            fx = fy = f
        else:
            fx, fy, cx, cy = self.params[:4]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    @classmethod
    def pinhole(cls, id: int, K: np.ndarray, width: int, height: int) -> Camera:
        return cls(
            id,
            "PINHOLE",
            width,
            height,
            [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
        )


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # cam_from_world rotation (w,x,y,z)
    tvec: np.ndarray  # cam_from_world translation
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    point3D_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def cam_from_world(self) -> SE3:
        return SE3(quat_to_rotmat(self.qvec), self.tvec)

    @property
    def world_from_cam(self) -> SE3:
        return self.cam_from_world.inverse()

    @property
    def center(self) -> np.ndarray:
        return self.world_from_cam.t

    @classmethod
    def from_world_from_cam(
        cls, id: int, T_world_from_cam: SE3, camera_id: int, name: str
    ) -> Image:
        cfw = T_world_from_cam.inverse()
        return cls(id, rotmat_to_quat(cfw.R), cfw.t, camera_id, name)


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float = 0.0
    image_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    point2D_idxs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


@dataclass
class RigSensor:
    """One camera in a rig. ``sensor_from_rig`` is None for the reference sensor."""

    camera_id: int
    sensor_from_rig: SE3 | None = None


@dataclass
class Rig:
    id: int
    sensors: list[RigSensor]


@dataclass
class Frame:
    """One capture instant of a rig: which images belong to it. ``rig_from_world``."""

    id: int
    rig_id: int
    rig_from_world: SE3
    image_ids: list[tuple[int, int]]  # (camera_id, image_id)


@dataclass
class ColmapModel:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    points3D: dict[int, Point3D]
    rigs: dict[int, Rig] = field(default_factory=dict)
    frames: dict[int, Frame] = field(default_factory=dict)

    def image_by_name(self) -> dict[str, Image]:
        return {im.name: im for im in self.images.values()}

    def points_xyz(self) -> np.ndarray:
        if not self.points3D:
            return np.zeros((0, 3))
        return np.array([p.xyz for p in self.points3D.values()])

    def points_rgb(self) -> np.ndarray:
        if not self.points3D:
            return np.zeros((0, 3), dtype=np.uint8)
        return np.array([p.rgb for p in self.points3D.values()], dtype=np.uint8)

    def transformed(self, T_new_from_old: SE3) -> ColmapModel:
        """Re-express the whole model in another rigid frame (e.g. TLS_GLOBAL -> LOCAL_METRIC)."""
        images = {}
        for iid, im in self.images.items():
            wfc = T_new_from_old @ im.world_from_cam
            images[iid] = Image.from_world_from_cam(iid, wfc, im.camera_id, im.name)
            images[iid].xys, images[iid].point3D_ids = im.xys, im.point3D_ids
        pts = {
            pid: Point3D(
                pid, T_new_from_old.apply(p.xyz), p.rgb, p.error, p.image_ids, p.point2D_idxs
            )
            for pid, p in self.points3D.items()
        }
        frames = {
            fid: Frame(
                fid,
                fr.rig_id,
                (T_new_from_old @ fr.rig_from_world.inverse()).inverse(),
                fr.image_ids,
            )
            for fid, fr in self.frames.items()
        }
        return ColmapModel(dict(self.cameras), images, pts, dict(self.rigs), frames)


# ---------------------------------------------------------------- write


def _fmt(v: float) -> str:
    return f"{v:.10g}"


def write_model(model: ColmapModel, sparse_dir: str | Path) -> Path:
    d = Path(sparse_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "cameras.txt", "w") as f:
        f.write(
            "# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        )
        f.write(f"# Number of cameras: {len(model.cameras)}\n")
        for c in model.cameras.values():
            f.write(
                f"{c.id} {c.model} {c.width} {c.height} {' '.join(_fmt(p) for p in c.params)}\n"
            )
    with open(d / "images.txt", "w") as f:
        f.write(
            "# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        )
        f.write(f"# Number of images: {len(model.images)}\n")
        for im in model.images.values():
            q = " ".join(_fmt(v) for v in im.qvec)
            t = " ".join(_fmt(v) for v in im.tvec)
            f.write(f"{im.id} {q} {t} {im.camera_id} {im.name}\n")
            pts = " ".join(
                f"{_fmt(x)} {_fmt(y)} {int(pid)}"
                for (x, y), pid in zip(im.xys, im.point3D_ids, strict=True)
            )
            f.write(pts + "\n")
    with open(d / "points3D.txt", "w") as f:
        f.write(
            "# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        f.write(f"# Number of points: {len(model.points3D)}\n")
        for p in model.points3D.values():
            track = " ".join(
                f"{int(i)} {int(j)}" for i, j in zip(p.image_ids, p.point2D_idxs, strict=True)
            )
            f.write(
                f"{p.id} {_fmt(p.xyz[0])} {_fmt(p.xyz[1])} {_fmt(p.xyz[2])} {int(p.rgb[0])} {int(p.rgb[1])} {int(p.rgb[2])} {_fmt(p.error)} {track}\n"
            )
    if model.rigs:
        write_rigs(model, d)
    return d


def write_rigs(model: ColmapModel, sparse_dir: str | Path) -> None:
    """COLMAP >= 3.12 text rig format.

    rigs.txt:   RIG_ID, NUM_SENSORS, SENSOR_TYPE, SENSOR_ID, [SENSOR_TYPE, SENSOR_ID, HAS_POSE, QW, QX, QY, QZ, TX, TY, TZ] ...
    frames.txt: FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)
    """
    d = Path(sparse_dir)
    with open(d / "rigs.txt", "w") as f:
        f.write(
            "# Rig calib list with one line of data per calib:\n#   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, SENSORS[] as (SENSOR_TYPE, SENSOR_ID, HAS_POSE[, QW, QX, QY, QZ, TX, TY, TZ])\n"
        )
        f.write(f"# Number of rigs: {len(model.rigs)}\n")
        for rig in model.rigs.values():
            ref = [s for s in rig.sensors if s.sensor_from_rig is None]
            if len(ref) != 1:
                raise ContractError(f"rig {rig.id} must have exactly one reference sensor")
            parts = [str(rig.id), str(len(rig.sensors)), "CAMERA", str(ref[0].camera_id)]
            for s in rig.sensors:
                if s.sensor_from_rig is None:
                    continue
                q = s.sensor_from_rig.quat()
                t = s.sensor_from_rig.t
                parts += [
                    "CAMERA",
                    str(s.camera_id),
                    "1",
                    *(_fmt(v) for v in q),
                    *(_fmt(v) for v in t),
                ]
            f.write(" ".join(parts) + "\n")
    with open(d / "frames.txt", "w") as f:
        f.write(
            "# Frame list with one line of data per frame:\n#   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)\n"
        )
        f.write(f"# Number of frames: {len(model.frames)}\n")
        for fr in model.frames.values():
            q = fr.rig_from_world.quat()
            t = fr.rig_from_world.t
            parts = [
                str(fr.id),
                str(fr.rig_id),
                *(_fmt(v) for v in q),
                *(_fmt(v) for v in t),
                str(len(fr.image_ids)),
            ]
            for cam_id, img_id in fr.image_ids:
                parts += ["CAMERA", str(cam_id), str(img_id)]
            f.write(" ".join(parts) + "\n")


# ---------------------------------------------------------------- read


def _lines(path: Path) -> list[list[str]]:
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s.split())
    return out


def read_model(sparse_dir: str | Path) -> ColmapModel:
    d = Path(sparse_dir)
    for fn in ("cameras.txt", "images.txt", "points3D.txt"):
        if not (d / fn).exists():
            raise ContractError(
                f"{d}: missing {fn} (binary COLMAP models are not supported; export text)"
            )
    cameras: dict[int, Camera] = {}
    for tok in _lines(d / "cameras.txt"):
        cameras[int(tok[0])] = Camera(
            int(tok[0]), tok[1], int(tok[2]), int(tok[3]), [float(v) for v in tok[4:]]
        )
    images: dict[int, Image] = {}
    raw = [
        line.strip()
        for line in (d / "images.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    i = 0
    while i < len(raw):
        tok = raw[i].split()
        im = Image(
            int(tok[0]),
            np.array([float(v) for v in tok[1:5]]),
            np.array([float(v) for v in tok[5:8]]),
            int(tok[8]),
            tok[9],
        )
        i += 1
        if i < len(raw):
            pts = raw[i].split()
            if len(pts) % 3 == 0 and pts and not _looks_like_image_line(pts):
                arr = np.array([float(v) for v in pts]).reshape(-1, 3)
                im.xys = arr[:, :2]
                im.point3D_ids = arr[:, 2].astype(np.int64)
                i += 1
            elif not pts:
                i += 1
        images[im.id] = im
    points: dict[int, Point3D] = {}
    for tok in _lines(d / "points3D.txt"):
        pid = int(tok[0])
        track = (
            np.array([int(v) for v in tok[8:]]).reshape(-1, 2)
            if len(tok) > 8
            else np.zeros((0, 2), dtype=np.int64)
        )
        points[pid] = Point3D(
            pid,
            np.array([float(v) for v in tok[1:4]]),
            np.array([int(v) for v in tok[4:7]], dtype=np.uint8),
            float(tok[7]),
            track[:, 0],
            track[:, 1],
        )
    model = ColmapModel(cameras, images, points)
    if (d / "rigs.txt").exists():
        _read_rigs(model, d)
    return model


def _looks_like_image_line(tok: list[str]) -> bool:
    return len(tok) == 10 and not tok[9].replace(".", "").lstrip("-").isdigit()


def _read_rigs(model: ColmapModel, d: Path) -> None:
    for tok in _lines(d / "rigs.txt"):
        rig_id, n = int(tok[0]), int(tok[1])
        sensors = [RigSensor(int(tok[3]), None)]
        k = 4
        while len(sensors) < n:
            cam_id = int(tok[k + 1])
            has_pose = tok[k + 2] == "1"
            if has_pose:
                q = [float(v) for v in tok[k + 3 : k + 7]]
                t = [float(v) for v in tok[k + 7 : k + 10]]
                sensors.append(RigSensor(cam_id, SE3.from_quat_t(q, t)))
                k += 10
            else:
                sensors.append(RigSensor(cam_id, SE3.identity()))
                k += 3
        model.rigs[rig_id] = Rig(rig_id, sensors)
    if (d / "frames.txt").exists():
        for tok in _lines(d / "frames.txt"):
            fid, rid = int(tok[0]), int(tok[1])
            q = [float(v) for v in tok[2:6]]
            t = [float(v) for v in tok[6:9]]
            n = int(tok[9])
            ids = [(int(tok[10 + 3 * j + 1]), int(tok[10 + 3 * j + 2])) for j in range(n)]
            model.frames[fid] = Frame(fid, rid, SE3.from_quat_t(q, t), ids)


# ---------------------------------------------------------------- helpers


def project(
    K: np.ndarray, cam_from_world: SE3, xyz_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole projection. Returns (uv (N,2), depth (N,))."""
    pc = cam_from_world.apply(xyz_world)
    z = pc[:, 2]
    uv = (pc[:, :2] / np.where(np.abs(z) < 1e-12, 1e-12, z)[:, None]) * np.array(
        [K[0, 0], K[1, 1]]
    ) + np.array([K[0, 2], K[1, 2]])
    return uv, z


def visible_mask(
    uv: np.ndarray, depth: np.ndarray, width: int, height: int, near: float = 0.05
) -> np.ndarray:
    return (
        (depth > near)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
