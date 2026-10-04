"""Stand-in for gsplat v1.5.3 ``examples/simple_trainer.py`` (tests only; see README.md).

The shape is upstream's and the rasteriser is a toy. Where upstream refuses something, this
refuses it too (a text-only model, a missing ``images_<factor>/``), so a staging regression
shows up as the failure a real run would have hit.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from datasets.normalize import similarity_from_cameras  # noqa: F401  (upstream imports it too)
from gsplat.distributed import cli
from gsplat.strategy import DefaultStrategy, MCMCStrategy


@dataclass
class Config:
    disable_viewer: bool = False
    data_dir: str = "data/360_v2/garden"
    data_factor: int = 4
    result_dir: str = "results/garden"
    test_every: int = 8
    patch_size: int | None = None
    global_scale: float = 1.0
    normalize_world_space: bool = True
    camera_model: str = "pinhole"
    batch_size: int = 1
    steps_scaler: float = 1.0
    max_steps: int = 30_000
    eval_steps: list[int] = field(default_factory=lambda: [7_000, 30_000])
    save_steps: list[int] = field(default_factory=lambda: [7_000, 30_000])
    save_ply: bool = False
    ply_steps: list[int] = field(default_factory=lambda: [7_000, 30_000])
    init_type: str = "sfm"
    sh_degree: int = 3
    init_opa: float = 0.1
    init_scale: float = 1.0
    near_plane: float = 0.01
    far_plane: float = 1e10
    strategy: Any = field(default_factory=DefaultStrategy)
    packed: bool = False
    antialiased: bool = False
    opacity_reg: float = 0.0
    scale_reg: float = 0.0
    pose_opt: bool = False
    pose_noise: float = 0.0
    app_opt: bool = False
    use_bilateral_grid: bool = False
    depth_loss: bool = False
    depth_lambda: float = 1e-2
    with_ut: bool = False
    with_eval3d: bool = False
    use_fused_bilagrid: bool = False


def _parse(argv: list[str]) -> Config:
    """A tyro-shaped parser for the flags MineGS emits (sub-command first)."""
    sub, rest = argv[0], argv[1:]
    if sub == "default":
        cfg = Config(strategy=DefaultStrategy(verbose=True))
    elif sub == "mcmc":
        cfg = Config(
            init_opa=0.5,
            init_scale=0.1,
            opacity_reg=0.01,
            scale_reg=0.01,
            strategy=MCMCStrategy(verbose=True),
        )
    else:
        raise SystemExit(f"unknown sub-command {sub}")
    i = 0
    while i < len(rest):
        key = rest[i].lstrip("-").replace("-", "_")
        vals = []
        i += 1
        while i < len(rest) and not rest[i].startswith("--"):
            vals.append(rest[i])
            i += 1
        on = not key.startswith("no_")
        key = key if on else key[3:]
        target, name = (
            (cfg.strategy, key.split(".", 1)[1]) if key.startswith("strategy.") else (cfg, key)
        )
        if not hasattr(target, name):
            raise SystemExit(f"unrecognized option --{key}")
        old = getattr(target, name)
        if not vals:
            setattr(target, name, on)
        elif isinstance(old, list):
            setattr(target, name, [int(v) for v in vals])
        elif isinstance(old, bool):
            setattr(target, name, vals[0] in ("True", "true"))
        elif isinstance(old, int):
            setattr(target, name, int(vals[0]))
        elif isinstance(old, float):
            setattr(target, name, float(vals[0]))
        else:
            setattr(target, name, vals[0])
    return cfg


def _read_bin_cameras(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        out = {}
        for _ in range(n):
            cid, mid, w, h = struct.unpack("<IiQQ", f.read(24))
            k = {0: 3, 1: 4}[mid]
            params = struct.unpack(f"<{k}d", f.read(8 * k))
            fx, fy, cx, cy = (params[0], params[0], params[1], params[2]) if k == 3 else params
            out[cid] = (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), int(w), int(h))
    return out


class Parser:
    """What upstream's Parser does, for the parts MineGS depends on."""

    def __init__(self, data_dir: str, factor: int, normalize: bool, test_every: int):
        from minegs.ingest.common import colmap_io

        d = Path(data_dir)
        sparse = d / "sparse" / "0"
        if not (sparse / "images.bin").is_file():
            # the pinned pycolmap fork cannot parse text on Python 3 (C0 §2.3-1)
            raise ValueError("no binary COLMAP model: the pinned reader would fail on text")
        model = colmap_io.read_model_binary(sparse)
        cams = _read_bin_cameras(sparse / "cameras.bin")
        image_dir = d / ("images" if factor == 1 else f"images_{factor}")
        for p in (image_dir, d / "images"):
            if not p.exists():
                raise ValueError(f"Image folder {p} does not exist.")
        names = sorted(im.name for im in model.images.values())
        by_name = model.image_by_name()
        self.image_names = names
        self.camtoworlds = np.stack([by_name[n].world_from_cam.matrix() for n in names])
        self.camera_ids = [by_name[n].camera_id for n in names]
        self.Ks_dict, self.imsize_dict = {}, {}
        for cid, (K, w, h) in cams.items():
            K = K.copy()
            K[:2, :] /= factor
            self.Ks_dict[cid] = K
            self.imsize_dict[cid] = (w // factor, h // factor)
        # upstream's actual-image-size rescale (colmap.py:262-273)
        if factor > 1 and os.path.splitext(sorted(os.listdir(image_dir))[0])[1].lower() == ".jpg":
            image_dir = image_dir  # the resize-from-jpg branch is not reproduced here
        from PIL import Image

        first = next(p for p in sorted(image_dir.rglob("*")) if p.is_file())
        with Image.open(first) as im:
            aw, ah = im.size
        cw, ch = self.imsize_dict[self.camera_ids[0]]
        sw, sh = aw / cw, ah / ch
        for cid, K in self.Ks_dict.items():
            K[0, :] *= sw
            K[1, :] *= sh
            w, h = self.imsize_dict[cid]
            self.imsize_dict[cid] = (int(w * sw), int(h * sh))
        self.points = model.points_xyz().astype(np.float32)
        self.points_rgb = model.points_rgb()
        c = self.camtoworlds[:, :3, 3]
        self.scene_scale = float(np.max(np.linalg.norm(c - c.mean(axis=0), axis=1)))
        self.test_every = test_every


class Dataset:
    def __init__(self, parser: Parser, split: str = "train", patch_size=None, load_depths=False):
        self.parser = parser
        idx = np.arange(len(parser.image_names))
        self.indices = (
            idx[idx % parser.test_every != 0]
            if split == "train"
            else idx[idx % parser.test_every == 0]
        )

    def __len__(self) -> int:
        return len(self.indices)


class Runner:
    def __init__(self, local_rank, world_rank, world_size, cfg: Config) -> None:
        import torch

        self.cfg = cfg
        self.world_rank = world_rank
        self.world_size = world_size
        self.device = "cpu"
        os.makedirs(cfg.result_dir, exist_ok=True)
        self.parser = Parser(
            cfg.data_dir, cfg.data_factor, cfg.normalize_world_space, cfg.test_every
        )
        self.trainset = Dataset(self.parser, split="train", load_depths=cfg.depth_loss)
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        rgb = np.asarray(self.parser.points_rgb, dtype=np.float64) / 255.0
        if cfg.app_opt and (np.any(rgb <= 0) or np.any(rgb >= 1)):
            raise ValueError("logit(0 or 1) is infinite: init colours reach 0/255")
        self.splats = {
            "means": torch.tensor(self.parser.points, dtype=torch.float32, requires_grad=True),
            "opacities": torch.zeros(len(self.parser.points), requires_grad=True),
        }
        self.grad_log: list[float] = []

    def rasterize_splats(
        self,
        camtoworlds,
        Ks,
        width,
        height,
        masks=None,
        rasterize_mode=None,
        camera_model=None,
        **kwargs,
    ):
        """Toy render: colour from opacity, ED from the camera-z of the splat centroid."""
        import torch

        kwargs.pop("image_ids", None)
        kwargs.pop("sh_degree", None)
        mode = kwargs.pop("render_mode", "RGB")
        w2c = torch.linalg.inv(camtoworlds)[0]
        z = (self.splats["means"] @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]
        # positive by construction: a toy depth behind the camera would hit the contract's
        # rho(d <= 0) = 0 branch and carry no gradient, which is not what this fixture tests
        zc = torch.nn.functional.softplus(z.mean()) + 0.1
        ed = (zc + torch.zeros(1, height, width, 1)) * (
            1 + 0.01 * torch.linspace(0, 1, width).reshape(1, 1, width, 1)
        )
        rgb = torch.sigmoid(self.splats["opacities"].mean()) + torch.zeros(1, height, width, 3)
        renders = torch.cat([rgb, ed], dim=-1) if mode == "RGB+ED" else rgb
        return renders, torch.ones(1, height, width, 1), {}

    def train(self) -> None:
        import torch

        cfg = self.cfg
        with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
            yaml.dump(vars(cfg), f)
        for d in ("ckpts", "stats", "ply", "renders"):
            os.makedirs(f"{cfg.result_dir}/{d}", exist_ok=True)
        steps = cfg.max_steps
        for step in range(steps):
            item = step % len(self.trainset)
            index = int(self.trainset.indices[item])
            c2w = torch.tensor(self.parser.camtoworlds[index], dtype=torch.float32)[None]
            K = torch.tensor(self.parser.Ks_dict[self.parser.camera_ids[index]])[None]
            w, h = self.parser.imsize_dict[self.parser.camera_ids[index]]
            renders, _, _ = self.rasterize_splats(
                camtoworlds=c2w,
                Ks=K,
                width=w,
                height=h,
                sh_degree=cfg.sh_degree,
                image_ids=torch.tensor([item]),
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
            )
            colors = renders[..., 0:3] if renders.shape[-1] == 4 else renders
            loss = (colors - 0.25).abs().mean()
            loss.backward()
            g = self.splats["means"].grad
            self.grad_log.append(float(g.abs().sum()) if g is not None else 0.0)
            with torch.no_grad():
                for p in self.splats.values():
                    if p.grad is not None:
                        p -= 1e-6 * p.grad
                        p.grad = None
            if step == steps - 1:
                stats = {"mem": 0.5, "ellipse_time": 1.0, "num_GS": len(self.splats["means"])}
                Path(f"{cfg.result_dir}/stats/train_step{step:04d}_rank0.json").write_text(
                    json.dumps(stats)
                )
                torch.save(
                    {"step": step, "splats": {k: v.detach() for k, v in self.splats.items()}},
                    f"{cfg.result_dir}/ckpts/ckpt_{step}_rank0.pt",
                )
                if cfg.save_ply:
                    self._ply(f"{cfg.result_dir}/ply/point_cloud_{step}.ply")
                Path(f"{cfg.result_dir}/renders/val_step{step}_0000.png").write_bytes(
                    b"\x89PNG\r\n\x1a\n"
                )
        Path(f"{cfg.result_dir}/fake_grad_log.json").write_text(json.dumps(self.grad_log))

    def _ply(self, path: str) -> None:
        xyz = self.splats["means"].detach().numpy()
        props = [
            "x",
            "y",
            "z",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ]
        hdr = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
        hdr += "".join(f"property float {q}\n" for q in props) + "end_header\n"
        rows = b"".join(
            struct.pack(f"<{len(props)}f", *p, 0, 0, 0, 0, -2, -2, -2, 1, 0, 0, 0) for p in xyz
        )
        Path(path).write_bytes(hdr.encode() + rows)


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    runner = Runner(local_rank, world_rank, world_size, cfg)
    runner.train()


if __name__ == "__main__":
    cfg = _parse(sys.argv[1:])
    cli(main, cfg, verbose=True)
