"""Hold-out group render metrics (§5 novel_view): PSNR, SSIM (numpy), LPIPS (torch, optional)."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.ndimage import gaussian_filter

from minegs.core.errors import MissingDependencyError


class RenderReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    test_groups: list[str]
    n_images: int
    psnr: float
    ssim: float
    lpips: float | None = None
    per_image: dict[str, dict[str, float]] = Field(default_factory=dict)
    claim: str = "render_quality"


def _to_float(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return img.astype(np.float64)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _to_float(a), _to_float(b)
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def ssim(
    a: np.ndarray, b: np.ndarray, sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03
) -> float:
    """Gaussian-window SSIM (Wang et al. 2004), averaged over channels."""
    a, b = _to_float(a), _to_float(b)
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]
    c1, c2 = k1**2, k2**2
    vals = []
    for c in range(a.shape[2]):
        x, y = a[..., c], b[..., c]
        mx, my = gaussian_filter(x, sigma), gaussian_filter(y, sigma)
        sxx = gaussian_filter(x * x, sigma) - mx**2
        syy = gaussian_filter(y * y, sigma) - my**2
        sxy = gaussian_filter(x * y, sigma) - mx * my
        s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))
        vals.append(float(s.mean()))
    return float(np.mean(vals))


def lpips(a: np.ndarray, b: np.ndarray, net: str = "alex") -> float:
    try:
        import lpips as _lpips
        import torch
    except ImportError as e:
        raise MissingDependencyError("lpips", "eval", "LPIPS") from e
    fn = _lpips.LPIPS(net=net, verbose=False)
    ta = torch.from_numpy(_to_float(a)).permute(2, 0, 1)[None].float() * 2 - 1
    tb = torch.from_numpy(_to_float(b)).permute(2, 0, 1)[None].float() * 2 - 1
    with torch.no_grad():
        return float(fn(ta, tb).item())


def evaluate_pairs(
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    test_groups: list[str],
    with_lpips: bool = False,
) -> RenderReport:
    per: dict[str, dict[str, float]] = {}
    for name, (render, gt) in pairs.items():
        m = {"psnr": psnr(render, gt), "ssim": ssim(render, gt)}
        if with_lpips:
            m["lpips"] = lpips(render, gt)
        per[name] = m
    n = len(per)
    return RenderReport(
        test_groups=test_groups,
        n_images=n,
        psnr=float(np.mean([m["psnr"] for m in per.values()])) if n else float("nan"),
        ssim=float(np.mean([m["ssim"] for m in per.values()])) if n else float("nan"),
        lpips=float(np.mean([m["lpips"] for m in per.values()])) if n and with_lpips else None,
        per_image=per,
    )
