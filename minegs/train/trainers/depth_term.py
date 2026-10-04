"""The MineGS depth term (Phase 4 AD-5): one formula, a numpy reference and the torch version.

Upstream gsplat v1.5.3 compares rendered expected depth with SfM track depth in disparity space
(``simple_trainer.py:688-706``)::

    L = depth_lambda * scene_scale * mean_m | rho(ED(u_m)) - 1 / z_m |,  rho(x) = 1/x if x > 0 else 0

That shape is kept, because disparity times ``scene_scale`` makes the value independent of
world scale, so ``depth_lambda`` means the same thing in metres as in upstream's normalised
frame. Three things are changed, each because upstream's form is wrong for an artifact that
carries its own pixel convention and confidence:

* **Pixel convention.** The rasteriser shades pixel ``j`` at ``j + 0.5``
  (``RasterizeToPixels3DGSFwd.cu:62-63``), the same convention as COLMAP and the artifact.
  Upstream samples with ``grid_sample(align_corners=True)`` at ``u / (W - 1)``, which treats
  ``u`` as an index, half a pixel off. Here the index is ``u - 0.5``. Samples whose index
  falls outside ``[0, W - 1] x [0, H - 1]`` of the trainer's image are dropped at load, and the
  term refuses a sample outside the image it is given, so the zero padding never contributes.
* **Weights.** Each sample's error is multiplied by its confidence and averaged over the
  image's samples that train (``sum w |..| / N``, ``N`` = samples with ``w > 0``). The weight is
  absolute: an image whose samples all have weight 0.001 counts a thousandth of one whose
  samples have weight 1. Nothing divides by a sum of weights, so a tiny weight cannot overflow
  a reciprocal. Weight 0 is not a sample (it is kept for audit and never trains), so ``N``
  excludes it. An image with no such sample contributes nothing; upstream would take the mean
  of an empty tensor, which is NaN.
* **Safe reciprocal.** ``torch.where(d > 0, 1 / d, 0)`` evaluates ``1 / 0`` in the masked branch
  and back-propagates ``0 * inf = NaN``. The denominator is made safe before it is inverted.

The images that do carry samples are averaged with equal weight, which is the mean over
images that upstream's per-step mean amounts to at ``batch_size = 1``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FORMULA = (
    "depth_lambda * scene_scale * mean_images[ sum_{m: w_m > 0} w_m |rho(ED(u_m', v_m')) - 1/z_m| "
    "/ #{m: w_m > 0} ], (u', v') = (u * fx_train / fx_full - 0.5, v * fy_train / fy_full - 0.5), "
    "rho(x) = 1/x (x > 0) else 0, bilinear ED at pixel index (align_corners=True)"
)


def to_training_index(
    u: np.ndarray, v: np.ndarray, K_full: np.ndarray, K_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Full-resolution COLMAP continuous pixels -> sampling index of the training image.

    Upstream builds the training K as ``diag(rx, ry, 1) @ K_full``: divided by ``data_factor``
    (``colmap.py:107``), then rescaled to the size of the image actually read
    (``colmap.py:262-273``). Focal length and principal point move together, so a continuous
    coordinate scales by the same ratio. A K whose principal point did not scale like its
    focal length was not produced that way, and is refused rather than approximated.
    """
    K_full = np.asarray(K_full, dtype=np.float64)
    K_train = np.asarray(K_train, dtype=np.float64)
    rx = K_train[0, 0] / K_full[0, 0]
    ry = K_train[1, 1] / K_full[1, 1]
    for got, want, what in (
        (K_train[0, 2], rx * K_full[0, 2], "cx"),
        (K_train[1, 2], ry * K_full[1, 2], "cy"),
    ):
        if abs(got - want) > 1e-6 * max(1.0, abs(want)):
            raise ValueError(
                f"training {what} {got} is not the full-resolution {what} scaled like the focal "
                f"length ({want}); the trainer's K was not a pure rescale of the dataset's"
            )
    return np.asarray(u, np.float64) * rx - 0.5, np.asarray(v, np.float64) * ry - 0.5


def in_domain(ui: np.ndarray, vi: np.ndarray, width: int, height: int) -> np.ndarray:
    """Samples whose bilinear footprint lies inside the image (no zero padding involved)."""
    return (ui >= 0) & (ui <= width - 1) & (vi >= 0) & (vi <= height - 1)


def bilinear(img: np.ndarray, ui: np.ndarray, vi: np.ndarray) -> np.ndarray:
    """``grid_sample(align_corners=True)`` at in-domain indices, in numpy."""
    H, W = img.shape
    x0 = np.clip(np.floor(ui).astype(np.int64), 0, W - 1)
    y0 = np.clip(np.floor(vi).astype(np.int64), 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    ax = ui - x0
    ay = vi - y0
    top = img[y0, x0] * (1 - ax) + img[y0, x1] * ax
    bot = img[y1, x0] * (1 - ax) + img[y1, x1] * ax
    return top * (1 - ay) + bot * ay


def image_term_reference(
    ed: np.ndarray, ui: np.ndarray, vi: np.ndarray, z: np.ndarray, w: np.ndarray
) -> float | None:
    """One image's weighted disparity error, or None when no sample has a positive weight."""
    w = np.asarray(w, dtype=np.float64)
    pos = w > 0
    if not pos.any():
        return None
    d = bilinear(np.asarray(ed, dtype=np.float64), np.asarray(ui)[pos], np.asarray(vi)[pos])
    disp = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    err = np.abs(disp - 1.0 / np.asarray(z, dtype=np.float64)[pos])
    return float((w[pos] * err).sum() / pos.sum())


def depth_term_reference(
    eds: list[np.ndarray],
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None],
    scene_scale: float,
    depth_lambda: float,
) -> tuple[float, int]:
    """The batch term and how many images contributed (numpy; the definition)."""
    terms = []
    for ed, s in zip(eds, samples, strict=True):
        if s is None:
            continue
        t = image_term_reference(ed, *s)
        if t is not None:
            terms.append(t)
    if not terms:
        return 0.0, 0
    return depth_lambda * scene_scale * float(np.mean(terms)), len(terms)


# ---------------------------------------------------------------- torch


def depth_term_torch(
    ed: Any,
    samples: list[tuple[Any, Any, Any, Any] | None],
    scene_scale: float,
    depth_lambda: float,
) -> tuple[Any, int]:
    """Same term on a rendered ``ED`` batch ``[B, H, W, 1]``; differentiable in ``ed``.

    ``samples[b]`` holds tensors ``(ui, vi, z, w)`` on ``ed``'s device, already restricted to
    the image domain, or None; samples with ``w = 0`` are skipped, as in the reference. A sample
    outside ``ed``'s own ``W x H`` is refused: ``grid_sample`` would mix its zero padding into
    the target. Returns ``(term, n_images)``; ``term`` is a 0-d tensor (0 when no image
    contributed, still attached to the graph so the caller's code path is one path).
    """
    import torch
    from torch.nn import functional as F  # noqa: N812 - torch's own convention

    B, H, W, _ = ed.shape
    terms = []
    for b in range(B):
        s = samples[b]
        if s is None:
            continue
        ui, vi, z, w = s
        pos = w > 0
        if not bool(pos.any()):
            continue
        ui, vi, z, w = ui[pos], vi[pos], z[pos], w[pos]
        if bool(((ui < 0) | (ui > W - 1) | (vi < 0) | (vi > H - 1)).any()):
            raise ValueError(
                f"a depth sample of batch item {b} lies outside the rendered {W}x{H} image; "
                "samples are filtered against the trainer's image size at load"
            )
        gx = ui / max(W - 1, 1) * 2 - 1
        gy = vi / max(H - 1, 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).to(ed.dtype)[None, :, None, :]  # [1, M, 1, 2]
        d = F.grid_sample(
            ed[b : b + 1].permute(0, 3, 1, 2), grid, mode="bilinear", align_corners=True
        )[0, 0, :, 0]
        valid = d > 0
        safe = torch.where(valid, d, torch.ones_like(d))
        disp = torch.where(valid, 1.0 / safe, torch.zeros_like(d))
        err = (disp - 1.0 / z.to(ed.dtype)).abs()
        terms.append((w.to(ed.dtype) * err).sum() / len(w))
    if not terms:
        return ed.sum() * 0.0, 0
    return depth_lambda * scene_scale * torch.stack(terms).mean(), len(terms)


_ADD_GRADIENT = None


def add_gradient(colors: Any, term: Any) -> Any:
    """Return ``colors`` unchanged, and make ``backward`` add ``d term`` exactly once.

    Upstream computes its loss inline in ``Runner.train`` with no hook, from the colours that
    ``rasterize_splats`` returns. Attaching the depth term to those colours through an identity
    whose backward also emits ``d term / d term = 1`` adds the term's gradient to every
    parameter it depends on. The upstream loss value, and every line of the upstream loop, stay
    as they are. Autograd calls a node's backward once with the summed upstream gradient, so
    the term enters once however many losses use the colours.
    """
    global _ADD_GRADIENT
    if _ADD_GRADIENT is None:
        import torch

        class _AddGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, c, t):  # type: ignore[override]
                ctx.t_shape = t.shape
                ctx.t_dtype = t.dtype
                return c.clone()

            @staticmethod
            def backward(ctx, grad):  # type: ignore[override]
                return grad, torch.ones(ctx.t_shape, dtype=ctx.t_dtype, device=grad.device)

        _ADD_GRADIENT = _AddGradient
    return _ADD_GRADIENT.apply(colors, term)


__all__ = [
    "FORMULA",
    "add_gradient",
    "bilinear",
    "depth_term_reference",
    "depth_term_torch",
    "image_term_reference",
    "in_domain",
    "to_training_index",
]
